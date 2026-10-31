import json
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "paysim dataset.csv"
ETL_SCRIPT = PROJECT_ROOT / "src" / "transaction_pipeline.py"
PREDICTION_SCRIPT = PROJECT_ROOT / "src" / "predict_transaction.py"
CURATED_PATH = PROJECT_ROOT / "output" / "curated_transactions"
QUARANTINE_PATH = PROJECT_ROOT / "output" / "quarantine_transactions"
ML_PATH = PROJECT_ROOT / "output" / "ml"
MODEL_PATH = ML_PATH / "random_forest_model"
METRICS_PATH = ML_PATH / "metrics.json"
LOG_PATH = PROJECT_ROOT / "logs"
HISTORY_PATH = LOG_PATH / "pipeline_history.csv"

SPARK_HOME = Path(
    os.environ.get(
        "SPARK_HOME",
        str(Path.home() / "Downloads" / "spark-4.0.0-bin-hadoop3"),
    )
)
SPARK_SUBMIT = SPARK_HOME / "bin" / "spark-submit"

LOG_PATH.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="PayShield | Transaction Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --page: #f4f7fb;
        --panel: #ffffff;
        --ink: #17243b;
        --muted: #62718a;
        --line: #e0e7f0;
        --blue: #2563eb;
        --teal: #0f9f95;
    }
    .stApp { background: var(--page); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244,247,251,.96); }
    [data-testid="stSidebar"] {
        background: #101c32;
        border-right: 1px solid #263650;
    }
    [data-testid="stSidebar"] * { color: #e8eef8 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #bdcbe0 !important;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    h1 { font-size: 2.05rem !important; font-weight: 760 !important; }
    h2 { font-size: 1.45rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.1rem !important; font-weight: 680 !important; }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 17px 18px;
        box-shadow: 0 3px 12px rgba(20,40,80,.035);
        min-width: 0;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    [data-testid="stMetricValue"] {
        color: #175cd3 !important;
        font-size: clamp(1.2rem, 2vw, 1.9rem) !important;
        overflow-wrap: anywhere;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 10px;
        min-height: 43px;
        font-weight: 650;
    }
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: var(--blue);
        border-color: var(--blue);
        color: white;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 11px; }
    hr { border-color: var(--line); }
    .hero {
        background: linear-gradient(120deg, #10264b 0%, #174ea6 65%, #0f9f95 130%);
        padding: 27px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }
    .hero h1, .hero p, .hero span { color: white !important; }
    .hero p { margin-bottom: 0; opacity: .88; }
    .eyebrow {
        font-size: .74rem; font-weight: 750; letter-spacing: .13em;
        text-transform: uppercase; opacity: .85; margin-bottom: 9px;
    }
    .section-note { color: #62718a; font-size: .9rem; }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    .stMarkdown p {
        color: #24344d;
    }
    input, textarea {
        color: #17243b !important;
    }
    [data-baseweb="select"] * {
        color: #17243b;
    }
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e0e7f0;
        border-radius: 16px;
        padding: 18px 20px 8px 20px;
    }
    .result-good {
        background: #e8f7ef; color: #146c43; padding: 16px 18px;
        border: 1px solid #b8e5cb; border-radius: 12px; font-weight: 700;
    }
    .result-review {
        background: #fff4e5; color: #9a4d00; padding: 16px 18px;
        border: 1px solid #f5d09b; border-radius: 12px; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Metrics JSON must contain an object.")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Could not read model metrics: {exc}")
        return None


def load_csv_sample(path, rows=50_000):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, nrows=rows)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return None


def count_parquet_files(path):
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*.parquet") if item.is_file())


def append_history(status, duration, message):
    entry = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "duration_seconds": round(float(duration), 2),
        "message": str(message)[:500],
    }])
    try:
        entry.to_csv(
            HISTORY_PATH,
            mode="a" if HISTORY_PATH.exists() else "w",
            header=not HISTORY_PATH.exists(),
            index=False,
        )
    except OSError as exc:
        st.warning(f"Could not write run history: {exc}")


def spark_environment():
    env = os.environ.copy()
    env.pop("HADOOP_CONF_DIR", None)
    env.pop("YARN_CONF_DIR", None)
    return env


def run_etl():
    if not DATA_PATH.exists():
        st.error(f"Input dataset not found: {DATA_PATH}")
        return
    if not ETL_SCRIPT.exists():
        st.error(f"ETL script not found: {ETL_SCRIPT}")
        return
    if not SPARK_SUBMIT.exists():
        st.error(f"spark-submit not found: {SPARK_SUBMIT}")
        return

    env = spark_environment()
    env["PAYSHIELD_INPUT_PATH"] = str(DATA_PATH)
    env["PAYSHIELD_QUARANTINE_PATH"] = str(QUARANTINE_PATH)
    env["PAYSHIELD_OUTPUT_PATH"] = str(CURATED_PATH)

    log_file = LOG_PATH / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    started = time.time()
    try:
        with st.spinner("Running Spark ETL. This may take several minutes..."):
            with log_file.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    [str(SPARK_SUBMIT), str(ETL_SCRIPT)],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3600,
                    check=False,
                )
        duration = time.time() - started
        if completed.returncode == 0:
            append_history("SUCCESS", duration, f"Log: {log_file.name}")
            st.success(f"ETL completed in {duration:.1f} seconds.")
        else:
            append_history("FAILED", duration, f"Exit code {completed.returncode}")
            st.error(f"ETL failed (exit code {completed.returncode}). See log below.")
        with st.expander("Latest ETL log", expanded=completed.returncode != 0):
            st.code(log_file.read_text(encoding="utf-8", errors="replace")[-12000:])
    except subprocess.TimeoutExpired:
        duration = time.time() - started
        append_history("TIMEOUT", duration, "Exceeded 1-hour timeout")
        st.error("ETL exceeded the one-hour timeout.")
    except Exception as exc:
        duration = time.time() - started
        append_history("ERROR", duration, str(exc))
        st.error(f"Could not run ETL: {exc}")


def run_prediction(transaction):
    if not MODEL_PATH.exists():
        return None, f"Saved model not found: {MODEL_PATH}"
    if not PREDICTION_SCRIPT.exists():
        return None, f"Prediction script not found: {PREDICTION_SCRIPT}"

    try:
        completed = subprocess.run(
            [sys.executable, str(PREDICTION_SCRIPT)],
            input=json.dumps(transaction),
            text=True,
            capture_output=True,
            cwd=str(PROJECT_ROOT),
            env=spark_environment(),
            timeout=240,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "Prediction timed out while starting or running Spark."
    except Exception as exc:
        return None, f"Could not start prediction process: {exc}"

    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                if completed.returncode != 0 or not payload.get("success"):
                    return None, payload.get("error", completed.stderr[-4000:])
                return payload, None
            except json.JSONDecodeError:
                continue

    details = (completed.stderr or completed.stdout or "No output returned.")[-5000:]
    return None, f"Prediction did not return valid JSON.\n{details}"

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 0 18px">
            <div style="font-size:1.8rem;font-weight:800">🛡️ PayShield</div>
            <div style="font-size:.72rem;letter-spacing:.13em;color:#a9c3e8">
                TRANSACTION INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Data Ingestion",
            "Pipeline Execution",
            "Transaction Explorer",
            "Fraud Prediction",
            "Run History",
            "AWS Architecture",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Environment**")
    st.caption("Engine · Apache Spark")
    st.caption("Storage · Parquet")
    st.caption("Mode · Local")
    st.divider()
    st.caption("Portfolio demonstration · Not a production fraud system")


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Financial data engineering</div>
            <h1>PayShield Transaction Intelligence</h1>
            <p>Validate transaction data, build curated datasets, and explore model-based fraud risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_bytes = DATA_PATH.stat().st_size if DATA_PATH.exists() else None
    curated_parts = count_parquet_files(CURATED_PATH)
    metrics = load_metrics()

    a, b, c, d = st.columns(4)
    a.metric("Raw dataset", "Available" if DATA_PATH.exists() else "Missing")
    b.metric(
        "CSV size",
        f"{dataset_bytes / (1024**2):.1f} MB" if dataset_bytes else "—",
    )
    c.metric("Curated Parquet parts", f"{curated_parts:,}")
    d.metric("Saved ML model", "Available" if MODEL_PATH.exists() else "Missing")

    st.subheader("Pipeline at a glance")
    stages = st.columns(4)

    stage_info = [
        ("01", "Ingest", "Read the PaySim CSV"),
        ("02", "Validate", "Check required fields and records"),
        ("03", "Curate", "Write partitioned Parquet data"),
        ("04", "Score", "Apply saved Spark MLlib model"),
    ]

    for column, (number, title, description) in zip(stages, stage_info):
        with column:
            st.markdown(f"### {number} · {title}")
            st.write(description)

    if metrics:
        st.subheader("Saved model evaluation")
        x, y, z = st.columns(3)

        x.metric("ROC-AUC", f"{metrics.get('area_under_roc', 0):.4f}")
        y.metric("PR-AUC", f"{metrics.get('area_under_pr', 0):.4f}")
        z.metric("Test records", f"{int(metrics.get('test_records', 0)):,}")

    st.info(
        "Use **Fraud Prediction** to score an entered transaction "
        "with the saved model."
    )


# ============================================================
# DATA INGESTION
# ============================================================
elif page == "Data Ingestion":
    st.title("Data Ingestion")
    st.write(
        "Inspect a sample of the raw PaySim CSV. "
        "The full file is not loaded into dashboard memory."
    )
    st.code(str(DATA_PATH))

    if not DATA_PATH.exists():
        st.error("CSV not found. Check the path above.")
    else:
        st.success("Raw dataset found.")

        sample_rows = st.selectbox(
            "Sample size",
            [1000, 10000, 50000, 100000],
            index=2,
        )

        data = load_csv_sample(DATA_PATH, rows=int(sample_rows))

        if data is not None:
            a, b, c = st.columns(3)

            a.metric("Rows sampled", f"{len(data):,}")
            b.metric("Columns", f"{len(data.columns):,}")
            c.metric(
                "Missing cells in sample",
                f"{int(data.isna().sum().sum()):,}",
            )

            with st.expander("Column schema", expanded=True):
                schema = pd.DataFrame({
                    "Column": data.columns,
                    "Type": [str(data[col].dtype) for col in data.columns],
                    "Missing": [
                        int(data[col].isna().sum())
                        for col in data.columns
                    ],
                })

                st.dataframe(
                    schema,
                    hide_index=True,
                    width="stretch",
                )

            st.subheader("Sample records")

            st.dataframe(
                data.head(100),
                hide_index=True,
                width="stretch",
            )

            st.caption(
                "Counts shown on this page describe the selected sample, "
                "not the full dataset."
            )


# ============================================================
# PIPELINE EXECUTION
# ============================================================
elif page == "Pipeline Execution":
    st.title("Pipeline Execution")
    st.write(
        "Run the existing local Spark ETL script. "
        "This may take several minutes."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Input CSV**")
        st.code(str(DATA_PATH))

        st.markdown("**ETL script**")
        st.code(str(ETL_SCRIPT))

    with right:
        st.markdown("**Curated output**")
        st.code(str(CURATED_PATH))

        st.markdown("**Spark submit**")
        st.code(str(SPARK_SUBMIT))

    if st.button(
        "Run PySpark ETL",
        type="primary",
        width="stretch",
    ):
        run_etl()

    st.divider()

    a, b = st.columns(2)

    a.metric(
        "Curated Parquet files",
        f"{count_parquet_files(CURATED_PATH):,}",
    )

    b.metric(
        "Quarantine output exists",
        "Yes" if QUARANTINE_PATH.exists() else "No",
    )

    st.caption(
        "This button runs ETL only; it does not retrain the fraud model."
    )


# ============================================================
# TRANSACTION EXPLORER
# ============================================================
elif page == "Transaction Explorer":
    st.title("Transaction Explorer")
    st.write("Explore and filter a bounded sample of the raw CSV.")

    data = load_csv_sample(DATA_PATH, rows=100_000)

    if data is None:
        st.warning("Raw CSV not found or could not be read.")

    else:
        filtered = data.copy()

        c1, c2, c3 = st.columns(3)

        if "type" in filtered:
            types = sorted(
                filtered["type"].dropna().unique().tolist()
            )

            with c1:
                selected_types = st.multiselect(
                    "Transaction type",
                    types,
                    default=types,
                )

            filtered = filtered[
                filtered["type"].isin(selected_types)
            ]

        if "amount" in filtered:
            lo = float(filtered["amount"].min())
            hi = float(filtered["amount"].max())

            if lo < hi:
                with c2:
                    amount_range = st.slider(
                        "Amount range",
                        min_value=lo,
                        max_value=hi,
                        value=(lo, hi),
                    )

                filtered = filtered[
                    filtered["amount"].between(*amount_range)
                ]

        if "isFraud" in filtered:
            with c3:
                label_filter = st.selectbox(
                    "Label",
                    ["All", "Fraud only", "Non-fraud only"],
                )

            if label_filter == "Fraud only":
                filtered = filtered[
                    filtered["isFraud"] == 1
                ]

            elif label_filter == "Non-fraud only":
                filtered = filtered[
                    filtered["isFraud"] == 0
                ]

        a, b = st.columns(2)

        a.metric("Rows in loaded sample", f"{len(data):,}")
        b.metric("Rows after filters", f"{len(filtered):,}")

        st.dataframe(
            filtered.head(500),
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download filtered rows",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="payshield_filtered_sample.csv",
            mime="text/csv",
        )

        st.caption(
            "Filters apply to the first 100,000 CSV rows only."
        )
