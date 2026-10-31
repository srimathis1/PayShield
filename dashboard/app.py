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
