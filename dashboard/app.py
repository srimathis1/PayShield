import json
import os
import sys
import subprocess
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
PREDICTION_SCRIPT = PROJECT_ROOT / "src" / "predict_transaction.py"
CURATED_PATH = PROJECT_ROOT / "output" / "curated_transactions"
ML_PATH = PROJECT_ROOT / "output" / "ml"
MODEL_PATH = ML_PATH / "random_forest_model"
METRICS_PATH = ML_PATH / "metrics.json"
LOG_PATH = PROJECT_ROOT / "logs"
HISTORY_PATH = LOG_PATH / "pipeline_history.csv"


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
    # PayShield uses local files. Prevent inherited HDFS/YARN config from
    # accidentally redirecting local paths to HDFS.
    env.pop("HADOOP_CONF_DIR", None)
    env.pop("YARN_CONF_DIR", None)
    return env



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

    # The prediction script emits one JSON object, while Spark may emit
    # startup/log messages around it. Search stdout from the end.
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
            "Transaction Explorer",
            "Fraud Prediction",
            "Run History",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Environment**")
    st.caption("Inference · Saved Spark MLlib model")
    st.caption("Mode · Local prediction service")
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
    b.metric("CSV size", f"{dataset_bytes / (1024**2):.1f} MB" if dataset_bytes else "—")
    c.metric("Curated Parquet parts", f"{curated_parts:,}")
    d.metric("Saved ML model", "Available" if MODEL_PATH.exists() else "Missing")

    st.subheader("Pipeline at a glance")
    stages = st.columns(4)
    stage_info = [
        ("01", "Ingest", "Read the PaySim CSV"),
        ("02", "Validate", "Check required fields and records"),
        ("03", "Store", "Use the completed curated dataset"),
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
    st.info("Use **Fraud Prediction** to score an entered transaction with the saved model.")


# ============================================================
# DATA INGESTION
# ============================================================
elif page == "Data Ingestion":
    st.title("Data Ingestion")
    st.write("Inspect a sample of the raw PaySim CSV. The full file is not loaded into dashboard memory.")
    st.code(str(DATA_PATH))

    if not DATA_PATH.exists():
        st.error("CSV not found. Check the path above.")
    else:
        st.success("Raw dataset found.")
        sample_rows = st.selectbox("Sample size", [1000, 10000, 50000, 100000], index=2)
        data = load_csv_sample(DATA_PATH, rows=int(sample_rows))
        if data is not None:
            a, b, c = st.columns(3)
            a.metric("Rows sampled", f"{len(data):,}")
            b.metric("Columns", f"{len(data.columns):,}")
            c.metric("Missing cells in sample", f"{int(data.isna().sum().sum()):,}")
            with st.expander("Column schema", expanded=True):
                schema = pd.DataFrame({
                    "Column": data.columns,
                    "Type": [str(data[col].dtype) for col in data.columns],
                    "Missing": [int(data[col].isna().sum()) for col in data.columns],
                })
                st.dataframe(schema, hide_index=True, width="stretch")
            st.subheader("Sample records")
            st.dataframe(data.head(100), hide_index=True, width="stretch")
            st.caption("Counts shown on this page describe the selected sample, not the full dataset.")



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
            types = sorted(filtered["type"].dropna().unique().tolist())
            with c1:
                selected_types = st.multiselect("Transaction type", types, default=types)
            filtered = filtered[filtered["type"].isin(selected_types)]

        if "amount" in filtered:
            lo = float(filtered["amount"].min())
            hi = float(filtered["amount"].max())
            if lo < hi:
                with c2:
                    amount_range = st.slider("Amount range", min_value=lo, max_value=hi, value=(lo, hi))
                filtered = filtered[filtered["amount"].between(*amount_range)]

        if "isFraud" in filtered:
            with c3:
                label_filter = st.selectbox("Label", ["All", "Fraud only", "Non-fraud only"])
            if label_filter == "Fraud only":
                filtered = filtered[filtered["isFraud"] == 1]
            elif label_filter == "Non-fraud only":
                filtered = filtered[filtered["isFraud"] == 0]

        a, b = st.columns(2)
        a.metric("Rows in loaded sample", f"{len(data):,}")
        b.metric("Rows after filters", f"{len(filtered):,}")
        st.dataframe(filtered.head(500), hide_index=True, width="stretch")
        st.download_button(
            "Download filtered rows",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="payshield_filtered_sample.csv",
            mime="text/csv",
        )
        st.caption("Filters apply to the first 100,000 CSV rows only.")


# ============================================================
# FRAUD PREDICTION
# ============================================================
elif page == "Fraud Prediction":
    st.title("Transaction Risk Scoring")
    st.write("Enter a transaction to obtain a score from the persisted Spark MLlib pipeline.")
    st.info(
        "Each submission invokes the saved model. This is a model-based risk estimate, "
        "not proof of fraud. The first request can take time while Spark starts."
    )

    if not MODEL_PATH.exists():
        st.error(f"Saved model directory is missing: {MODEL_PATH}")
        st.stop()
    if not PREDICTION_SCRIPT.exists():
        st.error(f"Prediction script is missing: {PREDICTION_SCRIPT}")
        st.stop()

    with st.form("predict_transaction_form", clear_on_submit=False):
        st.subheader("Transaction features")
        left, right = st.columns(2)
        with left:
            tx_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"])
            amount = st.number_input("Amount", min_value=0.0, value=25000.0, step=100.0, format="%.2f")
            origin_before = st.number_input("Origin balance before", min_value=0.0, value=30000.0, step=100.0, format="%.2f")
            origin_after = st.number_input("Origin balance after", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
        with right:
            destination_before = st.number_input("Destination balance before", min_value=0.0, value=10000.0, step=100.0, format="%.2f")
            destination_after = st.number_input("Destination balance after", min_value=0.0, value=35000.0, step=100.0, format="%.2f")
            threshold_pct = st.slider("Flag for review at probability", min_value=1, max_value=99, value=50, format="%d%%")

        submitted = st.form_submit_button("Score transaction", type="primary", width="stretch")

    if submitted:
        tx = {
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": origin_before,
            "newbalanceOrig": origin_after,
            "oldbalanceDest": destination_before,
            "newbalanceDest": destination_after,
        }

        with st.spinner("Loading saved model and scoring transaction..."):
            prediction, error = run_prediction(tx)

        if error:
            st.error("Prediction failed — the model did not return a valid result.")
            st.markdown("**Debug details**")
            st.code(error)
            st.caption(
                "No prediction has been fabricated. Check that the saved Spark "
                "PipelineModel is readable and that local Spark can start."
            )
        else:
            probability = float(prediction["fraud_probability"])
            model_class = int(prediction.get("model_prediction", 0))
            diagnostics = prediction.get("balance_diagnostics", {})
            sender_mismatch = bool(diagnostics.get("sender_balance_mismatch", False))
            receiver_mismatch = bool(diagnostics.get("receiver_balance_mismatch", False))
            balance_mismatch = bool(
                diagnostics.get(
                    "any_balance_mismatch",
                    sender_mismatch or receiver_mismatch,
                )
            )

            if not 0.0 <= probability <= 1.0:
                st.error(f"Model returned invalid probability: {probability}")
            else:
                threshold = threshold_pct / 100.0
                threshold_flag = probability >= threshold
                model_flag = model_class == 1
                requires_review = model_flag or threshold_flag or balance_mismatch

                if (model_flag or threshold_flag) and balance_mismatch:
                    risk_level = "HIGH"
                    risk_status = "MODEL_ALERT_AND_BALANCE_MISMATCH"
                    recommendation = "Prioritize this transaction for human review."
                elif model_flag:
                    risk_level = "HIGH"
                    risk_status = "ML_FRAUD_ALERT"
                    recommendation = "The classifier flagged this transaction; investigate it."
                elif threshold_flag:
                    risk_level = "HIGH"
                    risk_status = "PROBABILITY_THRESHOLD_EXCEEDED"
                    recommendation = "Fraud probability meets the selected review threshold."
                elif balance_mismatch:
                    risk_level = "REVIEW"
                    risk_status = "BALANCE_MISMATCH"
                    recommendation = (
                        "Investigate the balance inconsistency even though the model "
                        "did not flag fraud."
                    )
                else:
                    risk_level = "LOW"
                    risk_status = "NO_RULE_ALERT"
                    recommendation = (
                        "No configured alert was triggered. This does not guarantee "
                        "that the transaction is legitimate."
                    )

                st.subheader("Risk assessment")
                risk_col, status_col, review_col = st.columns(3)
                risk_col.metric("Combined risk level", risk_level)
                status_col.metric("Risk status", risk_status.replace("_", " ").title())
                review_col.metric("Human review required", "YES" if requires_review else "NO")
                if risk_level == "HIGH":
                    st.error(f"HIGH RISK — {recommendation}")
                elif risk_level == "REVIEW":
                    st.warning(f"REVIEW REQUIRED — {recommendation}")
                else:
                    st.success(f"LOW ALERT LEVEL — {recommendation}")

                st.subheader("Model output")
                a, b, c = st.columns(3)
                a.metric("Fraud probability", f"{probability:.2%}")
                b.metric("Selected review threshold", f"{threshold:.0%}")
                c.metric(
                    "Threshold decision",
                    "Flag for review" if threshold_flag else "Below threshold",
                )
                st.progress(probability, text="Model-estimated fraud probability")
                st.write(
                    f"**Random Forest class:** "
                    f"{'Fraud-like (1)' if model_flag else 'Not flagged (0)'}"
                )
                st.caption(
                    "The classifier's class, probability-threshold rule, and balance "
                    "checks are separate signals. A review alert is not proof of fraud."
                )

                st.subheader("Balance consistency checks")
                if balance_mismatch:
                    st.warning(
                        "At least one balance mismatch was detected. Review the "
                        "transaction; the mismatch alone does not establish fraud."
                    )
                else:
                    st.success("No balance mismatch detected by these checks.")

                d1, d2 = st.columns(2)
                with d1:
                    st.metric(
                        "Sender balance difference",
                        f"{float(diagnostics.get('sender_balance_difference', 0)):,.2f}",
                        help="Old sender balance minus amount minus new sender balance. A non-zero value indicates an inconsistency.",
                    )
                    st.caption(f"Mismatch detected: {'Yes' if sender_mismatch else 'No'}")
                with d2:
                    st.metric(
                        "Receiver balance difference",
                        f"{float(diagnostics.get('receiver_balance_difference', 0)):,.2f}",
                        help="Old receiver balance plus amount minus new receiver balance. A non-zero value indicates an inconsistency.",
                    )
                    st.caption(f"Mismatch detected: {'Yes' if receiver_mismatch else 'No'}")

                st.subheader("Submitted transaction")
                feature_df = pd.DataFrame(
                    [{
                        "Feature": name,
                        "Value": str(value),
                    } for name, value in tx.items()]
                )
                st.dataframe(feature_df, hide_index=True, width="stretch")

                st.download_button(
                    "Download prediction report (JSON)",
                    data=json.dumps({
                        "transaction": tx,
                        "model_prediction": model_class,
                        "fraud_probability": probability,
                        "threshold": threshold,
                        "risk_assessment": {
                            "risk_level": risk_level,
                            "risk_status": risk_status,
                            "requires_review": requires_review,
                            "recommendation": recommendation,
                        },
                        "balance_diagnostics": diagnostics,
                    }, indent=2).encode("utf-8"),
                    file_name="payshield_prediction_report.json",
                    mime="application/json",
                )

    st.divider()
    st.subheader("Offline model evaluation")
    metrics = load_metrics()
    if metrics:
        a, b, c = st.columns(3)
        a.metric("ROC-AUC", f"{float(metrics.get('area_under_roc', 0)):.4f}")
        b.metric("PR-AUC", f"{float(metrics.get('area_under_pr', 0)):.4f}")
        c.metric(
            "Test records",
            f"{int(metrics.get('test_records', 0)):,}",
        )
        st.caption("These are saved test-set metrics, separate from the individual prediction above.")

        threshold_rows = metrics.get("threshold_analysis", [])
        if threshold_rows:
            threshold_df = pd.DataFrame(threshold_rows)
            needed = {"threshold", "precision", "recall", "f1"}
            if needed.issubset(threshold_df.columns):
                st.subheader("Threshold comparison (exploratory)")
                st.warning(
                    "These threshold values were calculated on the test set. "
                    "For final reporting, choose a threshold using validation data "
                    "and evaluate once on an untouched test set."
                )
                chart_df = threshold_df.melt(
                    id_vars="threshold",
                    value_vars=["precision", "recall", "f1"],
                    var_name="Metric",
                    value_name="Score (%)",
                )
                chart = alt.Chart(chart_df).mark_line(point=True).encode(
                    x=alt.X("threshold:Q", title="Threshold", axis=alt.Axis(format="%")),
                    y=alt.Y("Score (%):Q", title="Score (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Metric:N", title="Metric"),
                    tooltip=[
                        alt.Tooltip("threshold:Q", format=".0%"),
                        "Metric:N",
                        alt.Tooltip("Score (%):Q", format=".2f"),
                    ],
                ).properties(height=300)
                st.altair_chart(chart, width="stretch")
    else:
        st.caption("No metrics.json found; individual prediction can still work if the saved model exists.")


# ============================================================
# RUN HISTORY
# ============================================================
elif page == "Run History":
    st.title("Pipeline Run History")
    if not HISTORY_PATH.exists():
        st.info("No dashboard-initiated pipeline runs have been recorded yet.")
    else:
        try:
            history = pd.read_csv(HISTORY_PATH)
            if history.empty:
                st.info("Run history is empty.")
            else:
                a, b, c = st.columns(3)
                a.metric("Total runs", f"{len(history):,}")
                b.metric("Successful", f"{int((history['status'] == 'SUCCESS').sum()):,}")
                c.metric("Other outcomes", f"{int((history['status'] != 'SUCCESS').sum()):,}")
                st.dataframe(history.iloc[::-1], hide_index=True, width="stretch")
                st.download_button(
                    "Download run history",
                    data=history.to_csv(index=False).encode("utf-8"),
                    file_name="payshield_run_history.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not read run history: {exc}")

    st.subheader("Pipeline logs")
    logs = sorted(LOG_PATH.glob("pipeline_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        selected_log = st.selectbox("Choose log", logs, format_func=lambda p: p.name)
        if selected_log:
            st.code(selected_log.read_text(encoding="utf-8", errors="replace")[-12000:])
    else:
        st.caption("No pipeline logs found.")



# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("PayShield · Apache Spark · PySpark · Spark MLlib · Streamlit | Portfolio demonstration")

