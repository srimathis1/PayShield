import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PAYSHIELD — TRANSACTION INTELLIGENCE DASHBOARD
# Uses the already-trained Spark MLlib model. No retraining.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "output" / "ml" / "random_forest_model"
PREDICT_SCRIPT = ROOT / "src" / "predict_transaction.py"
METRICS_PATH = ROOT / "output" / "ml" / "metrics.json"
RAW_DATA_PATH = ROOT / "data" / "raw" / "paysim dataset.csv"

TOTAL_TRANSACTIONS = 6_362_620
FRAUD_LABELLED = 8_213
TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]

st.set_page_config(
    page_title="PayShield | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Styling ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080f1d;
        --panel: #101c2e;
        --panel2: #14233a;
        --border: rgba(148, 163, 184, .16);
        --text: #edf4ff;
        --muted: #91a4bd;
        --cyan: #39d5d2;
        --blue: #5b8cff;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 75% 0%, #142b49 0%, #080f1d 48%); color: var(--text); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #0b1627; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] * { color: #dce8f8; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #91a4bd; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f3f7ff !important; letter-spacing: -.035em; }
    h1 { font-size: 2.25rem !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.08rem !important; }
    p, label, .stMarkdown { color: #d6e2f2; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #14243a, #101b2d);
        border: 1px solid var(--border); border-radius: 16px;
        padding: 17px 19px; box-shadow: 0 10px 28px rgba(0,0,0,.12);
    }
    [data-testid="stMetricLabel"] { color: #9db0c9 !important; font-size: .82rem !important; }
    [data-testid="stMetricValue"] { color: #f3f7ff !important; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stMetricDelta"] { font-size: .78rem; }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 10px; min-height: 44px; font-weight: 700;
        border: 1px solid rgba(91,140,255,.45);
    }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(90deg, #3978f6, #20aaa9); color: white; border: 0;
    }
    div[data-testid="stForm"] {
        background: #101c2e; border: 1px solid var(--border);
        border-radius: 17px; padding: 20px 22px 10px;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div {
        background: #14243a; border-color: #2b405d;
    }
    input, textarea { color: #edf4ff !important; }
    [data-testid="stWidgetLabel"] p { color: #c3d2e6 !important; }
    hr { border-color: var(--border); }
    .hero {
        padding: 30px 32px; border-radius: 21px; margin: 5px 0 23px;
        background: linear-gradient(115deg, rgba(27,57,99,.96), rgba(17,112,123,.86));
        border: 1px solid rgba(128,206,231,.22);
        box-shadow: 0 18px 45px rgba(0,0,0,.18);
    }
    .hero-kicker { color: #8ee8e4; font-size: .72rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; }
    .hero-title { color: white; font: 700 2.15rem 'Space Grotesk', sans-serif; margin: 9px 0 7px; }
    .hero-copy { color: #d2e4f5; font-size: .98rem; max-width: 760px; margin: 0; }
    .pill {
        display: inline-block; padding: 6px 10px; border-radius: 99px;
        background: rgba(57,213,210,.12); color: #73e6df;
        border: 1px solid rgba(57,213,210,.25); font-size: .75rem; font-weight: 700;
    }
    .section-label { color: #7f9ab9; font-size: .72rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; margin: 5px 0 9px; }
    .info-card {
        background: linear-gradient(145deg, #14243a, #101b2d);
        border: 1px solid var(--border); border-radius: 15px;
        padding: 18px 19px; min-height: 122px;
    }
    .info-number { color: #70deda; font: 700 1.45rem 'Space Grotesk', sans-serif; }
    .info-title { color: #f0f5fc; font-weight: 700; margin: 7px 0 4px; }
    .info-copy { color: #91a4bd; font-size: .83rem; line-height: 1.45; }
    .risk-high { background: rgba(239,68,68,.12); border: 1px solid rgba(248,113,113,.4); color: #ff9b9b; padding: 15px 17px; border-radius: 12px; font-weight: 700; }
    .risk-review { background: rgba(245,158,11,.12); border: 1px solid rgba(251,191,36,.4); color: #ffd17a; padding: 15px 17px; border-radius: 12px; font-weight: 700; }
    .risk-low { background: rgba(34,197,94,.11); border: 1px solid rgba(74,222,128,.35); color: #8be6ad; padding: 15px 17px; border-radius: 12px; font-weight: 700; }
    .small-muted { color: #91a4bd; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Helpers ----------------------------
@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner="Loading a sample of transactions...")
def load_sample(rows=30000):
    if not RAW_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RAW_DATA_PATH, nrows=rows)


def spark_environment():
    env = os.environ.copy()
    env.pop("HADOOP_CONF_DIR", None)
    env.pop("YARN_CONF_DIR", None)
    return env


def predict(transaction):
    if not MODEL_PATH.exists():
        return None, f"Saved model directory is missing: {MODEL_PATH}"
    if not PREDICT_SCRIPT.exists():
        return None, f"Prediction script is missing: {PREDICT_SCRIPT}"

    try:
        result = subprocess.run(
            [sys.executable, str(PREDICT_SCRIPT)],
            input=json.dumps(transaction),
            text=True,
            capture_output=True,
            cwd=str(ROOT),
            env=spark_environment(),
            timeout=240,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "Prediction timed out while Spark was starting or scoring."
    except Exception as exc:
        return None, f"Could not start prediction process: {exc}"

    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                if result.returncode != 0 or not payload.get("success"):
                    return None, payload.get("error", result.stderr[-4000:])
                return payload, None
            except json.JSONDecodeError:
                pass

    details = (result.stderr or result.stdout or "No output returned.")[-5000:]
    return None, f"Prediction did not return valid JSON.\n{details}"


def money(value):
    return f"₹{float(value):,.2f}"


# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:13px 4px 18px">
          <div style="font:700 1.65rem 'Space Grotesk';color:#f5f8ff">🛡️ PayShield</div>
          <div style="font-size:.68rem;letter-spacing:.17em;color:#6edbd7;font-weight:800">PAYMENT RISK INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Fraud Prediction", "Transaction Explorer", "Model Insights"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.markdown(
        f'<span class="pill">{"MODEL FILE FOUND" if MODEL_PATH.exists() else "MODEL FILE MISSING"}</span>',
        unsafe_allow_html=True,
    )
    st.caption("Inference · Saved Spark MLlib model")
    st.caption("Training · Not performed by this dashboard")
    st.divider()
    st.caption("PayShield is a portfolio screening demo, not a production fraud decision system.")

# ---------------------------- Overview ----------------------------
if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Financial data engineering · Machine learning</div>
          <div class="hero-title">See risk before it moves.</div>
          <p class="hero-copy">PayShield combines transaction data quality checks, a curated PaySim dataset,
          and a saved Random Forest model to surface transactions that may need closer review.</p>
          <div style="margin-top:18px"><span class="pill">SPARK MLlib</span>&nbsp;&nbsp;<span class="pill">BATCH-TRAINED MODEL</span>&nbsp;&nbsp;<span class="pill">ON-DEMAND SCORING</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Dataset snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{TOTAL_TRANSACTIONS:,}")
    c2.metric("Fraud-labelled", f"{FRAUD_LABELLED:,}")
    c3.metric("Fraud prevalence", f"{FRAUD_LABELLED / TOTAL_TRANSACTIONS:.3%}")
    c4.metric("Saved model", "Available" if MODEL_PATH.exists() else "Not found")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">How PayShield works</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [
        ("01", "INGEST", "PaySim transaction records are read from the raw CSV source."),
        ("02", "VALIDATE", "Data quality rules check required fields and transaction values."),
        ("03", "CURATE", "Validated records are stored in compressed Parquet format."),
        ("04", "SCORE", "The persisted Random Forest pipeline estimates fraud probability."),
    ]
    for col, (num, title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="info-card"><div class="info-number">{num}</div><div class="info-title">{title}</div><div class="info-copy">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Risk signals")
        st.write("PayShield keeps the model output and rule-based checks distinct:")
        st.write("- **Model class:** whether the saved classifier flags the transaction.")
        st.write("- **Fraud probability:** the model's estimated probability for the fraud class.")
        st.write("- **Review threshold:** a user-selected probability threshold.")
        st.write("- **Balance checks:** arithmetic consistency of sender and receiver balances.")
    with right:
        st.markdown("### Quick action")
        st.markdown(
            '<div class="info-card"><div class="info-title">Score a transaction</div>'
            '<div class="info-copy">Enter transaction details and receive the saved model output, '
            'balance diagnostics, and a review recommendation.</div></div>',
            unsafe_allow_html=True,
        )

    st.caption("Dataset figures refer to the PaySim dataset used in this project. Risk alerts are screening signals, not proof of fraud.")

# ---------------------------- Fraud Prediction ----------------------------
elif page == "Fraud Prediction":
    st.markdown('<div class="section-label">LIVE INFERENCE · SAVED MODEL</div>', unsafe_allow_html=True)
    st.title("Fraud risk assessment")
    st.write("Submit a transaction to score it with the persisted Spark MLlib Random Forest pipeline.")
    st.info("The dashboard loads the existing model for inference only. It does not retrain the model. The first request may take longer while Spark starts.")

    if not MODEL_PATH.exists():
        st.error(f"Saved model not found: {MODEL_PATH}")
        st.stop()
    if not PREDICT_SCRIPT.exists():
        st.error(f"Prediction script not found: {PREDICT_SCRIPT}")
        st.stop()

    with st.form("transaction_form"):
        st.markdown("### Transaction details")
        col_a, col_b = st.columns(2)
        with col_a:
            tx_type = st.selectbox("Transaction type", TRANSACTION_TYPES, index=0)
            amount = st.number_input("Transaction amount", min_value=0.0, value=25000.0, step=500.0)
            origin_before = st.number_input("Sender balance before", min_value=0.0, value=30000.0, step=500.0)
            origin_after = st.number_input("Sender balance after", min_value=0.0, value=5000.0, step=500.0)
        with col_b:
            destination_before = st.number_input("Receiver balance before", min_value=0.0, value=10000.0, step=500.0)
            destination_after = st.number_input("Receiver balance after", min_value=0.0, value=35000.0, step=500.0)
            threshold_pct = st.slider("Review threshold", min_value=1, max_value=99, value=20, format="%d%%")
            st.caption("A lower threshold flags more transactions for review.")
        submitted = st.form_submit_button("🛡️  Analyse transaction", type="primary", use_container_width=True)

    if submitted:
        tx = {
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": origin_before,
            "newbalanceOrig": origin_after,
            "oldbalanceDest": destination_before,
            "newbalanceDest": destination_after,
        }
        with st.spinner("Starting Spark and scoring with the saved model..."):
            output, error = predict(tx)

        if error:
            st.error("Prediction could not be completed.")
            with st.expander("Technical details"):
                st.code(error)
        else:
            probability = float(output["fraud_probability"])
            model_class = int(output.get("model_prediction", 0))
            diagnostics = output.get("balance_diagnostics", {})
            sender_bad = bool(diagnostics.get("sender_balance_mismatch", False))
            receiver_bad = bool(diagnostics.get("receiver_balance_mismatch", False))
            any_bad = bool(diagnostics.get("any_balance_mismatch", sender_bad or receiver_bad))
            threshold = threshold_pct / 100
            threshold_hit = probability >= threshold
            model_hit = model_class == 1
            review = model_hit or threshold_hit or any_bad

            if (model_hit or threshold_hit) and any_bad:
                level, status, advice = "HIGH", "Model / threshold alert + balance mismatch", "Prioritise this transaction for human investigation."
            elif model_hit:
                level, status, advice = "HIGH", "Classifier flagged this transaction", "Investigate the model alert."
            elif threshold_hit:
                level, status, advice = "HIGH", "Probability threshold exceeded", "The estimated probability meets your review threshold."
            elif any_bad:
                level, status, advice = "REVIEW", "Balance inconsistency detected", "Check the balance movement even though the model did not flag it."
            else:
                level, status, advice = "LOW", "No configured alert triggered", "No current alert was triggered; this does not guarantee legitimacy."

            st.markdown("---")
            st.markdown("### Assessment result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Combined risk", level)
            r2.metric("Fraud probability", f"{probability:.2%}")
            r3.metric("Human review", "RECOMMENDED" if review else "NOT TRIGGERED")

            css_class = "risk-high" if level == "HIGH" else "risk-review" if level == "REVIEW" else "risk-low"
            st.markdown(f'<div class="{css_class}">{level} RISK · {advice}</div>', unsafe_allow_html=True)
            st.write(f"**Signal:** {status}")
            st.progress(min(max(probability, 0.0), 1.0), text="Model-estimated fraud probability")
            st.caption(f"Selected threshold: {threshold:.0%} · Classifier output: {'Flagged (1)' if model_hit else 'Not flagged (0)'}")

            st.markdown("### Balance diagnostics")
            b1, b2 = st.columns(2)
            with b1:
                st.metric("Sender balance difference", money(diagnostics.get("sender_balance_difference", 0)))
                st.caption("Mismatch: " + ("Detected" if sender_bad else "Not detected"))
            with b2:
                st.metric("Receiver balance difference", money(diagnostics.get("receiver_balance_difference", 0)))
                st.caption("Mismatch: " + ("Detected" if receiver_bad else "Not detected"))

            st.markdown("### Submitted transaction")
            features = pd.DataFrame([{"Feature": k, "Value": str(v)} for k, v in tx.items()])
            st.dataframe(features, hide_index=True, use_container_width=True)

            report = {
                "transaction": tx,
                "model_prediction": model_class,
                "fraud_probability": probability,
                "review_threshold": threshold,
                "risk_level": level,
                "risk_status": status,
                "human_review_recommended": review,
                "recommendation": advice,
                "balance_diagnostics": diagnostics,
            }
            st.download_button(
                "Download assessment (JSON)",
                data=json.dumps(report, indent=2).encode("utf-8"),
                file_name="payshield_assessment.json",
                mime="application/json",
            )
            st.caption("This is a decision-support demonstration. A risk score or balance mismatch is not a definitive fraud determination.")

# ---------------------------- Transaction Explorer ----------------------------
elif page == "Transaction Explorer":
    st.markdown('<div class="section-label">DATA EXPLORATION</div>', unsafe_allow_html=True)
    st.title("Transaction explorer")
    st.write("Explore a bounded sample of the original PaySim data. Filters apply only to the loaded sample.")
    if not RAW_DATA_PATH.exists():
        st.warning("The raw PaySim CSV is not present in this deployment. Model inference can still work without it.")
    else:
        sample_size = st.selectbox("Rows to load", [5000, 10000, 30000, 50000], index=2)
        try:
            data = load_sample(sample_size)
            if data.empty:
                st.warning("No transaction rows were loaded.")
            else:
                a, b, c = st.columns(3)
                a.metric("Loaded sample", f"{len(data):,}")
                b.metric("Columns", f"{len(data.columns)}")
                fraud_count = int(data["isFraud"].sum()) if "isFraud" in data else 0
                c.metric("Fraud-labelled in sample", f"{fraud_count:,}")

                x, y = st.columns(2)
                if "type" in data:
                    options = sorted(data["type"].dropna().unique().tolist())
                    chosen = x.multiselect("Transaction types", options, default=options)
                    data = data[data["type"].isin(chosen)]
                if "isFraud" in data:
                    label = y.selectbox("Label filter", ["All records", "Fraud-labelled", "Non-fraud"])
                    if label == "Fraud-labelled":
                        data = data[data["isFraud"] == 1]
                    elif label == "Non-fraud":
                        data = data[data["isFraud"] == 0]

                st.metric("Rows after filters", f"{len(data):,}")
                st.dataframe(data.head(500), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download filtered sample",
                    data=data.to_csv(index=False).encode("utf-8"),
                    file_name="payshield_transaction_sample.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not read the sample: {exc}")

# ---------------------------- Model Insights ----------------------------
elif page == "Model Insights":
    st.markdown('<div class="section-label">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    st.title("Model insights")
    st.write("Saved evaluation metrics from the model training run. These values are not recalculated by the dashboard.")
    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics.json file was found. The prediction page may still work if the saved model exists.")
    else:
        a, b, c = st.columns(3)
        if "area_under_roc" in metrics:
            a.metric("ROC-AUC", f"{float(metrics['area_under_roc']):.4f}")
        if "area_under_pr" in metrics:
            b.metric("PR-AUC", f"{float(metrics['area_under_pr']):.4f}")
        if "test_records" in metrics:
            c.metric("Test records", f"{int(metrics['test_records']):,}")

        rows = metrics.get("threshold_analysis", [])
        if rows:
            df = pd.DataFrame(rows)
            st.markdown("### Threshold comparison")
            available = [x for x in ["threshold", "precision", "recall", "f1"] if x in df.columns]
            if available:
                st.dataframe(df[available], hide_index=True, use_container_width=True)
                st.caption("Use validation data to select a threshold; reserve the test set for final evaluation.")
        with st.expander("View saved metrics JSON"):
            st.json(metrics)

# ---------------------------- Footer ----------------------------
st.divider()
st.markdown(
    '<div class="small-muted">PAYSHIELD · APACHE SPARK · PYSPARK · SPARK MLlib · STREAMLIT<br>Portfolio demonstration · Risk alerts support review and do not prove fraud.</div>',
    unsafe_allow_html=True,
)
