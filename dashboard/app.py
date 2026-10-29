import os
import re
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path.home() / "payshield"
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "output"
LOG_DIR = PROJECT_DIR / "logs"

PIPELINE_FILE = PROJECT_DIR / "src" / "transaction_pipeline.py"

SPARK_SUBMIT = (
    PROJECT_DIR.parent
    / "Downloads"
    / "spark-4.0.0-bin-hadoop3"
    / "bin"
    / "spark-submit"
)

DEFAULT_DATASET = DATA_DIR / "paysim dataset.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="PayShield | Data Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. ENTERPRISE DARK UI
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --background: #080d19;
        --panel: #111b2d;
        --panel-light: #17253b;
        --border: #293d58;
        --text: #e7eef9;
        --muted: #9aadc7;
        --blue: #38a5ff;
        --green: #35d6a0;
        --red: #ff6878;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    header[data-testid="stHeader"] {
        background: rgba(8, 13, 25, 0.95);
    }

    .block-container {
        max-width: 1550px;
        padding: 2rem 2.5rem 3rem;
    }

    /* Global text contrast */

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    label,
    legend {
        color: var(--text) !important;
    }

    h1, h2, h3 {
        color: #f2f7ff !important;
        letter-spacing: -0.4px;
    }

    h1 {
        font-size: 2.45rem !important;
        font-weight: 800 !important;
    }

    h2 {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 1.12rem !important;
        font-weight: 650 !important;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #0d1729;
        border-right: 1px solid #263751;
    }

    section[data-testid="stSidebar"] * {
        color: #dce7f7;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap: 8px;
    }

    /* Header */

    .brand-label {
        color: #7dbdff;
        font-size: 0.73rem;
        font-weight: 750;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    .page-description {
        color: #9aadc7;
        font-size: 0.95rem;
        margin-bottom: 25px;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 20px;
        background: #10372f;
        border: 1px solid #236b57;
        color: #6ee7b7;
        font-size: 0.78rem;
        font-weight: 650;
    }

    /* Metric cards */

    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #17263d,
            #101a2b
        );
        border: 1px solid #2b405d;
        border-radius: 14px;
        padding: 19px 20px;
        min-height: 120px;
    }

    [data-testid="stMetricLabel"] {
        color: #aabbd2 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f1f7ff !important;
        font-weight: 750;
    }

    [data-testid="stMetricDelta"] {
        color: #62e6b8 !important;
    }

    /* Buttons */

    .stButton > button,
    .stDownloadButton > button {
        background: #1479e5;
        color: #ffffff !important;
        border: 1px solid #328ff0;
        border-radius: 9px;
        font-weight: 650;
        min-height: 42px;
        transition: all 0.15s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #0965c4;
        border-color: #7dc2ff;
        color: #ffffff !important;
    }

    /* Panels */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #111b2d;
        border: 1px solid #293d58;
        border-radius: 12px;
    }

    [data-testid="stStatusWidget"] {
        background: #111b2d;
        border: 1px solid #293d58;
        border-radius: 12px;
    }

    /* Upload area */

    [data-testid="stFileUploader"] {
        background: #111b2d;
        border: 1px dashed #456080;
        border-radius: 12px;
        padding: 12px;
    }

    /* Data tables */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #293d58;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Inputs */

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: #111b2d;
        border-color: #344966;
        color: #f1f7ff;
    }

    input, textarea {
        color: #f1f7ff !important;
    }

    /* Messages */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #293d58;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .footer {
        color: #7489a6;
        font-size: 0.8rem;
        text-align: center;
        padding-top: 20px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def format_number(value):
    """Format a numeric value for the UI."""

    if value is None:
        return "N/A"

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def read_preview(file_path, rows=20):
    """Read a small CSV preview without loading the entire dataset."""

    return pd.read_csv(
        file_path,
        nrows=rows,
        low_memory=False,
    )


def get_history():
    """Load pipeline history."""

    history_path = LOG_DIR / "run_history.csv"

    if not history_path.exists():
        return pd.DataFrame(
            columns=[
                "run_id",
                "status",
                "duration_seconds",
                "total_records",
                "valid_records",
                "invalid_records",
            ]
        )

    try:
        return pd.read_csv(history_path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def save_history(record):
    """Append a completed or failed pipeline run."""

    history_path = LOG_DIR / "run_history.csv"

    row = pd.DataFrame([record])

    if history_path.exists():
        row.to_csv(
            history_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        row.to_csv(
            history_path,
            index=False,
        )


def parse_count(pattern, log_text):
    """Extract a record count from pipeline output."""

    match = re.search(pattern, log_text)

    if not match:
        return None

    return int(match.group(1).replace(",", ""))


# ============================================================
# 4. SESSION STATE
# ============================================================

if "selected_input" not in st.session_state:
    st.session_state.selected_input = None

if "last_run" not in st.session_state:
    st.session_state.last_run = None


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 10px 0 22px;">
            <div style="font-size: 2rem;">🛡️ PayShield</div>
            <div style="
                color: #7dbdff;
                font-size: 0.72rem;
                letter-spacing: 1.5px;
                margin-top: 5px;
            ">
                DATA INTELLIGENCE
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
            "Run History",
            "AWS Architecture",
        ],
    )

    st.divider()

    st.markdown("### Environment")

    st.caption("Execution engine: Apache Spark")
    st.caption("Processing: Local")
    st.caption("Storage: Parquet")

    st.divider()

    st.caption("PayShield · Data Engineering Platform")


# ============================================================
# 6. PAGE HEADER
# ============================================================

st.markdown(
    '<div class="brand-label">'
    'FINANCIAL TRANSACTION INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([4, 1])

with header_left:
    st.title("PayShield")

    st.markdown(
        '<div class="page-description">'
        'Interactive financial data engineering workbench'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div style="text-align:right; padding-top:25px;">'
        '<span class="status-pill">LOCAL MODE</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# 7. OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Platform Overview")

    st.write(
        "Manage transaction ingestion, execute PySpark ETL jobs, "
        "inspect results, and monitor pipeline performance."
    )

    history = get_history()

    total_runs = len(history)

    successful_runs = 0
    failed_runs = 0

    if not history.empty and "status" in history.columns:
        successful_runs = int(
            (history["status"] == "SUCCESS").sum()
        )

        failed_runs = int(
            (history["status"] == "FAILED").sum()
        )

    latest = (
        history.iloc[-1].to_dict()
        if not history.empty
        else None
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Total Pipeline Runs",
        format_number(total_runs),
    )

    b.metric(
        "Successful Runs",
        format_number(successful_runs),
    )

    c.metric(
        "Failed Runs",
        format_number(failed_runs),
    )

    d.metric(
        "Latest Duration",
        (
            f"{latest['duration_seconds']} sec"
            if latest is not None
            else "N/A"
        ),
    )

    st.markdown("### Quick Actions")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("#### 📥 Data Ingestion")

            st.write(
                "Upload transaction data and inspect its schema."
            )

            if st.button(
                "Open Data Ingestion",
                use_container_width=True,
            ):
                st.session_state.navigate_to = "Data Ingestion"
                st.rerun()

    with right:
        with st.container(border=True):
            st.markdown("#### ⚙️ Pipeline Execution")

            st.write(
                "Execute your existing Spark pipeline and "
                "inspect actual processing results."
            )

            if st.button(
                "Open Pipeline Execution",
                use_container_width=True,
            ):
                st.session_state.navigate_to = "Pipeline Execution"
                st.rerun()

    st.markdown("### Latest Pipeline Run")

    if latest is not None:

        x1, x2, x3 = st.columns(3)

        x1.metric("Run ID", str(latest.get("run_id", "N/A")))

        x2.metric("Status", str(latest.get("status", "N/A")))

        x3.metric(
            "Input Records",
            format_number(latest.get("total_records")),
        )

        st.dataframe(
            history.tail(5).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No pipeline runs recorded yet. "
            "Open Data Ingestion, select a dataset, and run the pipeline."
        )


# ============================================================
# 8. DATA INGESTION
# ============================================================

elif page == "Data Ingestion":

    st.subheader("Data Ingestion")

    st.write(
        "Choose your existing PaySim dataset or upload a transaction CSV."
    )

    source = st.radio(
        "Data source",
        [
            "Existing PaySim dataset",
            "Upload a CSV",
        ],
        horizontal=True,
    )

    selected_file = None

    if source == "Existing PaySim dataset":

        if DEFAULT_DATASET.exists():

            selected_file = DEFAULT_DATASET

            size_mb = (
                DEFAULT_DATASET.stat().st_size
                / (1024 * 1024)
            )

            x1, x2 = st.columns(2)

            x1.metric(
                "Dataset Size",
                f"{size_mb:.2f} MB",
            )

            x2.metric(
                "File",
                DEFAULT_DATASET.name,
            )

            st.success("Existing dataset found.")

        else:
            st.warning(
                f"Dataset not found at {DEFAULT_DATASET}"
            )

    else:

        uploaded = st.file_uploader(
            "Upload transaction CSV",
            type=["csv"],
            help=(
                "The file is saved to your local project data directory."
            ),
        )

        if uploaded is not None:

            safe_name = Path(uploaded.name).name

            selected_file = DATA_DIR / safe_name

            if st.button("Save Uploaded Dataset"):

                with open(selected_file, "wb") as output_file:
                    output_file.write(uploaded.getbuffer())

                st.session_state.selected_input = str(
                    selected_file
                )

                st.success(
                    f"Dataset saved to {selected_file}"
                )

                st.rerun()

    if selected_file is not None:

        st.session_state.selected_input = str(selected_file)

        st.markdown("### Dataset Information")

        size_mb = (
            selected_file.stat().st_size
            / (1024 * 1024)
        )

        a, b, c = st.columns(3)

        a.metric("File Name", selected_file.name)

        b.metric("File Size", f"{size_mb:.2f} MB")

        c.metric("Preview Rows", "20 maximum")

        try:

            preview = read_preview(selected_file)

            st.markdown("### Data Preview")

            st.dataframe(
                preview,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Detected Schema")

            schema = pd.DataFrame(
                {
                    "Column": preview.columns,
                    "Detected Type": [
                        str(preview[column].dtype)
                        for column in preview.columns
                    ],
                    "Missing in Preview": [
                        int(preview[column].isna().sum())
                        for column in preview.columns
                    ],
                }
            )

            st.dataframe(
                schema,
                use_container_width=True,
                hide_index=True,
            )

            st.info(
                "The schema and missing-value counts above are based "
                "on the preview. Full-dataset validation occurs "
                "when the Spark pipeline executes."
            )

        except Exception as exc:
            st.error(f"Unable to preview the file: {exc}")


# ============================================================
# 9. PIPELINE EXECUTION
# ============================================================

elif page == "Pipeline Execution":

    st.subheader("Pipeline Execution")

    st.write(
        "Execute the real PySpark ETL pipeline in your local environment."
    )

    selected_input = st.session_state.selected_input

    if not selected_input and DEFAULT_DATASET.exists():
        selected_input = str(DEFAULT_DATASET)

    if not selected_input or not Path(selected_input).exists():

        st.warning(
            "No input dataset is selected. "
            "Open Data Ingestion first."
        )

        st.stop()

    st.markdown("### Selected Input")

    st.code(selected_input)

    st.markdown("### Processing Workflow")

    workflow = [
        ("01", "Read source data", "CSV ingestion"),
        ("02", "Validate records", "Quality checks"),
        ("03", "Quarantine failures", "Invalid-record isolation"),
        ("04", "Engineer features", "Derived transaction columns"),
        ("05", "Run Spark SQL", "Transaction analytics"),
        ("06", "Write curated data", "Partitioned Parquet"),
    ]

    for row_start in range(0, len(workflow), 3):

        columns = st.columns(3)

        for offset, item in enumerate(
            workflow[row_start:row_start + 3]
        ):

            number, title, description = item

            with columns[offset]:

                with st.container(border=True):

                    st.markdown(
                        f"### {number} · {title}"
                    )

                    st.caption(description)

    st.markdown("### Output Configuration")

    output_name = st.text_input(
        "Run output directory name",
        value=datetime.now().strftime("run_%Y%m%d_%H%M%S"),
    )

    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):

        st.error(
            "Use only letters, numbers, underscores, and hyphens."
        )

        st.stop()

    if not SPARK_SUBMIT.exists():

        st.error(
            f"Spark submit was not found: {SPARK_SUBMIT}"
        )

        st.info(
            "Update SPARK_SUBMIT at the top of dashboard/app.py "
            "if your Spark installation is in another directory."
        )

        st.stop()

    if not PIPELINE_FILE.exists():

        st.error(
            f"Pipeline file not found: {PIPELINE_FILE}"
        )

        st.stop()

    st.markdown("### Execute")

    st.write(
        "This operation runs your existing Spark script. "
        "Processing the full dataset may take some time."
    )

    if st.button(
        "▶ Run PySpark Pipeline",
        type="primary",
        use_container_width=True,
    ):

        run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        run_root = OUTPUT_DIR / "runs" / output_name

        curated_path = run_root / "curated_transactions"

        quarantine_path = run_root / "quarantine_transactions"

        run_root.mkdir(parents=True, exist_ok=True)

        log_path = LOG_DIR / f"run_{run_id}.log"

        env = os.environ.copy()

        env["PAYSHIELD_INPUT_PATH"] = selected_input

        env["PAYSHIELD_OUTPUT_PATH"] = str(curated_path)

        env["PAYSHIELD_QUARANTINE_PATH"] = str(
            quarantine_path
        )

        env["PYSPARK_PYTHON"] = sys.executable

        env["PYSPARK_DRIVER_PYTHON"] = sys.executable

        collected_lines = []

        start_time = time.perf_counter()

        st.markdown("### Live Execution Monitor")

        status = st.status(
            "Starting Spark job...",
            expanded=True,
        )

        log_display = st.empty()

        return_code = None

        try:

            with subprocess.Popen(
                [
                    str(SPARK_SUBMIT),
                    str(PIPELINE_FILE),
                ],
                cwd=PROJECT_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process:

                for line in process.stdout:

                    collected_lines.append(line)

                    log_display.code(
                        "".join(collected_lines[-80:]),
                        language="text",
                    )

                return_code = process.wait()

            duration = round(
                time.perf_counter() - start_time,
                2,
            )

            log_text = "".join(collected_lines)

            log_path.write_text(
                log_text,
                encoding="utf-8",
            )

            total = parse_count(
                r"Input records:\s*([\d,]+)",
                log_text,
            )

            valid = parse_count(
                r"Valid records:\s*([\d,]+)",
                log_text,
            )

            invalid = parse_count(
                r"Quarantined records:\s*([\d,]+)",
                log_text,
            )

            run_status = (
                "SUCCESS"
                if return_code == 0
                else "FAILED"
            )

            record = {
                "run_id": run_id,
                "status": run_status,
                "duration_seconds": duration,
                "total_records": total,
                "valid_records": valid,
                "invalid_records": invalid,
            }

            save_history(record)

            if return_code == 0:

                status.update(
                    label="Pipeline completed successfully",
                    state="complete",
                )

                st.success(
                    "The Spark pipeline completed successfully."
                )

                a, b, c, d = st.columns(4)

                a.metric(
                    "Input Records",
                    format_number(total),
                )

                b.metric(
                    "Valid Records",
                    format_number(valid),
                )

                c.metric(
                    "Quarantined",
                    format_number(invalid),
                )

                d.metric(
                    "Execution Time",
                    f"{duration:.2f} sec",
                )

                st.markdown("### Generated Outputs")

                st.write(
                    f"**Curated Parquet:** `{curated_path}`"
                )

                st.write(
                    f"**Quarantine Parquet:** `{quarantine_path}`"
                )

                st.write(
                    f"**Execution Log:** `{log_path}`"
                )

                st.download_button(
                    "Download Execution Log",
                    data=log_text,
                    file_name=f"payshield_{run_id}.log",
                    mime="text/plain",
                )

            else:

                status.update(
                    label="Spark pipeline failed",
                    state="error",
                )

                st.error(
                    f"Spark exited with code {return_code}. "
                    "Inspect the logs above."
                )

                st.download_button(
                    "Download Error Log",
                    data=log_text,
                    file_name=f"payshield_error_{run_id}.log",
                    mime="text/plain",
                )

        except Exception as exc:

            status.update(
                label="Unable to start Spark",
                state="error",
            )

            st.exception(exc)


# ============================================================
# 10. TRANSACTION EXPLORER
# ============================================================

elif page == "Transaction Explorer":

    st.subheader("Transaction Explorer")

    st.write(
        "Filter and inspect a sample of the selected CSV. "
        "This page uses a preview sample, not the entire dataset."
    )

    selected_input = st.session_state.selected_input

    if not selected_input and DEFAULT_DATASET.exists():
        selected_input = str(DEFAULT_DATASET)

    if not selected_input or not Path(selected_input).exists():

        st.warning(
            "Select a dataset in Data Ingestion first."
        )

        st.stop()

    try:

        sample = pd.read_csv(
            selected_input,
            nrows=50000,
            low_memory=False,
        )

        if "type" in sample.columns:

            types = sorted(
                sample["type"].dropna().astype(str).unique()
            )

            selected_types = st.multiselect(
                "Transaction Type",
                options=types,
                default=types,
            )

            sample = sample[
                sample["type"].astype(str).isin(selected_types)
            ]

        if "amount" in sample.columns:

            sample["amount"] = pd.to_numeric(
                sample["amount"],
                errors="coerce",
            )

            max_amount = float(
                sample["amount"].max()
                if not sample["amount"].dropna().empty
                else 0
            )

            low, high = st.slider(
                "Transaction Amount Range",
                min_value=0.0,
                max_value=max(max_amount, 1.0),
                value=(0.0, max(max_amount, 1.0)),
            )

            sample = sample[
                sample["amount"].between(low, high)
            ]

        if "isFraud" in sample.columns:

            labels = sorted(
                sample["isFraud"].dropna().unique().tolist()
            )

            selected_labels = st.multiselect(
                "Dataset Fraud Label",
                options=labels,
                default=labels,
            )

            sample = sample[
                sample["isFraud"].isin(selected_labels)
            ]

        a, b = st.columns(2)

        a.metric(
            "Matching Sample Rows",
            f"{len(sample):,}",
        )

        b.metric(
            "Displayed Columns",
            f"{len(sample.columns):,}",
        )

        st.dataframe(
            sample.head(500),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Filtered Sample",
            data=sample.to_csv(index=False),
            file_name="payshield_filtered_sample.csv",
            mime="text/csv",
        )

        st.caption(
            "The explorer reads up to 50,000 source rows for responsiveness. "
            "It does not claim to search the entire dataset."
        )

    except Exception as exc:

        st.error(
            f"Unable to inspect the selected dataset: {exc}"
        )


# ============================================================
# 11. RUN HISTORY
# ============================================================

elif page == "Run History":

    st.subheader("Pipeline Run History")

    history = get_history()

    if history.empty:

        st.info(
            "No execution history is available yet. "
            "Run the pipeline to record your first execution."
        )

    else:

        a, b, c = st.columns(3)

        a.metric(
            "Total Runs",
            f"{len(history):,}",
        )

        b.metric(
            "Successful",
            f"{(history['status'] == 'SUCCESS').sum():,}",
        )

        c.metric(
            "Failed",
            f"{(history['status'] == 'FAILED').sum():,}",
        )

        st.dataframe(
            history.sort_values(
                "run_id",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        if "duration_seconds" in history.columns:

            st.markdown("### Execution Duration Trend")

            st.line_chart(
                history.set_index("run_id")[
                    "duration_seconds"
                ]
            )

        st.download_button(
            "Download Run History",
            data=history.to_csv(index=False),
            file_name="payshield_run_history.csv",
            mime="text/csv",
        )


# ============================================================
# 12. AWS ARCHITECTURE
# ============================================================

elif page == "AWS Architecture":

    st.subheader("Target AWS Data Engineering Architecture")

    st.write(
        "The following is the proposed cloud architecture for PayShield. "
        "The current implementation runs locally."
    )

    aws_steps = [
        (
            "01",
            "Amazon S3 — Raw Zone",
            "Store original transaction files.",
        ),
        (
            "02",
            "AWS Glue / Apache Spark",
            "Run managed ETL and transformation jobs.",
        ),
        (
            "03",
            "Amazon S3 — Curated Zone",
            "Store validated, partitioned Parquet data.",
        ),
        (
            "04",
            "Amazon Athena",
            "Query curated datasets with SQL.",
        ),
        (
            "05",
            "Amazon CloudWatch",
            "Monitor job logs, failures, and execution metrics.",
        ),
        (
            "06",
            "PayShield Analytics",
            "Explore results through an interactive dashboard.",
        ),
    ]

    for start in range(0, len(aws_steps), 2):

        columns = st.columns(2)

        for index, item in enumerate(
            aws_steps[start:start + 2]
        ):

            number, service, description = item

            with columns[index]:

                with st.container(border=True):

                    st.markdown(
                        f"### {number} · {service}"
                    )

                    st.write(description)

    st.warning(
        "AWS services have not yet been connected. "
        "This page describes the planned architecture, "
        "not deployed cloud resources."
    )

    st.markdown("### Current Local Architecture")

    st.code(
        """
CSV Dataset
    |
    v
PySpark Data Validation
    |
    +------> Quarantine Parquet
    |
    v
Feature Engineering
    |
    v
Spark SQL Analytics
    |
    v
Partitioned Curated Parquet
    |
    v
Execution Logs + Run History
        """,
        language="text",
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        PayShield · Financial Data Engineering Platform<br>
        Python · Apache Spark · PySpark · Spark SQL · Parquet · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
