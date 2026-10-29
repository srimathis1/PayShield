# PayShield — Financial Transaction Intelligence Pipeline

PayShield is a data engineering project designed to process financial transaction data, validate data quality, engineer useful features, and generate curated datasets for downstream fraud analysis.

## Project Overview

Financial transaction datasets can contain millions of records. Processing this data requires scalable data processing, data validation, efficient storage, and monitoring.

PayShield demonstrates a pipeline built with Apache Spark that transforms raw transaction data into structured, analysis-ready datasets.

## Key Features

- **Data Ingestion:** Reads transaction data from CSV files.
- **Data Validation:** Checks transaction records against configured data quality rules.
- **Data Quarantine:** Separates records that fail configured validation checks.
- **Feature Engineering:** Creates useful transaction features for downstream analysis.
- **Distributed Processing:** Uses Apache Spark for large-scale data processing.
- **Optimized Storage:** Writes processed data in Parquet format.
- **Performance Benchmarking:** Compares CSV and Parquet storage and read performance.
- **Pipeline Monitoring:** Records pipeline execution information.
- **Interactive Dashboard:** Provides a Streamlit interface for pipeline execution, transaction exploration, and run history.
- **AWS Architecture:** Documents a proposed cloud architecture for future deployment.

## Technology Stack

- Python
- Apache Spark / PySpark
- Pandas
- Streamlit
- CSV and Apache Parquet
- Git and GitHub
- AWS architecture concepts: Amazon S3, AWS Glue, Amazon Athena, and Amazon CloudWatch

**Note:** AWS services are architectural targets and are not necessarily connected to the current implementation.

## Dataset

This project uses the PaySim financial transaction dataset.

The original dataset contains approximately 6.3 million transaction records with fields such as:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

The dataset includes simulated transactions and fraud labels. It is used for data engineering and analysis demonstrations, not as a production fraud detection service.

**Dataset note:** The original CSV is not included in this repository because of its large size.

## Project Structure

```text
PayShield/
├── dashboard/
│   └── app.py
├── src/
│   ├── transaction_pipeline.py
│   ├── data_quality.py
│   ├── data_validation_analysis.py
│   ├── benchmark_storage.py
│   ├── performance_tuning.py
│   ├── pipeline_monitor.py
│   ├── execution_plan.py
│   ├── first_spark_job.py
│   ├── inspect_dataset.py
│   └── test_pipeline.py
├── data/
│   └── raw/
├── output/
├── logs/
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3
- Java 17
- Apache Spark compatible with your environment
- Streamlit
- Pandas

## Installation

Clone the repository:

```bash
git clone https://github.com/srimathis1/PayShield.git
cd PayShield
```

Install the dashboard dependencies:

```bash
pip install streamlit pandas
```

Install and configure Apache Spark separately if you want to execute the Spark pipeline locally.

## Run the Dashboard

From the project directory, run:

```bash
streamlit run dashboard/app.py
```

Streamlit will display a local URL in your terminal. Open it in your browser.

## Run the Spark Pipeline

Place the PaySim CSV dataset in:

```text
data/raw/paysim dataset.csv
```

Then execute:

```bash
spark-submit src/transaction_pipeline.py
```

The pipeline uses the input and output paths configured in the source code or through supported environment variables.

Review the pipeline configuration before running it to confirm that the dataset path, Spark installation, and output directories are correct.

## Outputs

Depending on the selected configuration, the pipeline can produce:

- Curated transaction data in Parquet format
- Quarantined records
- Pipeline execution logs
- Data quality results
- Storage and performance benchmark results

Generated data and logs should remain outside version control unless small, non-sensitive demonstration files are intentionally included.

## Future Enhancements

- Deploy a lightweight interactive dashboard using Streamlit Community Cloud.
- Integrate Amazon S3 for raw and curated data storage.
- Use AWS Glue for managed data processing.
- Query curated Parquet data with Amazon Athena.
- Monitor pipeline execution with Amazon CloudWatch.
- Add automated testing and continuous integration.

## Disclaimer

PayShield is an educational and portfolio project using simulated transaction data. It is not a production financial monitoring system and should not be used to make real financial decisions.

## Author

**Srimathi S.**

GitHub: https://github.com/srimathis1
