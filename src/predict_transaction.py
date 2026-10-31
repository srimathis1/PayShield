#!/usr/bin/env python3

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    os.environ.get(
        "PAYSHIELD_MODEL_PATH",
        str(ROOT / "output" / "ml" / "random_forest_model")
    )
)

ALLOWED_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
}


def emit(data):
    print(json.dumps(data, allow_nan=False), flush=True)


def main():
    spark = None

    try:
        transaction = json.load(sys.stdin)

        tx_type = str(transaction.get("type", "")).strip().upper()

        if tx_type not in ALLOWED_TYPES:
            raise ValueError("Invalid transaction type.")

        fields = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest"
        ]

        values = {}

        for field in fields:
            value = float(transaction[field])

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{field} must be a finite non-negative number."
                )

            values[field] = value

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Saved Spark model not found: {MODEL_PATH}"
            )

        os.environ.pop("HADOOP_CONF_DIR", None)
        os.environ.pop("YARN_CONF_DIR", None)

        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            StructType,
            StructField,
            StringType,
            DoubleType
        )
        from pyspark.ml import PipelineModel

        spark = (
            SparkSession.builder
            .appName("PayShield-Fraud-Prediction")
            .master("local[*]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("ERROR")

        schema = StructType([
            StructField("type", StringType(), False),
            StructField("amount", DoubleType(), False),
            StructField("oldbalanceOrg", DoubleType(), False),
            StructField("newbalanceOrig", DoubleType(), False),
            StructField("oldbalanceDest", DoubleType(), False),
            StructField("newbalanceDest", DoubleType(), False)
        ])

        row = [(
            tx_type,
            values["amount"],
            values["oldbalanceOrg"],
            values["newbalanceOrig"],
            values["oldbalanceDest"],
            values["newbalanceDest"]
        )]

        input_df = spark.createDataFrame(row, schema)

        # Load the EXISTING trained Spark MLlib model.
        model = PipelineModel.load(str(MODEL_PATH))

        result = (
            model.transform(input_df)
            .select("prediction", "probability")
            .first()
        )

        prediction = int(result["prediction"])
        probability = float(result["probability"][1])

        if not math.isfinite(probability):
            raise ValueError("Model returned an invalid probability.")

        # Sender balance consistency.
        sender_difference = (
            values["oldbalanceOrg"]
            - values["amount"]
            - values["newbalanceOrig"]
        )

        # Receiver balance consistency.
        receiver_difference = (
            values["oldbalanceDest"]
            + values["amount"]
            - values["newbalanceDest"]
        )

        tolerance = max(0.01, values["amount"] * 1e-6)

        sender_mismatch = abs(sender_difference) > tolerance
        receiver_mismatch = abs(receiver_difference) > tolerance

        balance_mismatch = (
            sender_mismatch or receiver_mismatch
        )

        # Risk assessment combines independent review signals.
        if prediction == 1 and balance_mismatch:
            risk_level = "HIGH"
            risk_status = "MODEL_ALERT_AND_BALANCE_MISMATCH"
            recommendation = (
                "Prioritize this transaction for human investigation."
            )

        elif prediction == 1:
            risk_level = "HIGH"
            risk_status = "ML_FRAUD_ALERT"
            recommendation = (
                "The trained model flagged this transaction."
            )

        elif balance_mismatch:
            risk_level = "REVIEW"
            risk_status = "BALANCE_MISMATCH"
            recommendation = (
                "Investigate the balance inconsistency."
            )

        else:
            risk_level = "LOW"
            risk_status = "NO_RULE_ALERT"
            recommendation = (
                "No configured alert was triggered."
            )

        emit({
            "success": True,
            "model_prediction": prediction,
            "prediction_label": (
                "FRAUD" if prediction == 1 else "NOT_FLAGGED"
            ),
            "fraud_probability": probability,
            "model_path": str(MODEL_PATH.resolve()),

            "risk_assessment": {
                "risk_level": risk_level,
                "risk_status": risk_status,
                "recommendation": recommendation,
                "requires_review": risk_level != "LOW"
            },

            "input_features": {
                "type": tx_type,
                **values
            },

            "balance_diagnostics": {
                "sender_balance_difference": sender_difference,
                "receiver_balance_difference": receiver_difference,
                "sender_balance_mismatch": sender_mismatch,
                "receiver_balance_mismatch": receiver_mismatch,
                "any_balance_mismatch": balance_mismatch,
                "tolerance": tolerance
            }
        })

    except Exception as exc:
        emit({
            "success": False,
            "error": str(exc),
            "model_path": str(MODEL_PATH)
        })
        sys.exit(1)

    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
