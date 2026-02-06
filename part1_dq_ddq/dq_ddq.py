import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import Counter
from scipy.stats import entropy
import random
import re

# -----------------------------
# 1. DATA GENERATION
# -----------------------------
def generate_data(n=10000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    categories = ["Electronics", "Clothing", "Others"]
    category_probs = [0.6, 0.3, 0.1]

    sentiments = ["positive", "neutral", "negative"]
    sentiment_probs = [0.5, 0.3, 0.2]

    outcomes = ["purchase", "abandon", "escalate"]

    data = []

    base_date = datetime(2019, 1, 1)

    for i in range(n):
        record = {
            "customer_id": f"CUST_{np.random.randint(1, 4000)}",
            "product_category": np.random.choice(categories, p=category_probs),
            "sentiment": np.random.choice(sentiments, p=sentiment_probs),
            "timestamp": base_date + timedelta(days=np.random.randint(0, 1800)),
            "conversation_text": "Customer inquiry regarding product usage",
            "outcome": np.random.choice(outcomes)
        }
        data.append(record)

    df = pd.DataFrame(data)

    # Introduce missing values (20%)
    for col in df.columns:
        df.loc[df.sample(frac=0.2).index, col] = np.nan

    # Introduce duplicates (5%)
    duplicates = df.sample(frac=0.05)
    df = pd.concat([df, duplicates], ignore_index=True)

    # Introduce invalid formats (3%)
    df["timestamp"] = df["timestamp"].astype("object")  # important
    invalid_idx = df.sample(frac=0.03).index
    df.loc[invalid_idx, "timestamp"] = "invalid_date"

    return df


# -----------------------------
# 2. DATA QUALITY METRICS
# -----------------------------
def assess_dq(df):
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()

    completeness = 1 - (missing / total_cells)

    duplicate_rate = 1 - (df.duplicated().sum() / len(df))

    def valid_timestamp(x):
        try:
            pd.to_datetime(x)
            return True
        except:
            return False

    valid_timestamps = df["timestamp"].apply(
        lambda x: valid_timestamp(x) if not pd.isna(x) else False
    ).sum()

    format_validity = valid_timestamps / len(df)

    return {
        "completeness": round(completeness, 3),
        "duplicate_rate": round(duplicate_rate, 3),
        "format_validity": round(format_validity, 3)
    }


# -----------------------------
# 3. DDQ METRICS (KL-DIVERGENCE)
# -----------------------------
def kl_divergence(p, q):
    return entropy(p, q)

def assess_ddq(train_df, prod_distributions):
    metrics = {}

    for feature, prod_dist in prod_distributions.items():
        train_counts = Counter(train_df[feature].dropna())
        train_total = sum(train_counts.values())

        train_dist = []
        prod_dist_list = []

        for k in prod_dist:
            train_dist.append(train_counts.get(k, 0) / train_total)
            prod_dist_list.append(prod_dist[k])

        kl = kl_divergence(prod_dist_list, train_dist)
        metrics[feature] = round(float(kl), 4)

    return metrics


# -----------------------------
# 4. DATA CLEANING
# -----------------------------
def clean_data(df):
    df = df.drop_duplicates().copy()

    df.loc[:, "timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna(
        subset=["customer_id", "product_category", "sentiment", "timestamp"]
    ).copy()

    return df


# -----------------------------
# 5. REPORT GENERATION
# -----------------------------
def generate_report(dq_metrics, ddq_metrics):
    report = {
        "DQ_Metrics": {
            "Completeness": {
                "value": dq_metrics["completeness"],
                "pass": bool(dq_metrics["completeness"] >= 0.95)
            },
            "Duplicate_Rate": {
                "value": dq_metrics["duplicate_rate"],
                "pass": bool(dq_metrics["duplicate_rate"] >= 0.98)
            },
            "Format_Validity": {
                "value": dq_metrics["format_validity"],
                "pass": bool(dq_metrics["format_validity"] >= 0.99)
            }
        },
        "DDQ_Metrics": {},
        "Overall_Status": "PASS"
    }

    for feature, kl in ddq_metrics.items():
        passed = bool(kl < 0.1)
        report["DDQ_Metrics"][feature] = {
            "KL_Divergence": float(kl),
            "pass": passed
        }
        if not passed:
            report["Overall_Status"] = "FAIL"

    return report



# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    df = generate_data()

    dq_metrics = assess_dq(df)

    production_distributions = {
        "product_category": {
            "Electronics": 0.5,
            "Clothing": 0.35,
            "Others": 0.15
        },
        "sentiment": {
            "positive": 0.45,
            "neutral": 0.35,
            "negative": 0.20
        }
    }

    ddq_metrics = assess_ddq(df, production_distributions)

    cleaned_df = clean_data(df)

    report = generate_report(dq_metrics, ddq_metrics)

    with open("part1_dq_ddq/quality_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Quality report generated successfully.")
