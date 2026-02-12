import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import Counter
from scipy.stats import entropy
from .text_templates import CATEGORY_PHRASES, CONVERSATION_TEMPLATES, AGENT_RESPONSES
import random

# =========================================================
# 1. SYNTHETIC DATA GENERATION
# =========================================================
def generate_data(n=10000, seed=42):
    """
    Generates synthetic ecommerce chat data with realistic
    quality issues:
    - Missing values
    - Duplicates
    - Invalid timestamp formats
    - Imbalanced distributions
    """

    np.random.seed(seed)
    random.seed(seed)

    # Core categorical distributions
    categories = ["Electronics", "Clothing", "Others"]
    category_probs = [0.6, 0.3, 0.1]  # Imbalanced by design

    sentiments = ["positive", "neutral", "negative"]
    sentiment_probs = [0.5, 0.3, 0.2]

    segments = ["VIP", "regular", "new"]
    segment_probs = [0.15, 0.65, 0.20]  # Realistic retail mix

    outcomes = ["purchase", "abandon", "escalate"]

    base_date = datetime(2019, 1, 1)
    data = []

    for _ in range(n):

        sentiment = np.random.choice(sentiments, p=sentiment_probs)
        product_category = np.random.choice(categories, p=category_probs)
        template = random.choice(CONVERSATION_TEMPLATES[sentiment])
        category_phrase = random.choice(CATEGORY_PHRASES[product_category])
        agent_response = random.choice(AGENT_RESPONSES[sentiment])

        conversation_text = f"""
        {template} regarding {category_phrase}, {agent_response}
        """

        data.append({
            "customer_id": f"CUST_{np.random.randint(1, 4000)}",
            "customer_segment": np.random.choice(segments, p=segment_probs),
            "product_category": product_category,
            "sentiment": sentiment,
            "timestamp": base_date + timedelta(days=np.random.randint(0, 1800)),
            "conversation_text": conversation_text,
            "outcome": np.random.choice(outcomes)
        })

    df = pd.DataFrame(data)

    # Introduce missing values (20%)
    for col in df.columns:
        df.loc[df.sample(frac=0.2).index, col] = np.nan

    # Introduce duplicates (5%)
    df = pd.concat([df, df.sample(frac=0.05)], ignore_index=True)

    # Introduce invalid timestamp formats (3%)
    df["timestamp"] = df["timestamp"].astype("object")
    invalid_idx = df.sample(frac=0.03).index
    df.loc[invalid_idx, "timestamp"] = "invalid_date"

    return df


# =========================================================
# 2. DATA QUALITY (DQ) METRICS
# =========================================================
def assess_dq(df):
    """
    Computes completeness, duplicate rate and format validity
    as defined in the assignment.
    """

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = 1 - (missing_cells / total_cells)

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


# =========================================================
# 3. DISTRIBUTIONAL DATA QUALITY (DDQ)
# =========================================================
def assess_ddq(train_df, prod_distributions):
    """
    Computes KL divergence between production and training
    distributions for each specified feature.
    """

    metrics = {}

    for feature, prod_dist in prod_distributions.items():
        train_counts = Counter(train_df[feature].dropna())
        train_total = sum(train_counts.values())

        train_probs = []
        prod_probs = []

        # Ensure consistent support ordering
        for value in prod_dist.keys():
            train_probs.append(train_counts.get(value, 0) / train_total)
            prod_probs.append(prod_dist[value])

        # KL(P_prod || P_train)
        kl_value = entropy(prod_probs, train_probs)
        metrics[feature] = round(float(kl_value), 4)

    return metrics


# =========================================================
# 4. DATA CLEANING
# =========================================================
def clean_data(df):
    """
    Applies minimal cleaning:
    - Remove duplicates
    - Coerce timestamps
    - Drop rows missing critical fields
    - Clean conversation_text safely
    """

    # Remove duplicates
    df = df.drop_duplicates().copy()

    # Coerce timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows where conversation_text is NaN FIRST
    df = df.dropna(subset=["conversation_text"])

    # Now safely clean the column
    df["conversation_text"] = (
        df["conversation_text"]
        .astype(str)
        .str.strip()
    )

    # Drop empty strings after strip
    df = df[df["conversation_text"] != ""]

    # Drop rows missing other critical fields
    df = df.dropna(subset=[
        "customer_id",
        "customer_segment",
        "product_category",
        "sentiment",
        "timestamp"
    ])

    # Reset index once at the end
    df = df.reset_index(drop=True)

    # Final sanity checks (these SHOULD pass now)
    assert df["conversation_text"].isna().sum() == 0
    assert df["conversation_text"].apply(lambda x: isinstance(x, str)).all()

    df.to_csv(
        "part1_dq_ddq/cleaned_conv.csv",
        index=False
    )

    return df



# =========================================================
# 5. REPORT GENERATION
# =========================================================
def generate_report(dq_metrics, ddq_metrics):
    """
    Generates structured JSON report with pass/fail logic.
    """

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
            "KL_Divergence": kl,
            "pass": passed
        }
        if not passed:
            report["Overall_Status"] = "FAIL"

    return report


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":

    df = generate_data()
    dq_metrics = assess_dq(df)

    # Convert timestamp to time buckets for temporal DDQ
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["time_bucket"] = pd.cut(
        df["timestamp"].dt.year,
        bins=[2018, 2020, 2021, 2025],
        labels=["pre_covid", "covid", "post_covid"]
    )

    # Production reference distributions
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
        },
        "customer_segment": {
            "VIP": 0.20,
            "regular": 0.60,
            "new": 0.20
        },
        "time_bucket": {
            "pre_covid": 0.25,
            "covid": 0.25,
            "post_covid": 0.50
        }
    }

    ddq_metrics = assess_ddq(df, production_distributions)
    cleaned_df = clean_data(df)
    report = generate_report(dq_metrics, ddq_metrics)

    with open("part1_dq_ddq/quality_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Quality report generated successfully.")
