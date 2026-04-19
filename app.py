
"""
Diamond Scanner – Enhanced Analog Engine Version
Enhancements included:
1. TSI percentile normalization
2. CCI percentile normalization
3. BB% percentile normalization
4. Weighted feature‑vector similarity for analog matching
5. Basic market regime tagging
"""

import pandas as pd
import numpy as np

# -----------------------------
# Indicator percentile helpers
# -----------------------------

def add_percentile_features(df):
    df["TSI_424_pct"] = df["TSI_424"].rank(pct=True)
    df["TSI_747_pct"] = df["TSI_747"].rank(pct=True)
    df["CCI_pct"] = df["CCI15"].rank(pct=True)
    df["BB_pct_pct"] = df["BB_pct"].rank(pct=True)
    return df


# -----------------------------
# Market regime tagging
# -----------------------------

def tag_market_regime(df):
    df["regime"] = "neutral"

    df.loc[(df["TSI_747"] > 70) & (df["CCI15"] > 100), "regime"] = "overheated"
    df.loc[(df["TSI_747"] < 0) & (df["CCI15"] < -100), "regime"] = "weak"

    return df


# -----------------------------
# Weighted feature distance
# -----------------------------

FEATURE_WEIGHTS = {
    "TSI_424_pct": 2.0,
    "TSI_747_pct": 2.0,
    "CCI_pct": 1.5,
    "BB_pct_pct": 1.5,
    "VWAP_dist": 1.0,
    "ATR_pct": 1.0
}


def feature_distance(row, current):
    dist = 0
    for f, w in FEATURE_WEIGHTS.items():
        dist += w * abs(row[f] - current[f])
    return dist


# -----------------------------
# Analog finder
# -----------------------------

def find_analogs(df, current_row, n=20):

    # regime filter
    pool = df[df["regime"] == current_row["regime"]]

    if len(pool) < n:
        pool = df

    pool = pool.copy()

    pool["distance"] = pool.apply(lambda r: feature_distance(r, current_row), axis=1)

    analogs = pool.nsmallest(n, "distance")

    return analogs


# -----------------------------
# Example probability outputs
# -----------------------------

def compute_probabilities(analogs):

    dip_prob_1d = (analogs["ret_1d"] < 0).mean()
    dip_prob_2d = (analogs["ret_2d"] < 0).mean()
    dip_prob_5d = (analogs["ret_5d"] < 0).mean()

    expected_1d = analogs["ret_1d"].mean()
    expected_2d = analogs["ret_2d"].mean()
    expected_5d = analogs["ret_5d"].mean()

    return {
        "dip_prob_1d": dip_prob_1d,
        "dip_prob_2d": dip_prob_2d,
        "dip_prob_5d": dip_prob_5d,
        "expected_1d": expected_1d,
        "expected_2d": expected_2d,
        "expected_5d": expected_5d
    }


# -----------------------------
# Example usage
# -----------------------------

def run_scanner(df):

    df = add_percentile_features(df)
    df = tag_market_regime(df)

    current_row = df.iloc[-1]

    analogs = find_analogs(df, current_row)

    stats = compute_probabilities(analogs)

    return analogs, stats


if __name__ == "__main__":
    print("Diamond Scanner enhanced analog engine loaded.")
