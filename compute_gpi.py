#!/usr/bin/env python
"""
Example script for computing daily GPI from timestamped CGM data.

Required input columns:
    participant_id, timestamp, glucose_mg_dl

Example:
    python compute_gpi.py --input examples/synthetic_cgm.csv --output daily_gpi.csv
"""

import argparse

import numpy as np
import pandas as pd


def compute_gpi(glucose_values, sampling_interval_minutes=5):
    """
    Compute daily Glycemic Persistence Index (GPI).

    GPI is the largest integer k such that glucose is >= k mg/dL for at
    least k minutes. For 5-minute CGM data, sort glucose values from high
    to low, compute cumulative represented time, and find the largest
    integer value satisfying both the glucose and duration requirements.
    """
    glucose = pd.to_numeric(pd.Series(glucose_values), errors="coerce").dropna().to_numpy()
    if not np.isfinite(sampling_interval_minutes) or sampling_interval_minutes <= 0:
        raise ValueError("Sampling interval must be finite and positive.")
    if np.any(~np.isfinite(glucose)) or np.any(glucose <= 0):
        raise ValueError("Glucose must be finite and positive.")
    if glucose.size == 0:
        return np.nan

    glucose_sorted = np.sort(glucose.astype(float))[::-1]
    cumulative_minutes = np.arange(1, glucose_sorted.size + 1) * sampling_interval_minutes
    candidates = np.minimum(np.floor(glucose_sorted), np.floor(cumulative_minutes))
    return int(np.max(candidates))


def compute_daily_metrics(df, min_readings=200):
    """
    Compute daily GPI and standard CGM metrics.

    TIR/TAR/TBR metrics are reported as percentages of valid glucose readings.
    """
    required = {"participant_id", "timestamp", "glucose_mg_dl"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(sorted(missing)))

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["glucose_mg_dl"] = pd.to_numeric(data["glucose_mg_dl"], errors="coerce")
    data = data.dropna(subset=["participant_id", "timestamp", "glucose_mg_dl"])
    data["date"] = data["timestamp"].dt.date

    rows = []
    for (participant_id, date), day in data.groupby(["participant_id", "date"]):
        glucose = day["glucose_mg_dl"].to_numpy(dtype=float)
        n_readings = int(len(glucose))
        qc_pass = n_readings >= min_readings

        if qc_pass:
            mean_glucose = float(np.mean(glucose))
            sd_glucose = float(np.std(glucose, ddof=1)) if n_readings > 1 else np.nan
            cv_glucose = float(sd_glucose / mean_glucose) if mean_glucose > 0 else np.nan
            gpi = compute_gpi(glucose)
            tir_70_180 = float(np.mean((glucose >= 70) & (glucose <= 180)) * 100)
            tar_gt_180 = float(np.mean(glucose > 180) * 100)
            tar_gt_250 = float(np.mean(glucose > 250) * 100)
            tbr_lt_70 = float(np.mean(glucose < 70) * 100)
            tbr_lt_54 = float(np.mean(glucose < 54) * 100)
        else:
            mean_glucose = np.nan
            sd_glucose = np.nan
            cv_glucose = np.nan
            gpi = np.nan
            tir_70_180 = np.nan
            tar_gt_180 = np.nan
            tar_gt_250 = np.nan
            tbr_lt_70 = np.nan
            tbr_lt_54 = np.nan

        rows.append(
            {
                "participant_id": participant_id,
                "date": date,
                "n_readings": n_readings,
                "qc_pass": bool(qc_pass),
                "gpi": gpi,
                "mean_glucose": mean_glucose,
                "sd_glucose": sd_glucose,
                "cv_glucose": cv_glucose,
                "tir_70_180": tir_70_180,
                "tar_gt_180": tar_gt_180,
                "tar_gt_250": tar_gt_250,
                "tbr_lt_70": tbr_lt_70,
                "tbr_lt_54": tbr_lt_54,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute daily GPI from timestamped CGM data.")
    parser.add_argument("--input", required=True, help="Input CSV with participant_id, timestamp, glucose_mg_dl.")
    parser.add_argument("--output", required=True, help="Output CSV for daily GPI and CGM metrics.")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    output_df = compute_daily_metrics(input_df)
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
