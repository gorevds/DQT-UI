"""Synthetic demo dataset for users who want to try DQT without their own data.

Generates a wide scoring-style table (33 features + target) over 24 monthly
buckets with deliberate DQ-signals: drifting numerics, growing missingness,
shifting category share, heavy-tail outliers, type contamination, a noisy
binary target with a real-but-imperfect signal from the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    month_idx = rng.integers(0, len(months), size=n_rows)
    day_offset = rng.integers(0, 28, size=n_rows)
    application_date = pd.to_datetime([
        m + pd.Timedelta(days=int(d)) for m, d in zip(months[month_idx], day_offset)
    ])
    t = month_idx.astype(float)  # 0..23 — used to inject drift

    # --- Demographics (mostly stable) -----------------------------------
    age = rng.integers(20, 72, size=n_rows)
    gender = rng.choice(["M", "F"], size=n_rows, p=[0.55, 0.45])
    marital = rng.choice(
        ["single", "married", "divorced", "widowed"],
        size=n_rows, p=[0.40, 0.45, 0.12, 0.03],
    )
    education = rng.choice(
        ["secondary", "vocational", "higher", "phd"],
        size=n_rows, p=[0.35, 0.30, 0.30, 0.05],
    )
    dependents = rng.choice([0, 1, 2, 3, 4], size=n_rows, p=[0.45, 0.25, 0.18, 0.08, 0.04])

    # --- Region with shifting share over time ----------------------------
    p_moscow = 0.50 - 0.006 * t
    p_spb = 0.30 - 0.002 * t
    region = np.empty(n_rows, dtype=object)
    for i in range(n_rows):
        p_other = 1.0 - p_moscow[i] - p_spb[i]
        region[i] = rng.choice(["Moscow", "SPb", "Other"],
                                p=[p_moscow[i], p_spb[i], p_other])

    # --- Application -----------------------------------------------------
    app_amount = np.exp(rng.normal(loc=12.5 + 0.01 * t, scale=0.5, size=n_rows)).round(0)
    # Inject heavy-tail outliers in months 18+
    late = np.where(t >= 18)[0]
    if len(late) > 0:
        out_idx = rng.choice(late, size=max(1, int(len(late) * 0.03)), replace=False)
        app_amount[out_idx] = app_amount[out_idx] * rng.uniform(8, 15, size=len(out_idx))
    app_term_months = rng.choice(
        [12, 24, 36, 48, 60, 72, 84, 120], size=n_rows,
        p=[0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.04, 0.01],
    )
    app_type = rng.choice(
        ["consumer", "auto", "mortgage", "card"],
        size=n_rows, p=[0.55, 0.20, 0.10, 0.15],
    )

    # --- Income / employment with growing missingness --------------------
    monthly_income = np.exp(rng.normal(loc=10.8 + 0.005 * t, scale=0.4, size=n_rows)).round(0)
    employment_type = pd.Series(rng.choice(
        ["salaried", "self_employed", "student", "retired", "contractor"],
        size=n_rows, p=[0.55, 0.18, 0.07, 0.13, 0.07],
    )).astype(object)
    employment_years = np.clip(rng.exponential(scale=4, size=n_rows), 0, 30).round(1)
    employer_industry = pd.Series(rng.choice(
        ["finance", "tech", "manufacturing", "retail", "construction",
         "public", "agriculture", "transport", "other"], size=n_rows,
    )).astype(object)
    employment_type[rng.random(n_rows) < (0.02 + (0.23 / 23) * t)] = None
    employer_industry[rng.random(n_rows) < (0.01 + (0.30 / 23) * t)] = None

    # --- Bureau / credit history ----------------------------------------
    bureau_n_active = rng.poisson(lam=2.5 + 0.05 * t, size=n_rows)
    bureau_n_closed = rng.poisson(lam=4.0, size=n_rows)
    bureau_max_dpd_12m = rng.gamma(shape=1.0, scale=15, size=n_rows).round(0)
    bureau_total_balance = rng.exponential(scale=200000, size=n_rows).round(0)
    bureau_n_inquiries_3m = rng.poisson(lam=1.5, size=n_rows)
    bureau_oldest_credit_months = rng.integers(0, 240, size=n_rows)
    bureau_utilization = np.clip(rng.beta(a=2, b=5, size=n_rows), 0, 1).round(3)

    # --- Previous behavior ----------------------------------------------
    previous_loans_n = rng.poisson(lam=2.0, size=n_rows)
    previous_default_rate = np.clip(rng.beta(a=1.2, b=12, size=n_rows), 0, 1).round(3)
    days_since_last_payment = rng.integers(0, 90, size=n_rows)
    balance_to_credit_ratio = np.clip(rng.beta(a=2, b=4, size=n_rows), 0, 1).round(3)

    # --- External scores (one drifting, one stable) ---------------------
    score_v1 = (rng.normal(size=n_rows)
                + 0.01 * (age - 45)
                + 3e-6 * (monthly_income - 60000)).round(3)
    score_v2 = rng.normal(loc=0.05 * t, scale=1.0, size=n_rows).round(3)  # drift!
    score_external_a = np.clip(rng.normal(loc=600, scale=80, size=n_rows), 300, 850).round(0)
    score_external_b = np.clip(rng.normal(loc=0.5, scale=0.2, size=n_rows), 0, 1).round(3)

    # --- Channel / device / IP ------------------------------------------
    channel = rng.choice(["mobile_app", "web", "branch", "agent"],
                          size=n_rows, p=[0.50, 0.30, 0.15, 0.05])
    device_type = rng.choice(["ios", "android", "desktop", "tablet"],
                              size=n_rows, p=[0.30, 0.42, 0.22, 0.06])
    ip_country = rng.choice(["RU", "BY", "KZ", "UZ", "AM", "other"],
                             size=n_rows, p=[0.78, 0.07, 0.06, 0.04, 0.03, 0.02])

    # --- Extra fields with constant high missingness ---------------------
    # phone_verification_score: ~18% NaN throughout (optional verification step)
    phone_verification_score = pd.Series(
        np.clip(rng.beta(a=4, b=2, size=n_rows), 0, 1).round(3)
    ).astype(object)
    phone_verification_score[rng.random(n_rows) < 0.18] = None

    # risk_segment_legacy: ~60% NaN (legacy column being deprecated)
    risk_segment_legacy = pd.Series(rng.choice(
        ["A", "B", "C", "D", "E"], size=n_rows, p=[0.10, 0.25, 0.35, 0.20, 0.10],
    )).astype(object)
    risk_segment_legacy[rng.random(n_rows) < 0.60] = None

    # --- Extra fields with heavy outliers --------------------------------
    # delinquency_balance: log-normal with random 5% spike (10..30x)
    delinquency_balance = np.exp(rng.normal(loc=8.0, scale=1.0, size=n_rows)).round(0)
    spike_idx = rng.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    delinquency_balance[spike_idx] *= rng.uniform(10, 30, size=len(spike_idx))

    # device_age_months: stable mass + 3% extreme tail (very old devices)
    device_age_months = np.clip(rng.exponential(scale=14, size=n_rows), 0, 120).round(0)
    tail_idx = rng.choice(n_rows, size=int(n_rows * 0.03), replace=False)
    device_age_months[tail_idx] = rng.uniform(180, 360, size=len(tail_idx))
    device_age_months = pd.Series(device_age_months).astype(object)
    # Plus ~25% missing throughout
    device_age_months[rng.random(n_rows) < 0.25] = None

    # --- Target: noisy binary with multi-feature signal -----------------
    region_other = (region == "Other").astype(float)
    logit = (
        -2.5
        + 0.6 * score_v1
        + 0.4 * (score_v2 - 0.05 * t)        # underlying signal, drift removed
        - 0.015 * (age - 40)
        + 0.002 * bureau_max_dpd_12m
        + 0.5 * region_other
        + 0.7 * previous_default_rate
        + 3e-7 * (app_amount - 100000)
        + 0.04 * bureau_n_inquiries_3m
        - 0.0008 * (score_external_a - 600)
    )
    p_default = 1.0 / (1.0 + np.exp(-logit))
    default_flag = (rng.random(n_rows) < p_default).astype(int)

    df = pd.DataFrame({
        "application_date": application_date,
        "client_age": age.astype(int),
        "gender": gender,
        "marital_status": marital,
        "education": education,
        "region": region,
        "dependents": dependents.astype(int),
        "app_amount": app_amount,
        "app_term_months": app_term_months.astype(int),
        "app_type": app_type,
        "monthly_income": monthly_income,
        "employment_type": employment_type,
        "employment_years": employment_years,
        "employer_industry": employer_industry,
        "bureau_n_active": bureau_n_active.astype(int),
        "bureau_n_closed": bureau_n_closed.astype(int),
        "bureau_max_dpd_12m": bureau_max_dpd_12m,
        "bureau_total_balance": bureau_total_balance,
        "bureau_n_inquiries_3m": bureau_n_inquiries_3m.astype(int),
        "bureau_oldest_credit_months": bureau_oldest_credit_months.astype(int),
        "bureau_utilization": bureau_utilization,
        "previous_loans_n": previous_loans_n.astype(int),
        "previous_default_rate": previous_default_rate,
        "days_since_last_payment": days_since_last_payment.astype(int),
        "balance_to_credit_ratio": balance_to_credit_ratio,
        "score_v1": score_v1,
        "score_v2": score_v2,
        "score_external_a": score_external_a,
        "score_external_b": score_external_b,
        "channel": channel,
        "device_type": device_type,
        "ip_country": ip_country,
        "phone_verification_score": phone_verification_score,
        "risk_segment_legacy": risk_segment_legacy,
        "delinquency_balance": delinquency_balance,
        "device_age_months": device_age_months,
        "default_flag": default_flag,
    })
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
