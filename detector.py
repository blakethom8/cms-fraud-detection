#!/usr/bin/env python3
"""
detector.py — Final Fraud Detector (V15)
==========================================
Best approach discovered: pure max(subscores)
Fraud = extreme on ANY single dimension.

AUC progression: 0.5561 → 0.7695 → 0.8098 → 0.81+
"""

import sys
import pandas as pd
import duckdb
import numpy as np

DUCKDB_PATH = "/home/dataops/cms-data/data/provider_searcher.duckdb"

HIGH_VOLUME_SPECIALTIES = {
    'hematology-oncology', 'medical oncology', 'hematology', 'oncology',
    'radiation oncology', 'gynecological oncology', 'pharmacy',
    'surgical oncology', 'neuro-oncology'
}


def sigmoid(x, scale=2.0):
    return 1 / (1 + np.exp(-x / scale))


def zscore_within_group(df, group_col, value_col, min_group_size=5):
    vals = df[value_col].astype(float)
    groups = df[group_col]
    global_mu = vals.mean()
    global_sigma = vals.std() + 1e-9
    def _z(x):
        if len(x) >= min_group_size:
            return (x - x.mean()) / (x.std() + 1e-9)
        return (x - global_mu) / global_sigma
    return vals.groupby(groups).transform(_z).astype(float)


def score_providers(npi_list: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    # ── Part B ────────────────────────────────────────────────────────────
    try:
        part_b = con.execute("""
            SELECT
                CAST(Rndrng_NPI AS VARCHAR) AS npi,
                LOWER(Rndrng_Prvdr_Type) AS specialty,
                Tot_Mdcr_Pymt_Amt AS total_payment,
                Tot_Srvcs AS total_svc,
                Tot_Benes AS total_benes,
                CAST(Tot_Srvcs AS DOUBLE) / NULLIF(Tot_Benes, 0) AS svc_per_bene,
                CAST(Tot_Mdcr_Pymt_Amt AS DOUBLE) / NULLIF(Tot_Benes, 0) AS pay_per_bene
            FROM raw_physician_by_provider
            WHERE Rndrng_NPI IS NOT NULL AND Tot_Benes > 0
        """).df()
        print(f"[detector] Part B: {len(part_b):,} rows", file=sys.stderr)
    except Exception as e:
        print(f"[detector] Part B failed: {e}", file=sys.stderr)
        part_b = pd.DataFrame()

    # ── Part D ────────────────────────────────────────────────────────────
    try:
        part_d = con.execute("""
            SELECT
                CAST(PRSCRBR_NPI AS VARCHAR) AS npi,
                LOWER(Prscrbr_Type) AS specialty,
                Tot_Drug_Cst AS total_drug_cost,
                Tot_Benes AS total_benes,
                COALESCE(Opioid_LA_Prscrbr_Rate, 0) AS opioid_la_rate
            FROM raw_part_d_by_provider
            WHERE PRSCRBR_NPI IS NOT NULL
        """).df()
        print(f"[detector] Part D: {len(part_d):,} rows", file=sys.stderr)
    except Exception as e:
        print(f"[detector] Part D failed: {e}", file=sys.stderr)
        part_d = pd.DataFrame()

    # ── Open Payments ─────────────────────────────────────────────────────
    try:
        payments = con.execute("""
            SELECT
                CAST(Covered_Recipient_NPI AS VARCHAR) AS npi,
                SUM(CAST(Total_Amount_of_Payment_USDollars AS DOUBLE)) AS total_usd
            FROM raw_open_payments_general
            WHERE Covered_Recipient_NPI IS NOT NULL
            GROUP BY Covered_Recipient_NPI
        """).df()
        print(f"[detector] Open Payments: {len(payments):,} rows", file=sys.stderr)
    except Exception as e:
        print(f"[detector] Open Payments failed: {e}", file=sys.stderr)
        payments = pd.DataFrame()

    # ── PECOS ─────────────────────────────────────────────────────────────
    try:
        pecos = con.execute("SELECT DISTINCT CAST(NPI AS VARCHAR) AS npi FROM raw_pecos_enrollment WHERE NPI IS NOT NULL").df()
        pecos_npis = set(pecos['npi'].astype(str))
        print(f"[detector] PECOS: {len(pecos_npis):,} enrolled", file=sys.stderr)
    except Exception as e:
        print(f"[detector] PECOS failed: {e}", file=sys.stderr)
        pecos_npis = set()

    con.close()

    # ─────────────────────────────────────────────────────────────────────
    # BUILD SUBSCORES
    # ─────────────────────────────────────────────────────────────────────
    df = pd.DataFrame({'npi': [str(n) for n in npi_list]})

    df['sub_spb'] = 0.5
    df['sub_ppb'] = 0.5
    df['sub_la'] = 0.5
    df['sub_cpb'] = 0.5
    df['sub_pay'] = 0.0
    df['sub_pecos'] = 0.0

    # ── Subscore 1: Services-per-bene ─────────────────────────────────────
    if not part_b.empty:
        part_b['npi'] = part_b['npi'].astype(str)
        part_b['is_high_vol'] = part_b['specialty'].isin(HIGH_VOLUME_SPECIALTIES)

        part_b['spb_z'] = zscore_within_group(part_b, 'specialty', 'svc_per_bene')
        part_b['ppb_z'] = zscore_within_group(part_b, 'specialty', 'pay_per_bene')

        part_b['sub_spb'] = sigmoid(part_b['spb_z'], scale=2)
        part_b.loc[part_b['is_high_vol'], 'sub_spb'] *= 0.5
        part_b['sub_ppb'] = sigmoid(part_b['ppb_z'], scale=2)

        df = df.merge(part_b[['npi', 'sub_spb', 'sub_ppb', 'total_benes']],
                      on='npi', how='left', suffixes=('_old', ''))
        df['sub_spb'] = df['sub_spb'].fillna(0.5)
        df['sub_ppb'] = df['sub_ppb'].fillna(0.5)

    # ── Subscore 2: LA opioid rate + cost per bene ────────────────────────
    if not part_d.empty:
        part_d['npi'] = part_d['npi'].astype(str)
        part_d['cpb'] = part_d['total_drug_cost'] / (part_d['total_benes'] + 1)

        part_d['la_z'] = zscore_within_group(part_d, 'specialty', 'opioid_la_rate')
        part_d['cpb_z'] = zscore_within_group(part_d, 'specialty', 'cpb')

        part_d['sub_la'] = sigmoid(part_d['la_z'], scale=2)
        part_d['sub_cpb'] = sigmoid(part_d['cpb_z'], scale=2)

        df = df.merge(part_d[['npi', 'sub_la', 'sub_cpb']], on='npi', how='left', suffixes=('_old', ''))
        df['sub_la'] = df['sub_la'].fillna(0.5)
        df['sub_cpb'] = df['sub_cpb'].fillna(0.5)

    # ── Subscore 3: Open Payments ─────────────────────────────────────────
    if not payments.empty:
        payments['npi'] = payments['npi'].astype(str)
        payments['sub_pay'] = np.log1p(payments['total_usd'].fillna(0).clip(lower=0))
        payments['sub_pay'] /= (payments['sub_pay'].max() + 1e-9)

        df = df.merge(payments[['npi', 'sub_pay']], on='npi', how='left', suffixes=('_old', ''))
        df['sub_pay'] = df['sub_pay'].fillna(0)

    # ── Subscore 4: PECOS gap ─────────────────────────────────────────────
    df['no_pecos'] = (~df['npi'].isin(pecos_npis)).astype(float)
    has_billing = df.get('total_benes', pd.Series(0.0, index=df.index)).fillna(0) > 50
    df['sub_pecos'] = df['no_pecos'] * has_billing.astype(float)

    # ─────────────────────────────────────────────────────────────────────
    # FINAL SCORE: Pure max (fraud = extreme on ANY dimension)
    # ─────────────────────────────────────────────────────────────────────
    subscore_cols = ['sub_spb', 'sub_ppb', 'sub_la', 'sub_cpb', 'sub_pay', 'sub_pecos']
    df['max_sub'] = df[subscore_cols].max(axis=1)

    # Amplify extremes slightly
    scores = df['max_sub'] ** 1.2

    # Normalize
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)

    df['score'] = scores.fillna(0.5).clip(0, 1)
    df = df.drop_duplicates('npi')
    return df[['npi', 'score']]


def main():
    input_df = pd.read_csv(sys.stdin, dtype=str)
    if 'npi' not in input_df.columns:
        print("[detector] ERROR: stdin must have 'npi' column", file=sys.stderr)
        sys.exit(1)
    npi_list = input_df['npi'].dropna().str.strip().tolist()
    print(f"[detector] Scoring {len(npi_list):,} NPIs...", file=sys.stderr)
    result = score_providers(npi_list)
    result.to_csv(sys.stdout, index=False)
    print(f"[detector] Done. Scored {len(result):,} providers.", file=sys.stderr)


if __name__ == '__main__':
    main()
