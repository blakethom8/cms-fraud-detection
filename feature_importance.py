#!/usr/bin/env python3
"""
feature_importance.py — Subscore Analysis & Feature Impact
============================================================
Analyzes which subscores best separate LEIE-excluded providers
from the general population. Reports:
1. Per-subscore AUC (individual predictive power)
2. Score distributions: positives vs negatives
3. Top 100 suspects with subscore breakdown ("why flagged")
4. Correlation between subscores
"""

import sys
import duckdb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DUCKDB_PATH = "/home/dataops/cms-data/data/provider_searcher.duckdb"
LEIE_PATH   = "/home/dataops/fraud-detector/.leie_cache.csv"

HIGH_VOLUME_SPECIALTIES = {
    'hematology-oncology','medical oncology','hematology','oncology',
    'radiation oncology','gynecological oncology','pharmacy',
    'surgical oncology','neuro-oncology'
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

def main():
    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    # ── Load all data ──────────────────────────────────────────────────────
    print("\n[1] Loading data...")

    part_b = con.execute("""
        SELECT
            CAST(Rndrng_NPI AS VARCHAR) AS npi,
            LOWER(Rndrng_Prvdr_Type) AS specialty,
            Rndrng_Prvdr_Last_Org_Name AS last_name,
            Rndrng_Prvdr_First_Name AS first_name,
            Rndrng_Prvdr_State_Abrvtn AS state,
            Tot_Mdcr_Pymt_Amt AS total_payment,
            Tot_Srvcs AS total_svc,
            Tot_Benes AS total_benes,
            CAST(Tot_Srvcs AS DOUBLE) / NULLIF(Tot_Benes, 0) AS svc_per_bene,
            CAST(Tot_Mdcr_Pymt_Amt AS DOUBLE) / NULLIF(Tot_Benes, 0) AS pay_per_bene
        FROM raw_physician_by_provider
        WHERE Rndrng_NPI IS NOT NULL AND Tot_Benes > 0
    """).df()
    part_b['npi'] = part_b['npi'].astype(str)

    part_d = con.execute("""
        SELECT
            CAST(PRSCRBR_NPI AS VARCHAR) AS npi,
            LOWER(Prscrbr_Type) AS specialty,
            Tot_Drug_Cst AS total_drug_cost,
            Tot_Benes AS total_benes_d,
            COALESCE(Opioid_LA_Prscrbr_Rate, 0) AS opioid_la_rate
        FROM raw_part_d_by_provider
        WHERE PRSCRBR_NPI IS NOT NULL
    """).df()
    part_d['npi'] = part_d['npi'].astype(str)

    payments = con.execute("""
        SELECT
            CAST(Covered_Recipient_NPI AS VARCHAR) AS npi,
            SUM(CAST(Total_Amount_of_Payment_USDollars AS DOUBLE)) AS total_payments_usd
        FROM raw_open_payments_general
        WHERE Covered_Recipient_NPI IS NOT NULL
        GROUP BY Covered_Recipient_NPI
    """).df()
    payments['npi'] = payments['npi'].astype(str)

    pecos_npis = set(
        con.execute("SELECT DISTINCT CAST(NPI AS VARCHAR) FROM raw_pecos_enrollment WHERE NPI IS NOT NULL")
        .df().iloc[:,0]
    )

    # NPPES for names/location
    nppes = con.execute("""
        SELECT CAST(npi AS VARCHAR) as npi, last_name, first_name,
               practice_city, practice_state, taxonomy_1
        FROM raw_nppes WHERE entity_type = 1
    """).df()
    nppes['npi'] = nppes['npi'].astype(str)
    con.close()

    # ── LEIE labels ────────────────────────────────────────────────────────
    print("[2] Loading LEIE labels...")
    leie = pd.read_csv(LEIE_PATH, dtype=str)
    leie_npis = set(
        leie['NPI'].dropna().str.strip()
        .loc[lambda s: s != '0000000000']
        .loc[lambda s: s.str.len() == 10]
    )
    print(f"    LEIE excluded NPIs: {len(leie_npis):,}")

    # ── Build scored dataframe ─────────────────────────────────────────────
    print("[3] Computing subscores...")

    df = part_b[['npi','specialty','last_name','first_name','state',
                 'total_benes','total_svc','total_payment',
                 'svc_per_bene','pay_per_bene']].copy()

    df['is_high_vol'] = df['specialty'].isin(HIGH_VOLUME_SPECIALTIES)

    df['spb_z']  = zscore_within_group(df, 'specialty', 'svc_per_bene')
    df['ppb_z']  = zscore_within_group(df, 'specialty', 'pay_per_bene')
    df['sub_spb'] = sigmoid(df['spb_z'], scale=2)
    df.loc[df['is_high_vol'], 'sub_spb'] *= 0.5
    df['sub_ppb'] = sigmoid(df['ppb_z'], scale=2)

    # Part D subscores
    part_d['cpb'] = part_d['total_drug_cost'] / (part_d['total_benes_d'] + 1)
    part_d['la_z']  = zscore_within_group(part_d, 'specialty', 'opioid_la_rate')
    part_d['cpb_z'] = zscore_within_group(part_d, 'specialty', 'cpb')
    part_d['sub_la']  = sigmoid(part_d['la_z'], scale=2)
    part_d['sub_cpb'] = sigmoid(part_d['cpb_z'], scale=2)

    df = df.merge(part_d[['npi','sub_la','sub_cpb','opioid_la_rate']], on='npi', how='left')
    df['sub_la']  = df['sub_la'].fillna(0.5)
    df['sub_cpb'] = df['sub_cpb'].fillna(0.5)

    # Open Payments
    payments['sub_pay'] = np.log1p(payments['total_payments_usd'].clip(lower=0))
    payments['sub_pay'] /= (payments['sub_pay'].max() + 1e-9)
    df = df.merge(payments[['npi','sub_pay','total_payments_usd']], on='npi', how='left')
    df['sub_pay'] = df['sub_pay'].fillna(0)
    df['total_payments_usd'] = df['total_payments_usd'].fillna(0)

    # PECOS gap
    df['no_pecos'] = (~df['npi'].isin(pecos_npis)).astype(float)
    df['sub_pecos'] = df['no_pecos'] * (df['total_benes'] > 50).astype(float)

    # Final score
    subscore_cols = ['sub_spb','sub_ppb','sub_la','sub_cpb','sub_pay','sub_pecos']
    df['max_sub'] = df[subscore_cols].max(axis=1)
    df['score']   = (df['max_sub'] ** 1.2)
    s_min, s_max = df['score'].min(), df['score'].max()
    df['score'] = (df['score'] - s_min) / (s_max - s_min + 1e-9)

    # Labels
    df['label'] = df['npi'].isin(leie_npis).astype(int)
    positives = df[df['label']==1]
    negatives = df[df['label']==0]

    print(f"    Total providers: {len(df):,}")
    print(f"    Positives (LEIE): {len(positives):,}")

    # ── Per-subscore AUC ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INDIVIDUAL SUBSCORE AUC (standalone predictive power)")
    print("=" * 60)
    subscore_names = {
        'sub_spb':  'Services/Beneficiary (specialty-normalized)',
        'sub_ppb':  'Payment/Beneficiary (specialty-normalized)',
        'sub_la':   'Long-Acting Opioid Rate (specialty-normalized)',
        'sub_cpb':  'Drug Cost/Beneficiary (specialty-normalized)',
        'sub_pay':  'Open Payments (industry $)',
        'sub_pecos':'PECOS Enrollment Gap',
        'score':    '>>> FINAL COMBINED SCORE <<<',
    }
    aucs = {}
    for col, name in subscore_names.items():
        if col in df.columns:
            try:
                auc = roc_auc_score(df['label'], df[col])
                aucs[col] = auc
                marker = ' ◄ BEST' if auc == max(aucs.values()) else ''
                print(f"  {auc:.4f}  {name}{marker}")
            except Exception as e:
                print(f"  ERROR  {name}: {e}")

    # ── Score distribution: positives vs negatives ─────────────────────────
    print("\n" + "=" * 60)
    print("SCORE DISTRIBUTIONS: Positives vs Negatives")
    print("=" * 60)
    print(f"\n{'Subscore':<12} {'Pos mean':>10} {'Pos p50':>8} {'Pos p95':>8} | {'Neg mean':>10} {'Neg p50':>8} {'Neg p95':>8}")
    print("-" * 75)
    for col in subscore_cols:
        pos_vals = positives[col].dropna()
        neg_vals = negatives[col].dropna()
        print(f"{col:<12} {pos_vals.mean():>10.3f} {pos_vals.median():>8.3f} "
              f"{pos_vals.quantile(0.95):>8.3f} | "
              f"{neg_vals.mean():>10.3f} {neg_vals.median():>8.3f} "
              f"{neg_vals.quantile(0.95):>8.3f}")

    # ── Which subscore is "max" for top suspects ───────────────────────────
    print("\n" + "=" * 60)
    print("WHAT DRIVES THE FLAGS: Which subscore is MAX for top 500?")
    print("=" * 60)
    top500 = df.nlargest(500, 'score')
    driving_sub = top500[subscore_cols].idxmax(axis=1)
    print(driving_sub.value_counts().to_string())
    print("\nFor LEIE positives — which subscore drives their score?")
    pos_driving = positives[subscore_cols].idxmax(axis=1)
    print(pos_driving.value_counts().to_string())

    # ── Top 100 suspect table ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TOP 100 SUSPECTS WITH SUBSCORE BREAKDOWN")
    print("=" * 60)

    top100 = df.nlargest(100, 'score').merge(
        nppes[['npi','last_name','first_name','practice_city','practice_state','taxonomy_1']],
        on='npi', how='left', suffixes=('_b','_n')
    )

    # Best name: NPPES preferred
    top100['display_name'] = (
        top100['first_name_n'].fillna(top100.get('first_name_b', top100['last_name_b'])) + ' ' +
        top100['last_name_n'].fillna(top100['last_name_b'])
    ).str.title().str.strip()
    top100['city_state'] = (
        top100['practice_city'].fillna('').str.title() + ', ' +
        top100['practice_state'].fillna(top100['state'])
    )
    top100['driving_sub'] = top100[subscore_cols].idxmax(axis=1)
    top100['is_leie'] = top100['npi'].isin(leie_npis)

    print(f"\n{'Rank':<5} {'Score':>6} {'NPI':<12} {'Name':<28} {'Specialty':<25} {'Location':<20} {'Max Signal':<12} {'LEIE?'}")
    print("-" * 125)
    for rank, (_, row) in enumerate(top100.iterrows(), 1):
        leie_flag = '🚨 YES' if row['is_leie'] else ''
        print(f"{rank:<5} {row['score']:>6.3f} {row['npi']:<12} "
              f"{str(row['display_name'])[:27]:<28} "
              f"{str(row['specialty'])[:24]:<25} "
              f"{str(row['city_state'])[:19]:<20} "
              f"{row['driving_sub']:<12} {leie_flag}")

    # ── Save enriched top suspects for pipeline ────────────────────────────
    suspects_out = top100[[
        'npi','display_name','specialty','city_state','state',
        'score','sub_spb','sub_ppb','sub_la','sub_cpb','sub_pay','sub_pecos',
        'driving_sub','total_benes','total_svc','total_payment',
        'total_payments_usd','opioid_la_rate','is_leie'
    ]].copy()

    suspects_out.to_csv('/home/dataops/fraud-detector/top_suspects.csv', index=False)
    print(f"\n[Saved] /home/dataops/fraud-detector/top_suspects.csv ({len(suspects_out)} rows)")

    # Also save full scored df for pipeline use
    df[['npi','score','sub_spb','sub_ppb','sub_la','sub_cpb',
        'sub_pay','sub_pecos','driving_sub','label',
        'specialty','last_name','first_name','state','total_benes']]\
        .to_csv('/home/dataops/fraud-detector/all_scores.csv', index=False)
    print(f"[Saved] /home/dataops/fraud-detector/all_scores.csv ({len(df)} rows)")

if __name__ == '__main__':
    main()
