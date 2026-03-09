#!/usr/bin/env python3
"""
eval.py — Fixed Oracle for Healthcare Fraud Autoresearch
=========================================================
DO NOT MODIFY THIS FILE. It is the ground-truth evaluator.

Loads OIG LEIE exclusion list (ground truth fraud labels), joins against
CMS DuckDB provider data, and measures AUC-ROC of detector.py's scores.

Usage:
    python eval.py                    # runs full eval, appends to results.tsv
    python eval.py --dry-run          # prints stats without writing results
    python eval.py --describe-labels  # show LEIE exclusion reason breakdown
"""

import sys
import csv
import os
import argparse
import subprocess
import datetime
import hashlib
import requests
import pandas as pd
import duckdb
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Config ──────────────────────────────────────────────────────────────────
DUCKDB_PATH = "/home/dataops/cms-data/data/provider_searcher.duckdb"
LEIE_URL    = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"
LEIE_CACHE  = "/home/dataops/fraud-detector/.leie_cache.csv"
RESULTS_TSV = "/home/dataops/fraud-detector/results.tsv"
DETECTOR    = "/home/dataops/fraud-detector/detector.py"


def fetch_leie(force_refresh=False):
    """Download LEIE CSV (cached for 24h)."""
    if not force_refresh and os.path.exists(LEIE_CACHE):
        age_hours = (datetime.datetime.now().timestamp() - os.path.getmtime(LEIE_CACHE)) / 3600
        if age_hours < 24:
            print(f"[eval] Using cached LEIE ({age_hours:.1f}h old)")
            return pd.read_csv(LEIE_CACHE, dtype=str)

    print("[eval] Downloading fresh LEIE from OIG...")
    r = requests.get(LEIE_URL, timeout=60)
    r.raise_for_status()
    with open(LEIE_CACHE, "wb") as f:
        f.write(r.content)
    print(f"[eval] LEIE downloaded: {len(r.content):,} bytes")
    return pd.read_csv(LEIE_CACHE, dtype=str)


def load_cms_npis():
    """Load all NPIs from CMS DuckDB with basic features for label matching."""
    print("[eval] Loading NPIs from CMS DuckDB...")
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    df = con.execute("""
        SELECT
            p.npi,
            p.last_org_name,
            p.first_name,
            p.state,
            p.zip5
        FROM core_providers p
        WHERE p.npi IS NOT NULL
    """).df()
    con.close()
    print(f"[eval] Loaded {len(df):,} providers from CMS")
    return df


def build_labels(leie_df, cms_df):
    """
    Join LEIE exclusions to CMS providers by NPI.
    Returns cms_df with a 'label' column: 1=excluded, 0=not excluded.
    LEIE has ~70K excluded providers; most won't match CMS (they've been
    removed from Medicare), but enough will for a meaningful signal.
    """
    print("[eval] Building fraud labels...")

    # LEIE NPI column is called 'NPI' — normalize
    leie_npis = set(
        leie_df['NPI'].dropna()
        .str.strip()
        .str.replace(r'\D', '', regex=True)  # digits only
        .loc[lambda s: s.str.len() == 10]
    )
    print(f"[eval] LEIE excluded NPIs with valid 10-digit NPI: {len(leie_npis):,}")

    cms_df = cms_df.copy()
    cms_df['npi_str'] = cms_df['npi'].astype(str).str.strip()
    cms_df['label'] = cms_df['npi_str'].isin(leie_npis).astype(int)

    n_pos = cms_df['label'].sum()
    n_neg = len(cms_df) - n_pos
    prevalence = n_pos / len(cms_df) * 100
    print(f"[eval] Labels: {n_pos:,} positive (excluded), {n_neg:,} negative | prevalence: {prevalence:.3f}%")

    if n_pos < 10:
        print("[eval] WARNING: Very few positive labels matched. Check LEIE NPI format.")

    return cms_df


def run_detector(cms_df):
    """
    Run detector.py to get fraud scores for each NPI.
    detector.py must accept a CSV of NPIs on stdin and write NPI,score to stdout.
    """
    print(f"[eval] Running detector: {DETECTOR}")

    # Pass NPIs to detector via stdin
    npi_csv = cms_df[['npi']].to_csv(index=False)

    result = subprocess.run(
        ["/home/dataops/cms-data/.venv/bin/python3", DETECTOR],
        input=npi_csv,
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        print(f"[eval] ERROR: detector.py failed:\n{result.stderr[:2000]}")
        sys.exit(1)

    if result.stderr.strip():
        print(f"[eval] detector stderr:\n{result.stderr[:500]}")

    # Parse NPI,score output
    scores_df = pd.read_csv(
        pd.io.common.StringIO(result.stdout),
        dtype={'npi': str}
    )
    scores_df.columns = scores_df.columns.str.lower().str.strip()

    if 'npi' not in scores_df.columns or 'score' not in scores_df.columns:
        print(f"[eval] ERROR: detector.py must output CSV with columns: npi, score")
        print(f"[eval] Got columns: {list(scores_df.columns)}")
        sys.exit(1)

    print(f"[eval] Received {len(scores_df):,} scores from detector")
    return scores_df


def compute_metrics(labels_df, scores_df):
    """Merge labels + scores, compute AUC-ROC and Average Precision."""
    merged = labels_df[['npi_str', 'label']].merge(
        scores_df.rename(columns={'npi': 'npi_str'}),
        on='npi_str',
        how='inner'
    )

    n_matched = len(merged)
    n_pos = merged['label'].sum()
    print(f"[eval] Matched {n_matched:,} NPIs with scores | positives: {n_pos:,}")

    if n_pos == 0:
        print("[eval] ERROR: No positive labels in scored set — can't compute AUC")
        sys.exit(1)

    auc = roc_auc_score(merged['label'], merged['score'])
    ap  = average_precision_score(merged['label'], merged['score'])

    print(f"\n{'='*50}")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  Avg Precision:     {ap:.4f}")
    print(f"  Providers scored:  {n_matched:,}")
    print(f"  Positives:         {n_pos:,}")
    print(f"{'='*50}\n")

    return auc, ap, n_matched, int(n_pos)


def get_git_commit():
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True,
            cwd='/home/dataops/fraud-detector'
        )
        return result.stdout.strip() or 'no-git'
    except Exception:
        return 'no-git'


def append_results(auc, ap, n_scored, n_pos, description=""):
    """Append a row to results.tsv."""
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    commit = get_git_commit()

    # Create file with header if it doesn't exist
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, 'w') as f:
            f.write("timestamp\tcommit\tauc_roc\tavg_precision\tn_scored\tn_positives\tdescription\n")

    with open(RESULTS_TSV, 'a') as f:
        f.write(f"{now}\t{commit}\t{auc:.4f}\t{ap:.4f}\t{n_scored}\t{n_pos}\t{description}\n")

    print(f"[eval] Results appended to {RESULTS_TSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Skip writing to results.tsv')
    parser.add_argument('--describe-labels', action='store_true', help='Show LEIE exclusion reason breakdown')
    parser.add_argument('--description', default='', help='Description for this run')
    args = parser.parse_args()

    leie_df = fetch_leie()

    if args.describe_labels:
        print("\nLEIE Exclusion Reasons:")
        print(leie_df['EXCLTYPE'].value_counts().head(20).to_string())
        print()

    cms_df    = load_cms_npis()
    labels_df = build_labels(leie_df, cms_df)
    scores_df = run_detector(labels_df)
    auc, ap, n_scored, n_pos = compute_metrics(labels_df, scores_df)

    if not args.dry_run:
        append_results(auc, ap, n_scored, n_pos, args.description)
    else:
        print("[eval] Dry run — results not written.")


if __name__ == '__main__':
    main()
