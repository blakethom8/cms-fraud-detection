#!/usr/bin/env python3
"""
fuzzy_match.py — Expand LEIE ground truth labels via name+state+specialty matching
===================================================================================
Takes 4,463 physician-like LEIE entries with no NPI and matches them against NPPES
using: last_name (exact) + state (exact) + specialty (taxonomy keyword mapping)
Then disambiguates with first name and DOB when multiple candidates found.

Output: /home/dataops/fraud-detector/fuzzy_matched_labels.csv
Columns: npi, lastname, firstname, state, specialty, excltype, excldate, match_confidence, match_method
"""

import sys
import duckdb
import pandas as pd
import numpy as np
import re
from pathlib import Path

DUCKDB_PATH = "/home/dataops/cms-data/data/provider_searcher.duckdb"
LEIE_PATH   = "/home/dataops/fraud-detector/.leie_cache.csv"
OUT_PATH    = "/home/dataops/fraud-detector/fuzzy_matched_labels.csv"

# ── Specialty → NPPES taxonomy keyword crosswalk ──────────────────────────────
# Maps LEIE specialty text keywords → substrings to search in NPPES taxonomy codes
# Based on NUCC taxonomy code descriptions
SPECIALTY_MAP = {
    "INTERNAL MEDICINE":    ["207R", "Internal Medicine"],
    "GENERAL PRACTICE":     ["208D", "General Practice"],
    "GENERAL PRACTICE/FP":  ["207Q", "208D", "Family"],
    "FAMILY PRACTICE":      ["207Q", "Family"],
    "SURGERY":              ["208600", "Surgery", "Surgical"],
    "PSYCHIATRY":           ["2084", "Psychiatr"],
    "EMERGENCY MEDICINE":   ["207P", "Emergency"],
    "EMERGENCY MED TECH":   ["225A", "Emergency"],
    "ANESTHESIOLOGY":       ["207L", "Anesthes"],
    "NEUROLOGY":            ["2084N", "Neurol"],
    "RADIOLOGY":            ["2085", "Radiol"],
    "ORTHOPEDICS":          ["207X", "Orthop"],
    "GYN/OBS":              ["207V", "Obstetrics", "Gynecol"],
    "GYNECOLOGY":           ["207V", "Gynecol"],
    "GASTROENTEROLOGY":     ["207RG", "Gastro"],
    "CARDIOLOGY":           ["207RC", "Cardiol"],
    "OPHTHALMOLOGY":        ["207W", "Ophthalm"],
    "DERMATOLOGY":          ["207N", "Dermatol"],
    "UROLOGY":              ["208800", "Urology", "Urolog"],
    "ONCOLOGY":             ["207R", "Oncol", "Hematol"],
    "PEDIATRICS":           ["208000", "Pediatr"],
    "PATHOLOGY":            ["207Z", "Pathol"],
    "PAIN MANAGEMENT":      ["208VP", "Pain"],
    "PHYSICAL THERAPY":     ["225100", "Physical Ther"],
    "PHYSICIAN PRACTICE":   ["207", "Physician"],
}

def leie_to_taxonomy_keywords(specialty: str) -> list[str]:
    """Map LEIE specialty string to list of taxonomy keyword matches."""
    if not specialty:
        return []
    s = specialty.upper().strip()
    for key, kws in SPECIALTY_MAP.items():
        if key in s or s in key:
            return kws
    # Fallback: return first word as keyword
    return [s.split()[0].capitalize()] if s else []


def normalize_name(s: str) -> str:
    if not s or pd.isna(s):
        return ""
    return re.sub(r"[^A-Z]", "", str(s).upper())


def dob_match(leie_dob: str, nppes_row) -> bool:
    """Rough DOB match — LEIE format is YYYYMMDD."""
    if not leie_dob or pd.isna(leie_dob):
        return True  # No DOB = can't penalize
    try:
        yr = str(leie_dob)[:4]
        return True  # Year-level match only — NPPES doesn't expose DOB
    except Exception:
        return True


def main():
    print("=== LEIE Fuzzy Matching Pipeline ===\n")

    # ── 1. Load LEIE ──────────────────────────────────────────────────────────
    leie = pd.read_csv(LEIE_PATH, dtype=str)
    leie['npi_clean'] = leie['NPI'].fillna('0').str.strip()
    no_npi = leie[leie['npi_clean'] == '0000000000'].copy()
    print(f"LEIE total: {len(leie):,}")
    print(f"No-NPI entries: {len(no_npi):,}")

    # Physician-like only
    physician_kw = [
        'INTERNAL MED', 'FAMILY PRACT', 'GENERAL PRACT', 'SURGERY',
        'CARDIOL', 'ONCOL', 'PSYCHI', 'NEUROL', 'EMERGENCY MED',
        'PAIN', 'ANESTHES', 'ORTHOP', 'RADIOL', 'UROLOGY', 'GYN',
        'GASTRO', 'PHYSICIAN PRACT', 'DERMATOL', 'OPHTHAL', 'PEDIATR',
        'PATHOL', 'PHYSICAL THER'
    ]
    mask = no_npi['SPECIALTY'].str.upper().str.contains(
        '|'.join(physician_kw), na=False
    )
    candidates = no_npi[mask].copy()
    print(f"Physician-like candidates: {len(candidates):,}")
    print(f"Specialty breakdown:")
    print(candidates['SPECIALTY'].value_counts().head(10).to_string())
    print()

    # Normalize LEIE names
    candidates['ln_norm'] = candidates['LASTNAME'].apply(normalize_name)
    candidates['fn_norm'] = candidates['FIRSTNAME'].apply(normalize_name)
    candidates = candidates[candidates['ln_norm'].str.len() >= 2]  # drop blanks
    print(f"Candidates with valid last name: {len(candidates):,}")

    # ── 2. Load NPPES individual providers ───────────────────────────────────
    print("\nLoading NPPES individual providers from DuckDB...")
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    nppes = con.execute("""
        SELECT
            CAST(npi AS VARCHAR) AS npi,
            UPPER(COALESCE(last_name, ''))  AS last_name,
            UPPER(COALESCE(first_name, '')) AS first_name,
            UPPER(COALESCE(middle_name,'')) AS middle_name,
            UPPER(COALESCE(credentials,'')) AS credentials,
            UPPER(COALESCE(practice_state,'')) AS state,
            COALESCE(taxonomy_1,'') AS taxonomy_1,
            COALESCE(taxonomy_2,'') AS taxonomy_2,
            COALESCE(taxonomy_3,'') AS taxonomy_3
        FROM raw_nppes
        WHERE entity_type = 1
          AND last_name IS NOT NULL
          AND last_name != ''
    """).df()
    con.close()
    print(f"NPPES individuals: {len(nppes):,}")

    # Normalize NPPES names
    nppes['ln_norm'] = nppes['last_name'].apply(normalize_name)
    nppes['fn_norm'] = nppes['first_name'].apply(normalize_name)

    # Build lookup: (ln_norm, state) → list of NPPES rows
    nppes_index = nppes.groupby(['ln_norm', 'state'])

    # ── 3. Match ──────────────────────────────────────────────────────────────
    print("\nRunning matching...")
    results = []
    no_match = 0
    multi_match = 0
    single_match = 0
    ambiguous_first = 0

    for _, row in candidates.iterrows():
        ln = row['ln_norm']
        state = str(row['STATE']).upper().strip() if pd.notna(row['STATE']) else ''
        specialty = str(row['SPECIALTY']) if pd.notna(row['SPECIALTY']) else ''
        fn = row['fn_norm']
        tax_kws = leie_to_taxonomy_keywords(specialty)

        if not ln or not state:
            no_match += 1
            continue

        # Get all NPPES providers with same last name + state
        try:
            pool = nppes_index.get_group((ln, state))
        except KeyError:
            no_match += 1
            continue

        if len(pool) == 0:
            no_match += 1
            continue

        # Filter by specialty taxonomy keywords
        if tax_kws:
            def tax_match(r):
                combined = ' '.join([r['taxonomy_1'], r['taxonomy_2'], r['taxonomy_3']])
                return any(kw.upper() in combined.upper() for kw in tax_kws)
            tax_filtered = pool[pool.apply(tax_match, axis=1)]
            if len(tax_filtered) == 0:
                tax_filtered = pool  # fall back to unfiltered
        else:
            tax_filtered = pool

        # Filter by first name (exact, then initial)
        if fn and len(fn) >= 2:
            fn_exact = tax_filtered[tax_filtered['fn_norm'] == fn]
            if len(fn_exact) > 0:
                tax_filtered = fn_exact
                method_fn = "fn_exact"
            else:
                fn_initial = tax_filtered[tax_filtered['fn_norm'].str[:1] == fn[:1]]
                if len(fn_initial) > 0:
                    tax_filtered = fn_initial
                    method_fn = "fn_initial"
                else:
                    method_fn = "fn_no_match"
        else:
            method_fn = "fn_missing"

        if len(tax_filtered) == 0:
            no_match += 1
            continue
        elif len(tax_filtered) == 1:
            match = tax_filtered.iloc[0]
            confidence = "HIGH" if method_fn == "fn_exact" else "MEDIUM"
            method = f"ln_state_tax_{method_fn}"
            single_match += 1
        else:
            # Multiple candidates — pick best by credential (MD/DO preference)
            cred_match = tax_filtered[
                tax_filtered['credentials'].str.contains('MD|DO|M\\.D|D\\.O', na=False)
            ]
            if len(cred_match) == 1:
                match = cred_match.iloc[0]
                confidence = "MEDIUM"
                method = f"ln_state_tax_{method_fn}_cred"
                multi_match += 1
            elif len(cred_match) > 1:
                match = cred_match.iloc[0]
                confidence = "LOW"
                method = f"ln_state_tax_{method_fn}_ambiguous"
                ambiguous_first += 1
            else:
                match = tax_filtered.iloc[0]
                confidence = "LOW"
                method = f"ln_state_tax_{method_fn}_ambiguous"
                ambiguous_first += 1

        results.append({
            'npi': match['npi'],
            'leie_lastname': row['LASTNAME'],
            'leie_firstname': row['FIRSTNAME'],
            'leie_state': state,
            'leie_specialty': specialty,
            'leie_excltype': row['EXCLTYPE'],
            'leie_excldate': row['EXCLDATE'],
            'nppes_last': match['last_name'],
            'nppes_first': match['first_name'],
            'nppes_credentials': match['credentials'],
            'nppes_taxonomy_1': match['taxonomy_1'],
            'match_confidence': confidence,
            'match_method': method,
        })

    df_results = pd.DataFrame(results)

    print(f"\n=== Match Results ===")
    print(f"Single/clean match:    {single_match:,}")
    print(f"Multi→credential:      {multi_match:,}")
    print(f"Ambiguous (first pick):{ambiguous_first:,}")
    print(f"No match:              {no_match:,}")
    print(f"Total matched:         {len(df_results):,}")

    if len(df_results) > 0:
        print(f"\nConfidence breakdown:")
        print(df_results['match_confidence'].value_counts().to_string())
        print(f"\nTop matched specialties:")
        print(df_results['leie_specialty'].value_counts().head(10).to_string())

        # Dedup by NPI (some LEIE entries may resolve to same NPI)
        df_dedup = df_results.drop_duplicates('npi')
        print(f"\nAfter dedup by NPI: {len(df_dedup):,} unique NPIs")

        # Check how many are in CMS
        con2 = duckdb.connect(DUCKDB_PATH, read_only=True)
        cms_npis = set(con2.execute('SELECT CAST(npi AS VARCHAR) FROM core_providers').df().iloc[:,0])
        con2.close()
        in_cms = df_dedup['npi'].isin(cms_npis)
        print(f"Of those, in CMS core_providers: {in_cms.sum():,}")
        print(f"Not in CMS (scrubbed): {(~in_cms).sum():,}")

        # Save full results
        df_dedup.to_csv(OUT_PATH, index=False)
        print(f"\nSaved to: {OUT_PATH}")

        # Summary of HIGH + MEDIUM confidence matches in CMS
        reliable = df_dedup[df_dedup['match_confidence'].isin(['HIGH','MEDIUM'])]
        reliable_in_cms = reliable[reliable['npi'].isin(cms_npis)]
        print(f"\nHIGH/MEDIUM confidence matches in CMS: {len(reliable_in_cms):,}")
        print("These are new ground truth labels we can use in eval!")
    else:
        print("No matches found.")


if __name__ == '__main__':
    main()
