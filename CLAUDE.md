# CLAUDE.md — Agent Instructions

You are improving `detector.py` to maximize AUC-ROC against the OIG LEIE ground truth.

## The Rules

1. **DO NOT modify `eval.py`** — it is the fixed oracle
2. `detector.py` reads NPI CSV from stdin, outputs `npi,score` CSV to stdout
3. Run `python eval.py --dry-run` to test without logging
4. Run `python eval.py --description "your description"` to log a real result
5. Check `results.tsv` to see all previous attempts

## Current Best

See `results.tsv` — aim to beat the highest AUC-ROC in that file.

## Data Available

DuckDB at `/home/dataops/cms-data/data/provider_searcher.duckdb`:
- `raw_physician_by_provider` — Part B billing (Rndrng_NPI, Rndrng_Prvdr_Type, Tot_Srvcs, Tot_Benes, Tot_Mdcr_Pymt_Amt)
- `raw_part_d_by_provider` — Part D prescribing (PRSCRBR_NPI, Prscrbr_Type, Tot_Drug_Cst, Tot_Benes, Opioid_LA_Prscrbr_Rate)
- `raw_physician_by_provider_and_service` — HCPCS-level billing (9.6M rows)
- `raw_open_payments_general` — Industry payments (14.7M rows, Covered_Recipient_NPI)
- `raw_nppes` — Provider registry (7.1M, NPI, taxonomy_1, practice_address_1, practice_phone)
- `raw_pecos_enrollment` — Medicare enrollment (2.54M, NPI, PROVIDER_TYPE_DESC — has duplicate NPIs)
- `raw_order_and_referring` — Referral patterns (1.99M rows)
- `core_providers` — Cleaned provider table (1.2M, npi, provider_type, state, zip5)
- `taxonomy_lookup` — Taxonomy code → specialty name mapping

## What Has Worked

- **Services-per-bene z-score within specialty** — the single strongest signal
- **LA opioid rate z-score within specialty** — catches pill mills
- **`max(subscores)`** — taking the maximum across all feature subscores beats weighted averaging
- High-volume specialties (oncology, hematology) need a dampening factor (0.4-0.5x)

## What Hasn't Worked

- HCPCS concentration HHI — adds noise, hurts AUC
- Taxonomy mismatch — too many false positives
- Raw percentile buckets — z-scores are better
- Adding too many features — each new weak feature dilutes the strong ones

## Key Gotchas

- `raw_pecos_enrollment` has up to 75 rows per NPI — always `SELECT DISTINCT NPI`
- HCPCS table has 9.6M rows — compute HHI in SQL, not pandas apply()
- Sigmoid scale=2.0 works better than 1.5 for z-score transformation
- Always fill NaN with 0.5 (neutral) before outputting scores
- Deduplicate output on NPI before returning

## Python Environment

```bash
/home/dataops/cms-data/.venv/bin/python3
```
