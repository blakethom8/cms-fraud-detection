# CMS Healthcare Fraud Detection

> Applying Karpathy's autoresearch pattern to Medicare fraud detection — an AI agent iterating overnight on 90M+ CMS claim records against OIG LEIE ground truth labels.

**Live AUC progression: 0.5561 → 0.81+ in one evening of iterations**

---

## What This Is

An experiment in autonomous research loops for healthcare fraud detection. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), we applied the same pattern to a real-world problem with genuine stakes: finding fraudulent healthcare providers in Medicare claims data.

**The loop:**
1. `eval.py` — fixed oracle, downloads OIG LEIE exclusion list, measures AUC-ROC
2. `detector.py` — agent-editable scoring logic
3. Agent iterates on `detector.py` overnight, eval tracks progress in `results.tsv`

**The data:**
- ~90M rows across 30 CMS tables in DuckDB (6GB)
- 1.2M Medicare providers scored per run
- 181 ground-truth LEIE-excluded providers matched in CMS
- Data sources: Part B claims, Part D prescribing, Open Payments, NPPES, PECOS

---

## Results

| Version | AUC-ROC | Key Change |
|---------|---------|------------|
| Baseline | 0.5561 | Raw billing z-scores |
| V2 | 0.7695 | Specialty-normalized services-per-beneficiary |
| V7 | 0.7904 | Ensemble: max(subscores) + weighted mean |
| V12 | 0.8098 | Pure max of all subscores |
| **V14** | **0.81+** | Max amplified with power transform |

**Key discovery:** Fraud shows up as being **extreme on any single dimension**, not just above average across all dimensions. Taking `max(subscores)` across billing patterns, opioid rates, and enrollment gaps dramatically outperforms weighted averaging.

---

## Key Findings

### What the Model Learns
- **Services-per-beneficiary within specialty** is the single strongest fraud signal. Excluded providers bill 5-50x more services per patient than peers in the same specialty.
- **Long-acting opioid prescribing rate** (normalized within specialty) catches pill mills that simple opioid rates miss.
- **PECOS enrollment gaps** — providers actively billing Medicare but not enrolled in PECOS — are a real signal.
- Adding more features (HCPCS concentration, upcoding ratios, taxonomy mismatch) can hurt AUC if noisy — less is more.

### Real-World Validation (Web Research)
Top-scoring providers investigated via web search:

| Provider | Score | Finding |
|----------|-------|---------|
| Robert Morton MD (Ada, OK psychiatrist) | 0.88 | **NOT fraud** — prescribes ultra-expensive specialty psych drugs (Ingrezza $7,700/claim for tardive dyskinesia). Legitimate specialty practice. |
| Anne Greist MD (Indianapolis hematologist) | Top | **NOT fraud** — IU Health cancer center, 48K services/bene from hemophilia/blood disorder infusions |
| Kashif Ali MD (Greenbelt MD oncologist) | 0.85 | **Needs investigation** — 629,920 services on 1,308 patients (481/patient). Even for infusion oncology, extreme outlier. |
| Harsha Vyas MD (Dublin GA oncologist) | 0.90 | **Needs investigation** — 322,669 services on 764 patients (422/patient). CCMG Cancer Center. |
| "Indianapolis Cluster" | Top | **NOT fraud** — cluster of hematology-oncology providers all at IU Health / academic cancer center |
| "Farmington Hills MI cluster" (27777 Inkster Rd, 15K providers) | — | **NOT fraud** — Centria Healthcare ABA therapy, legitimate large employer |

### Key Model Limitation
The LEIE contains ~70K excluded providers, but most are **non-physicians** (nurses, aides, personal care workers) who don't bill Part B directly. Only 181 matched our CMS physician universe — a small signal in 1.2M providers (0.015% prevalence). This makes AUC a useful but imperfect metric.

---

## Data Sources

| Source | Rows | Purpose |
|--------|------|---------|
| CMS Part B Physician Claims | 1.26M providers | Billing volume and service counts |
| CMS Part D Prescribing | 1.38M providers | Drug costs, opioid patterns |
| Open Payments | 14.7M payments | Industry financial relationships |
| NPPES NPI Registry | 7.1M providers | Provider demographics, taxonomy |
| PECOS Enrollment | 2.54M records | Medicare enrollment status |
| OIG LEIE | ~70K exclusions | Ground truth fraud labels |

All CMS data is publicly available at [data.cms.gov](https://data.cms.gov).

---

## How to Run

### Requirements
- DuckDB database (see [cms-data](https://github.com/blakethom8/cms-data) for pipeline)
- Python 3.12+ with venv
- ~6GB disk for DuckDB

```bash
pip install duckdb pandas numpy scikit-learn requests

# Run evaluation against LEIE ground truth
python eval.py

# Dry run (no results logged)
python eval.py --dry-run

# Describe LEIE exclusion type breakdown
python eval.py --describe-labels
```

### Running a Research Loop
```bash
# 1. Pick a strategy
cat strategies/strategy-C-domain-first.md

# 2. Open Claude Code pointed at this directory
claude  # or: codex

# 3. Prompt the agent:
# "Read strategy-C-domain-first.md and improve detector.py to beat the current best AUC in results.tsv"

# 4. Wake up. Review results.tsv.
```

---

## Strategies

Three research strategies defined:

- **strategy-A-conservative.md** — Statistical billing outliers only (safe starting point)
- **strategy-B-aggressive.md** — Kitchen sink approach, find the AUC ceiling
- **strategy-C-domain-first.md** — Fraud typologies (pill mills, kickback rings, phantom billing, quality paradox)

---

## Architecture

```
eval.py (FIXED — do not edit)
  └── Downloads OIG LEIE CSV (ground truth)
  └── Loads CMS NPI universe from DuckDB
  └── Runs detector.py via subprocess (stdin/stdout CSV)
  └── Computes AUC-ROC, Average Precision
  └── Appends to results.tsv

detector.py (AGENT EDITS THIS)
  └── Reads NPI list from stdin
  └── Queries CMS DuckDB for features
  └── Outputs NPI + score (0-1) to stdout

results.tsv
  └── timestamp | commit | auc_roc | avg_precision | description
```

---

## Next Steps

- [ ] Activate LLM matching module for entity resolution
- [ ] Add network analysis — providers who refer to each other in suspicious patterns
- [ ] Train a proper ML model (gradient boosting) on the hand-crafted features
- [ ] Real-time web validation: auto-search top suspects, check for news/OIG press releases
- [ ] Frontend explorer app for interactive validation
- [ ] Extend to DME (durable medical equipment) fraud — different billing patterns

---

## Related

- [cms-data](https://github.com/blakethom8/cms-data) — Pipeline that built the DuckDB
- [provider-search](https://github.com/blakethom8/provider-search) — Provider search tool using this data
- Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — The pattern that inspired this

---

*Built in one evening using CMS public data + OIG exclusion list. All data is publicly available.*
