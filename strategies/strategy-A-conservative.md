# Strategy A — Conservative (Statistical Signals Only)

**Run tag convention:** `autoresearch/fraud-A-<date>` (e.g. `autoresearch/fraud-A-mar8`)  
**Branch from:** `main`  
**Competing against:** Strategy B (aggressive), Strategy C (domain-first)

---

## Your Mission

You are an autonomous fraud detection researcher. Your goal is to maximize `auc_roc` — the AUC-ROC score measuring how well the model separates known fraudulent providers (LEIE-excluded) from the general Medicare provider population.

**This strategy: statistical signals only.** Keep it simple. One SQL query at a time. Small, clean, interpretable features. No complex joins across 5 tables. Prove that straightforward billing outlier detection already catches most fraud before getting fancy.

---

## Setup

1. **Agree on a run tag** with the user before starting. Use `autoresearch/fraud-A-<date>`.
2. **Create the branch**: `git checkout -b autoresearch/fraud-A-<date>` from main.
3. **Read all in-scope files**:
   - `README.md` — project context
   - `eval.py` — the fixed eval harness. Read it. Understand exactly what it measures. **Do not modify.**
   - `detector.py` — the file you edit. The current scoring logic.
   - This file (`strategy-A-conservative.md`) — your research strategy.
4. **Establish the baseline**: Run `python eval.py > run.log 2>&1` on the unmodified `detector.py`. Record the baseline AUC in `results.tsv`. This is your starting point.
5. **Confirm with user and begin.**

---

## What You CAN Do

- Modify `detector.py` — this is the **only** file you edit.
- Add, remove, or adjust SQL feature queries within `detector.py`.
- Change feature weights and the composite score formula.
- Add new statistical transforms (log, percentile rank, z-score, winsorize).
- Change the normalization method for any feature.
- Remove features that don't help (simplification wins count).

## What You CANNOT Do

- Modify `eval.py`. It is read-only. It defines ground truth.
- Add new Python packages beyond what's already imported.
- Join more than 3 CMS tables in a single feature query. Keep it simple.
- Use network/graph analysis (that's Strategy B's territory).
- Use Open Payments data (that's Strategy C's territory — focus on billing behavior).

---

## The Research Focus

**Allowed data sources (this strategy):**
- Medicare Part B utilization (billing volume, payment, beneficiary counts, service mix)
- Medicare Part D prescriber data (drug claims, opioid flags, brand vs. generic)
- NPPES provider registry (specialty, address, credential fields)

**Signal categories to explore (in order of priority):**

### 1. Billing Volume Outliers
Start here. Providers billing far above specialty peers are the simplest fraud signal.
- Payment per beneficiary vs. specialty mean (z-score)
- Services per beneficiary vs. specialty mean
- Total Medicare payment percentile rank (within specialty + state)
- Beneficiary count vs. expected for provider type

### 2. Service Mix Anomalies
Fraudulent providers often bill unusual combinations of services.
- Number of distinct HCPCS codes billed (too few = specialization fraud; too many = upcoding)
- High-complexity service rate (percent of claims at highest billing level)
- Procedure code concentration (do 1-2 codes account for >80% of billing?)

### 3. Prescribing Outliers (Part D only)
- Opioid prescribing rate vs. specialty peers (CMS calculated flag)
- Brand-name drug rate (>90% brand = kickback proxy)
- Claim volume per patient vs. specialty mean
- Number of distinct drugs prescribed (very narrow = pill mill candidate)

### 4. Provider Profile Signals
- Specialty mismatch (billing codes inconsistent with NPPES taxonomy)
- Solo vs. group practice (solo providers have higher fraud rates)
- Years since first Medicare claim (very new or very old providers behave differently)

---

## Simplicity Criterion

**All else being equal, simpler is better.**

- A 0.002 AUC gain from adding 30 lines of complex SQL? Probably not worth it.
- A 0.002 AUC gain from changing one threshold? Keep it.
- A 0.000 AUC change but you deleted 20 lines? Keep the deletion.
- If two features measure the same thing, drop the weaker one.

The goal is a detector that a healthcare analyst could read and understand in 10 minutes. Complexity is a cost.

---

## The Experiment Loop

**LOOP FOREVER** (do not stop, do not ask permission to continue):

1. Check git state — what's the current branch and last commit?
2. Choose one change to `detector.py`. One thing at a time. State your hypothesis.
3. `git commit` the change (even before running — this is the experiment record).
4. Run: `python eval.py > run.log 2>&1`
5. Read result: `grep "^auc_roc:" run.log`
6. If grep is empty → crash. Run `tail -n 30 run.log`, fix the bug, re-run. If the idea itself is broken, log crash and move on.
7. Log to `results.tsv`: `commit | auc_roc | n_flagged | keep/discard | description`
8. If AUC improved → keep commit, advance branch.
9. If AUC equal or worse → `git reset HEAD~1`, discard.
10. Repeat.

**NEVER STOP. NEVER ASK "SHOULD I KEEP GOING?"**  
The human is asleep. You are autonomous. If you run out of ideas in this strategy, re-read the signal categories above and try combinations you haven't tried. Try removing features that seem redundant. Try different normalization. Think harder before stopping.

**Timeout**: If eval.py runs longer than 5 minutes, kill it and treat as crash.

**Crash handling**: Typo or import error → fix and re-run. Fundamentally broken idea → log crash, move on.

---

## Results Log Format

File: `results.tsv` (tab-separated)

```
commit	auc_roc	n_flagged	status	description
a1b2c3d	0.512000	1423	keep	baseline — payment z-score only
b2c3d4e	0.531000	1398	keep	added services_per_bene z-score
c3d4e5f	0.528000	1410	discard	switched to log transform — no improvement
```

Columns:
1. Git commit hash (7 chars)
2. AUC-ROC score (6 decimal places)
3. Number of providers flagged above 0.5 threshold (for reference)
4. `keep`, `discard`, or `crash`
5. One-sentence description of what changed

---

## What Success Looks Like

| AUC-ROC | Interpretation |
|---|---|
| 0.50 | Random — no signal |
| 0.60 | Weak signal — better than chance |
| 0.70 | Useful — worth building on |
| 0.75+ | Good — this strategy is working |
| 0.80+ | Strong — publishable result |

Baseline expectation for pure billing outliers: 0.62–0.70.  
If we can't crack 0.65 with statistical signals alone, that's also a finding worth reporting.

---

*Strategy A — Conservative | Authored by Chief | March 2026*  
*Competing branch: fraud-A | Compare against fraud-B (aggressive) and fraud-C (domain-first)*
