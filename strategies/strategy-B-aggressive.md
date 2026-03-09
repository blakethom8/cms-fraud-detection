# Strategy B — Aggressive (Kitchen Sink)

**Run tag convention:** `autoresearch/fraud-B-<date>` (e.g. `autoresearch/fraud-B-mar8`)  
**Branch from:** `main`  
**Competing against:** Strategy A (conservative), Strategy C (domain-first)

---

## Your Mission

You are an autonomous fraud detection researcher. Your goal is to maximize `auc_roc` — the AUC-ROC score measuring how well the model separates known fraudulent providers (LEIE-excluded) from the general Medicare provider population.

**This strategy: find the ceiling.** Use everything available. Cross every data source. Build the most powerful composite score you can, regardless of complexity. Strategy A is proving what simple signals can do — your job is to find out what's actually possible when you throw everything at the problem. Don't be afraid of complexity. We can simplify later.

---

## Setup

1. **Agree on a run tag** with the user before starting. Use `autoresearch/fraud-B-<date>`.
2. **Create the branch**: `git checkout -b autoresearch/fraud-B-<date>` from main.
3. **Read all in-scope files**:
   - `README.md` — project context
   - `eval.py` — the fixed eval harness. Read it. Understand exactly what it measures. **Do not modify.**
   - `detector.py` — the file you edit. The current scoring logic.
   - This file (`strategy-B-aggressive.md`) — your research strategy.
4. **Establish the baseline**: Run `python eval.py > run.log 2>&1` on the unmodified `detector.py`. Record the baseline AUC in `results.tsv`.
5. **Confirm with user and begin.**

---

## What You CAN Do

- Modify `detector.py` — the **only** file you edit.
- Join ANY combination of CMS tables. Complex multi-table queries are fine.
- Build cross-source composite signals (billing + prescribing + payments together).
- Use Open Payments data to detect financial conflicts of interest.
- Use facility affiliations data to detect referral concentration.
- Add weighted ensemble scoring (different weights for different fraud types).
- Try non-linear scoring (percentile rank → sigmoid, log transforms, winsorizing).
- Add interaction terms between features (e.g., high billing AND high Open Payments).

## What You CANNOT Do

- Modify `eval.py`. It is read-only. It defines ground truth.
- Add Python packages that require pip install (use what's already imported).
- Train a machine learning model that takes >60 seconds on the full provider dataset. Keep it SQL/pandas scoring — no sklearn model.fit() on 8M rows.

---

## The Research Focus

**All data sources are available:**

| Source | Key Fraud Signals |
|---|---|
| Medicare Part B utilization | Billing volume outliers, service mix anomalies, high-complexity upcoding |
| Medicare Part D prescriber | Opioid pill mills, brand kickbacks, narrow formulary |
| Open Payments (General) | Speaking fees, consulting payments, ownership interests |
| Open Payments (Research) | Research funding conflicts |
| NPPES registry | Specialty mismatches, address anomalies |
| Facility affiliations | Referral concentration, exclusive relationships |
| MIPS Performance | Quality score outliers, low performers with high billing |

---

## Signal Strategy — Attack in Layers

### Layer 1: Base Billing Score (must have)
Same as Strategy A — billing outliers relative to specialty peers. This is table stakes.

### Layer 2: Cross-Source Amplification
The real power comes from signals that reinforce each other:
- Provider is a billing outlier **AND** receives high Open Payments → much stronger signal than either alone
- Provider is an opioid outlier **AND** has high brand-name preference → pill mill + kickback combo
- Provider bills high complexity **AND** has low MIPS quality score → upcoding proxy

Try multiplicative interactions, not just additive ones. A provider scoring high on 3 independent signals should score much higher than one scoring high on 1.

### Layer 3: Financial Conflict Signals (Open Payments)
- Total Open Payments received (any amount is a flag)
- Payment type breakdown (ownership interests > speaking fees > consulting fees in risk)
- Number of distinct payers (many companies = industry darling or kickback network)
- Payment recency (payments in last 2 years more relevant than older)
- Ownership interest specifically (highest risk payment type)

### Layer 4: Referral Concentration (Facility Affiliations)
- Number of distinct affiliated facilities (too few = exclusive relationship)
- Whether affiliated facilities are in same zip code (tight geographic clustering)
- Affiliation type distribution (hospital vs. clinic vs. DME supplier)

### Layer 5: Quality-Billing Divergence
- MIPS composite score vs. billing volume (low quality + high volume = red flag)
- Whether provider has MIPS data at all (no participation = avoidance signal)

### Layer 6: Temporal and Career Signals
- Years since first Medicare claim (new providers AND very old providers have elevated risk)
- Year-over-year billing growth rate (sudden spikes are fraud indicators)

---

## Composite Score Architecture to Explore

Start simple, then evolve:

**v1 — Additive:**
```
score = w1*billing_zscore + w2*opioid_flag + w3*open_payments_total
```

**v2 — Multiplicative amplification:**
```
score = billing_zscore * (1 + open_payments_weight) * (1 + opioid_weight)
```

**v3 — Percentile rank ensemble:**
```
score = mean(billing_pct, opioid_pct, payments_pct, quality_pct)
```

**v4 — Weighted by fraud type:**
```
# Pill mill component
pill_mill = 0.6*opioid_rate + 0.4*brand_preference
# Kickback component  
kickback = 0.7*open_payments + 0.3*billing_outlier
# Upcoding component
upcoding = 0.5*complexity_rate + 0.5*billing_outlier
# Final composite
score = max(pill_mill, kickback, upcoding)  # or weighted sum
```

Try all of these. Try variations. The goal is finding the architecture that maximizes AUC.

---

## Simplicity Criterion (Relaxed for This Strategy)

Unlike Strategy A, complexity is acceptable here IF it improves AUC.
- A 0.005 AUC gain from 50 lines of complex SQL? Worth it — that's real signal.
- A 0.001 AUC gain from 50 lines? Probably not.
- If you're not finding gains with complexity, try simplifying — sometimes less is more.

The goal is to find the **maximum achievable AUC**. We can always simplify a winning approach later.

---

## The Experiment Loop

**LOOP FOREVER** (do not stop, do not ask permission to continue):

1. Check git state — current branch and last commit.
2. Choose one change to `detector.py`. Document your hypothesis in the commit message.
3. `git commit` the change.
4. Run: `python eval.py > run.log 2>&1`
5. Read result: `grep "^auc_roc:" run.log`
6. If grep empty → crash. Check `tail -n 30 run.log`, fix or move on.
7. Log to `results.tsv`: `commit | auc_roc | n_flagged | keep/discard | description`
8. If AUC improved → keep commit.
9. If not → `git reset HEAD~1`, discard.
10. Repeat.

**NEVER STOP. NEVER ASK FOR PERMISSION.**  
If stuck, try a completely different approach — change score architecture, add a new data source, try removing features that may be adding noise. If you've exhausted all obvious ideas, look at which providers are near the LEIE boundary (score ~0.5) and ask what features would separate them. Think like an investigator.

**Timeout**: Kill and treat as crash if eval.py exceeds 5 minutes.

---

## Results Log Format

File: `results.tsv` (tab-separated)

```
commit	auc_roc	n_flagged	status	description
a1b2c3d	0.512000	1423	keep	baseline
b2c3d4e	0.554000	1876	keep	added open payments total + billing interaction
c3d4e5f	0.561000	1901	keep	added opioid x brand multiplicative term
d4e5f6g	0.558000	1890	discard	facility affiliation count — no signal
```

---

## What Success Looks Like

| AUC-ROC | Interpretation |
|---|---|
| 0.50 | Random |
| 0.65 | Strategy A territory — stats only |
| 0.75 | Cross-source signals working |
| 0.80+ | Strong composite model — publishable |
| 0.85+ | Exceptional — this would be a real finding |

Target for this strategy: break 0.75+. If we can't beat Strategy A by at least 0.05 AUC with all this extra data, that's also a finding (simpler models may be sufficient).

---

*Strategy B — Aggressive | Authored by Chief | March 2026*  
*Competing branch: fraud-B | Compare against fraud-A (conservative) and fraud-C (domain-first)*
