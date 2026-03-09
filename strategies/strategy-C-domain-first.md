# Strategy C — Domain-First (Fraud Typology Driven)

**Run tag convention:** `autoresearch/fraud-C-<date>` (e.g. `autoresearch/fraud-C-mar8`)  
**Branch from:** `main`  
**Competing against:** Strategy A (conservative), Strategy B (aggressive)

---

## Your Mission

You are an autonomous fraud detection researcher. Your goal is to maximize `auc_roc` — the AUC-ROC score measuring how well the model separates known fraudulent providers (LEIE-excluded) from the general Medicare provider population.

**This strategy: think like an investigator, not a statistician.** Don't start with data — start with how healthcare fraud actually works. Build targeted detectors for specific known fraud schemes, then combine them. Statistical outliers are a starting point; domain knowledge is the edge. This strategy draws on 30+ years of OIG enforcement patterns.

---

## Setup

1. **Agree on a run tag** with the user before starting. Use `autoresearch/fraud-C-<date>`.
2. **Create the branch**: `git checkout -b autoresearch/fraud-C-<date>` from main.
3. **Read all in-scope files**:
   - `README.md` — project context
   - `eval.py` — the fixed eval harness. Read it. Understand exactly what it measures. **Do not modify.**
   - `detector.py` — the file you edit.
   - This file (`strategy-C-domain-first.md`) — your research strategy.
4. **Establish the baseline**: Run `python eval.py > run.log 2>&1`. Record baseline AUC in `results.tsv`.
5. **Confirm with user and begin.**

---

## What You CAN Do

- Modify `detector.py` — the **only** file you edit.
- Build separate sub-scores for each fraud typology (pill mills, kickbacks, phantom billing, upcoding).
- Join any CMS tables needed. Domain-appropriate complexity is fine.
- Use domain knowledge encoded in the strategy below to design targeted signals.
- Combine sub-scores in any way (max, weighted sum, boolean flags).
- Use specialty-specific logic (what's normal for a cardiologist ≠ what's normal for a psychiatrist).

## What You CANNOT Do

- Modify `eval.py`. It is read-only. It defines ground truth.
- Add packages that require installation.
- Use generic ML model training (sklearn fit/predict). All scoring must be interpretable rule-based or SQL logic.

---

## The Four Fraud Typologies — Build These First

Do not start by finding statistical outliers. Start by asking: *what does this fraud scheme look like in claims data?* Build one typology at a time, test AUC after each, keep what works.

---

### Typology 1 — The Pill Mill 💊

**What it is:** Provider prescribing controlled substances (especially opioids) at volumes that suggest distribution, not treatment.

**How investigators find it:**
- Patients drive long distances to see this provider (not local doctor shopping)
- Cash patients predominate (avoid insurance oversight)
- High-dose, long-duration opioid prescriptions
- Little variation in prescription (everyone gets the same drug/dose)
- Multiple patients at same residential address

**What we can measure:**
- Opioid claim rate vs. specialty peers (CMS Part D opioid flag)
- Total opioid drug claims (raw volume, not rate-adjusted)
- Brand-to-generic ratio for opioids (brand preference = kickback or billing fraud)
- Drug portfolio concentration (few drugs, high volume = pill mill pattern)
- Prescribing volume per patient (outlier = excessive prescribing)
- Specialty appropriateness for opioid prescribing (primary care with high opioids = flag)

**Build the pill mill sub-score first. This is the highest-signal typology.**

---

### Typology 2 — The Kickback Ring 💰

**What it is:** Financial relationships between providers and suppliers/manufacturers that corrupt medical decision-making (illegal under Anti-Kickback Statute).

**Classic schemes:**
- Physician-Owned Distributorships (PODs): surgeons own equity in implant companies, then prescribe their own products
- Speaker programs: drug companies pay doctors to "speak" — really just meals and friendship
- DME suppliers paying doctors for referrals
- Lab companies paying for referral orders

**What we can measure (Open Payments):**
- Ownership interests received (highest risk — direct equity)
- Speaking/teaching payment amounts (>$10K/year is flagged by OIG)
- Total payments from pharmaceutical manufacturers (volume)
- Payments from medical device companies (especially implant manufacturers)
- Number of distinct companies paying this doctor (breadth of relationships)
- Recency of payments (within last 3 years)
- Payment type breakdown (consulting > speaking > food in risk hierarchy)

**Combine with billing data:** A doctor receiving $50K/year in speaking fees who ALSO bills Medicare at the 99th percentile for their specialty is a much stronger signal than either alone.

---

### Typology 3 — Phantom Billing / Upcoding 👻

**What it is:** Billing for services never rendered, or billing more complex/expensive services than actually provided.

**How it shows up in data:**
- Extremely high service volume (can't physically see this many patients)
- Billing at highest complexity level (99215) for nearly all visits
- Services billed on weekends/holidays at high volume
- Same service code repeated at maximum frequency
- Geographic impossibility (provider billing in two distant cities same day — data artifact, but worth flagging)
- High beneficiary-to-provider ratio (patients per day exceeds human limits)

**What we can measure:**
- Services per day of service (total claims ÷ distinct service dates)
- High-complexity code rate (% of E&M at 99214/99215)
- Average number of services per beneficiary vs. peers
- Beneficiary count per provider vs. specialty capacity norms
- Single HCPCS code concentration (one procedure = 80%+ of all claims)
- Consistency of billing (low variance = copy-paste fraud indicator)

---

### Typology 4 — The Quality-Billing Paradox 📉

**What it is:** Providers billing at maximum volume while providing substandard care — a pattern consistent with patient warehousing and churning (billing for minimal/poor care repeatedly).

**What we can measure:**
- MIPS composite score vs. billing volume (low quality + high volume = red flag)
- MIPS quality measure performance on relevant measures
- Gap between specialty-expected quality scores and actual
- Non-participation in MIPS when peers participate (avoidance signal)
- Beneficiary churn rate (patients seen only once at high frequency = churning)

This is the weakest typology for detecting LEIE exclusions (most exclusions are clear fraud, not quality failures), but worth testing.

---

## Score Architecture

Build progressively. After each typology is built, test it alone, then in combination:

```python
# Sub-scores (each 0-1)
pill_mill_score = compute_pill_mill(provider_npi)
kickback_score  = compute_kickback(provider_npi)
phantom_score   = compute_phantom_billing(provider_npi)
quality_score   = compute_quality_paradox(provider_npi)

# Combination strategies to try:
# 1. Simple average
fraud_score = mean([pill_mill, kickback, phantom, quality])

# 2. Max (any red flag = flagged)
fraud_score = max(pill_mill, kickback, phantom, quality)

# 3. Weighted by typology prevalence in LEIE data
# (After baseline: check which typology predicts LEIE better)
fraud_score = 0.40*pill_mill + 0.30*kickback + 0.25*phantom + 0.05*quality

# 4. Multiplicative amplification for co-occurring signals
fraud_score = pill_mill * (1 + 0.5*kickback) + phantom * (1 + 0.3*kickback)
```

Try each combination architecture as a separate experiment. Record which one wins.

---

## Specialty-Specific Logic

Generic z-scores miss the point: a psychiatrist prescribing any opioids is different from an internist prescribing many. Consider:

- **Pain management / anesthesiology**: high opioid volume may be appropriate → use strict specialty-specific thresholds
- **Primary care**: opioid prescribing above 95th percentile for PCPs = strong signal
- **Oncology**: high drug payments from manufacturers = potential kickback but also legitimate education
- **Surgery**: device company payments (implants) = higher risk than pharma payments
- **DME / home health**: geographic clustering = high signal (known fraud hot zone)
- **Psychiatry**: no opioid baseline → any opioid prescribing = anomaly

Build specialty-aware scoring where possible. A flag means more when it's specialty-inappropriate.

---

## The Experiment Loop

**LOOP FOREVER** (do not stop, do not ask permission to continue):

1. Check git state.
2. Choose one change — one typology component, one weight adjustment, one architecture variation.
3. State your hypothesis explicitly in the commit message. *"Hypothesis: pill mill sub-score will outperform generic billing outlier for LEIE match."*
4. `git commit`
5. Run: `python eval.py > run.log 2>&1`
6. Read: `grep "^auc_roc:" run.log`
7. Crash → check log, fix or discard.
8. Log to `results.tsv` with the typology name in the description.
9. Improved → keep. Not improved → `git reset HEAD~1`.
10. Repeat.

**NEVER STOP. NEVER ASK FOR PERMISSION.**  
If stuck, go deeper on a typology. Check which specific LEIE exclusion categories (EXCLTYPE field) are most common in the dataset — build targeted signals for the top categories. If the pill mill score isn't moving AUC, examine which LEIE-excluded providers it misses and why. Think like an OIG investigator.

**Timeout**: 5 minutes max per eval run. Kill and treat as crash if exceeded.

---

## Results Log Format

File: `results.tsv` (tab-separated)

```
commit	auc_roc	n_flagged	status	description
a1b2c3d	0.512000	1423	keep	baseline
b2c3d4e	0.558000	1654	keep	[pill-mill] opioid rate + brand preference sub-score
c3d4e5f	0.571000	1700	keep	[kickback] added ownership interests from Open Payments
d4e5f6g	0.574000	1712	keep	[phantom] services-per-day outlier added
e5f6g7h	0.579000	1723	keep	[combo] max() across typologies beats mean()
f6g7h8i	0.572000	1700	discard	[quality] MIPS paradox — no signal for LEIE
```

---

## OIG Enforcement Context

The LEIE exclusion categories most common in the dataset (EXCLTYPE codes):
- **1128(a)(1)**: Conviction for program-related crimes (most common)
- **1128(a)(3)**: Conviction for healthcare fraud
- **1128(b)(15)**: Owners/officers of sanctioned entities
- **1156**: Gross and flagrant violations (quality-adjacent)

Different exclusion types may respond to different signals. Consider splitting the LEIE labels by EXCLTYPE and testing which typology best predicts each category. That's a sophisticated experiment worth trying if you plateau.

---

## What Success Looks Like

| AUC-ROC | Interpretation |
|---|---|
| 0.50 | Random |
| 0.62-0.68 | Stats-only territory (Strategy A baseline) |
| 0.72-0.78 | Domain signals adding real lift |
| 0.80+ | Strong — domain knowledge pays off |
| 0.85+ | Exceptional finding |

The hypothesis: domain-first signals should outperform pure statistical outliers because we're encoding investigator knowledge that data can't surface on its own. If Strategy C doesn't beat Strategy A, that's a finding — pure statistical models are sufficient and domain knowledge adds noise.

---

*Strategy C — Domain-First | Authored by Chief | March 2026*  
*Competing branch: fraud-C | Compare against fraud-A (conservative) and fraud-B (aggressive)*  
*Reference: OIG Work Plan, DOJ healthcare fraud press releases 2018–2024, CMS Program Integrity Manual*
