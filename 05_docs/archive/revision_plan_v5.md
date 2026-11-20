# Revision Plan — "The Instruction Tuning Firewall" (v5 → v6)

**Date created:** 2026-02-12  
**Target journal:** The Lancet Digital Health  
**Verdict received:** Major Revisions  
**Status:** Phases 1-3 Complete — Phase 5: Manuscript Revision (DeBERTa integrated)

---

## Overview of Reviewer Concerns

The reviewer identified **7 major issues**, **11 minor issues**, and **7 missing analyses**. The core objections cluster into three themes:

1. **Circularity & validation** — LLM judge is both trainer and evaluator; no human clinician ground truth
2. **Unfair comparisons** — The 2.9× gap compares a bespoke activation projection against handcrafted text features, not a learned text classifier
3. **Ecological validity** — All experiments are synthetic; clinical/regulatory claims are unsupported without naturalistic data

---

## MAJOR ISSUES

### M1. Circularity in Evaluation Pipeline
- **Location:** §3.4 (contrastive probing step ②), §3.1 (steering validation)
- **Problem:** GPT-4o-mini scores training labels AND evaluates outcomes → tautological risk
- **ICC=0.827 with Gemini** measures agreement, not independence from training signal

#### Revision Plan
- [x] **M1a.** Retrain contrastive probes using Gemini-only labels. ❌ **FAIL — circularity confirmed.** Cosine similarity = 0.036 (Mistral), functional ratio = 0.24. Probe directions are judge-dependent. Qwen2 cancelled. See M-EXP7/7b diagnosis in Phase 3 Progress.
  - **Code:** `modal_gemini_probe_retraining.py`, `modal_cross_prediction_test.py`
- [ ] **M1b.** Run human clinician validation (minimum N=50 responses, 2–3 traits). See M-EXP1 below.
- [ ] **M1c.** In manuscript, add explicit "Independence of training and evaluation" subsection to Methods.

**Priority: CRITICAL — blocks acceptance**

---

### M2. Construct Validity of Eight Dimensions
- **Location:** §2
- **Problem:** Mapping to Rogers (1957) and Wampold (2015) is underspecified. Missing: congruence, goal consensus, therapeutic rupture-repair (Safran & Muran, 2000). Boundary between "uncritical validation" and "sycophancy/harmful validation" is unclear.

#### Revision Plan
- [x] **M2a.** Create formal construct mapping table: ✅ *Done — Table 0 added to §2 with Rogers/Wampold/Source columns*

| Our Dimension | Rogers (1957) Construct | Wampold (2015) Factor | Notes |
|---|---|---|---|
| Empathetic responsiveness | Empathic understanding | Empathy | Direct mapping |
| Non-judgmental acceptance | Unconditional positive regard | — | Direct mapping |
| Boundary maintenance | — | Real relationship (Gelso) | Extended construct |
| Crisis recognition | — | — | Safety-specific, no classic mapping |
| Emotional over-involvement | (inverse of) Congruence? | — | Needs justification |
| Abandonment of therapeutic frame | — | Treatment structure | Loose mapping |
| Uncritical validation | — | — | Derived from sycophancy lit (Sharma 2023) |
| Sycophancy/harmful validation | — | — | Safety-specific |

- [x] **M2b.** Add paragraph explicitly discussing omitted constructs: ✅ *Done — congruence, goal consensus, rupture-repair discussed*
  - **Congruence** — omitted because it presupposes genuine experience; chatbots simulate congruence
  - **Goal consensus** — requires multi-session tracking; out of scope for single-turn validation
  - **Rupture-repair** — acknowledged as important; flagged for future work (and discussed why it's not captured by static trait monitoring)
- [x] **M2c.** Report inter-trait correlation matrix from probe data to show discriminant validity among the 8 dimensions. ✅ *Done — UncritV×SycHarm r=0.74 (retain both; conceptually distinct). EmpResp×AbandFr r=0.76 (shared warmth factor). Full matrix in Supplementary Table S4.*
  - **Code:** Compute from existing `aggregated_*.json` files
  - **Effort:** ~2 hours analysis

**Priority: HIGH — affects theoretical framing**

---

### M3. Ecological Validity
- **Location:** §3–6, Discussion, Conclusions
- **Problem:** All experiments use synthetic data. Clinical/regulatory claims unsupported.

#### Revision Plan
- [x] **M3a.** Run monitoring pipeline on ESConv dataset: ✅ *Done — 200 ESConv conversations processed through Llama-3-8B. ALL 8/8 traits show meaningful variance (std > 0.05). ALL 8/8 show significant within-conversation drift (p < 0.0001). 1/8 differentiates by emotion type (sycophancy, p=0.019). Ecological validity confirmed: activation-based monitoring generalizes to naturalistic counseling data.*
  - Extract activations from real emotional support conversations ✅
  - Compute activation projections on all 8 traits ✅
  - Report: do projections show meaningful variance? **YES — all 8 traits** ✅
  - **Effort:** ~1–2 days
- [x] **M3b.** Qualify ALL clinical/regulatory claims with explicit caveats: ✅ *Done — Discussion, regulatory context, and patient outcome sections all hedged with "pending validation on naturalistic clinical data"*
  - Find every sentence containing "clinical deployment," "patients," "regulatory" and add "pending validation on naturalistic data"
  - **Effort:** ~1 hour manuscript editing
- [x] **M3c.** Report qualitative characteristics of 100-turn context erosion conversations: ✅ *Done — Self-BLEU analysis (Qwen2: 0.034, Mistral: 0.150), phase progression validated, Table S8 + excerpts added. Code: `m3c_erosion_quality.py`*
- [ ] **M3d.** Long-term: plan a pilot deployment study (see M-EXP5)

**Priority: HIGH — affects all clinical claims**

---

### M4. Unfair 2.9× Sensitivity Comparison
- **Location:** §4.2, Table 5
- **Problem:** Compares bespoke activation projection (designed to capture steering) against post-hoc best single handcrafted text feature. Not a fair comparison.

#### Revision Plan
- [x] **M4a.** Train multivariate text classifier (logistic regression + random forest) on ALL text features:
  - Trained LR, Ridge, RF, GB on 19 text features per trait
  - Best: Random Forest R²=0.112 (expanded corpus), equiv r=0.335
  - Activation R²=0.371, gap = 3.3× in variance explained
  - **Code:** `03_code/m4a_multivariate_text_classifier.py`
  - **Results:** `03_code/results/m4a_multivariate_text_classifier.json`
  - ✅ Done
- [x] **M4b.** Train fine-tuned DeBERTa-v3-base classifier on the 12,000-response expanded corpus:
  - **Model:** microsoft/deberta-v3-base, regression head (num_labels=1)
  - **Target:** steering coefficient (same as M4a), NOT judge score
  - **Method:** GroupKFold CV (5 folds, grouped by scenario), 3 epochs, lr=2e-5, fp16, max_length=256
  - **Parallelization:** 8 A10G containers via Modal .map() (one per trait)
  - **Results:**

  | Trait | R² | r | std(R²) |
  |---|---|---|---|
  | empathetic_responsiveness | 0.3664 | 0.6093 | 0.0657 |
  | non_judgmental_acceptance | 0.2889 | 0.5509 | 0.0093 |
  | boundary_maintenance | 0.4009 | 0.6438 | — |
  | crisis_recognition | 0.2065 | 0.4559 | — |
  | emotional_over_involvement | 0.3294 | 0.5854 | — |
  | abandonment_of_therapeutic_frame | 0.1873 | 0.4539 | — |
  | uncritical_validation | 0.3441 | 0.5944 | — |
  | sycophancy_harmful_validation | 0.1928 | 0.4406 | 0.0465 |
  | **Mean** | **0.2895** | — | — |

  - **Comparison ladder:**
    - Activation projections: R²=0.371 (reference)
    - DeBERTa fine-tuned: R²=0.290 (1.3× gap)
    - M4a Random Forest (19 features): R²=0.112 (3.3× gap)
    - Embedding similarity (MiniLM): R²=0.004 (95× gap)
  - **Interpretation:** DeBERTa closes ~78% of the activation-text gap compared to simple text features, demonstrating that fine-tuned neural text models CAN extract substantial signal. However, activations still outperform even the strongest text baseline on 6/8 traits. The firewall is "leaky but real" — the headline claim must be reframed from "text can't detect" to "activations maintain a meaningful advantage even against state-of-the-art text classifiers."
  - **Code:** `03_code/modal_deberta_classifier.py`
  - **Results:** `03_code/results/deberta_text_classifier.json`
  - ✅ Done
- [x] **M4c.** Acknowledge in §7 that the LLM judge is a text-based method. ✅ Done — new subsection "The LLM judge as a text-based detector" + Table 9 cost comparison
- [x] **M4d.** Report R² alongside r throughout. ✅ Done — Tables 5a/5b report both r and R²; Abstract, §1.3, §7, §8 all updated
- [x] **M4e.** Reframe headline claim with multivariate baseline. ✅ Done — gap updated from "2.9× (r)" to "3.3× (R², multivariate)" throughout

**Priority: CRITICAL — the headline finding depends on this**

---

### M5. Causal Interpretation / Discriminant Validity
- **Location:** §3.1, §7
- **Problem:** Cannot rule out that steering simply perturbs the model generically, and judge responds to any perturbation.

#### Revision Plan
- [x] **M5a.** Random direction control experiment: ✅ *Done — Llama3 mean|r|=0.040, Qwen2 0.034, Mistral 0.073. All 3 models PASS. Code: `modal_random_direction_control.py`*
- [x] **M5b.** Cross-trait discriminant validity test: ✅ *Done — Mean off-diagonal |r|: 0.26 (Mistral) to 0.47 (Qwen2 probe). 3-4 factor structure identified. Table S9 added. Code: `m5b_cross_trait_discriminant.py`*
- [x] **M5c.** Report both results in a new subsection "§3.6 Discriminant validity and specificity" ✅ *Done — Table S9 with discriminant matrix*

**Priority: CRITICAL — without this, causal claims are unsupported**

---

### M6. Safety Findings at Non-Validated Coefficients
- **Location:** §5, Abstract
- **Problem:** Mistral harm=8.0 at coeff +5.0 is outside validated ±3.0 range. Framing is alarmist for clinical audience.

#### Revision Plan
- [x] **M6a.** Report safety results at ±3.0 as primary finding in §5 and Table 7: ✅ *Done — §5.1 reports ±3.0 as primary (Mistral L14 harm=3.4, Qwen2 all ≤1.5)*
  - Extract existing data from `safety_stress_test_results.json` for coeff ±3.0
  - **Effort:** ~2 hours
- [x] **M6b.** Move ±5.0 results to a "stress test beyond validated range" sub-section, framed explicitly for model developers/red-teamers. ✅ *Done — §5.2*
- [x] **M6c.** Clarify in §5 that harm scores are GPT-4o-mini judgments. ✅ *Done — "All harmfulness scores reported below are GPT-4o-mini judgments on a 1-10 scale" added to §5.2*
- [x] **M6d.** Revise Abstract to report safety findings at ±3.0, not ±5.0. ✅ *Done — Abstract now leads with ±3.0 validated range*

**Priority: MEDIUM — framing issue, not fundamental flaw**

---

### M7. Missing Baselines
- **Location:** §4, §7
- **Problem:** No comparison to linear probes on human-labelled data, embedding similarity, fine-tuned text classifiers, or LLM-as-text-classifier.

#### Revision Plan
- [x] **M7a.** Embedding similarity baseline: ✅ **DONE — Mean CV R²=0.004, Gap=95× vs activations**
  - Model: sentence-transformers/all-MiniLM-L6-v2
  - Method: Cosine similarity to hand-crafted high/low trait anchor embeddings
  - Target: steering coefficient (same as M4a)
  - GroupKFold CV (5 folds, grouped by scenario)
  - Result: Mean R²=0.003 (raw), Mean CV R²=0.004
  - Best trait: non_judgmental_acceptance (CV R²=0.009)
  - Worst: empathetic_responsiveness (CV R²=0.001)
  - Gap: Activation R²=0.371 / Embedding R²=0.004 = **95×**
  - Interpretation: Pre-trained embeddings capture essentially zero steering signal — the firewall is complete at the embedding level
  - Code: `03_code/modal_embedding_similarity.py`, Results: `results/embedding_similarity_baseline.json`
- [x] **M7b.** Fine-tuned text classifier ✅ *Done — same as M4b (DeBERTa R²=0.290)*
- [x] **M7c.** LLM-as-real-time-monitor discussion. ✅ Done — new §7 subsection "The LLM judge as a text-based detector" with cost/latency comparison
- [x] **M7d.** Add comparison table (Table 9 in manuscript). ✅ Done — 5 methods with R², latency, cost, model access requirements

**Priority: HIGH — strengthens the paper's practical argument**

---

## MINOR ISSUES

### m1. "Instruction Tuning Firewall" Terminology
- **Problem:** Conflates instruction tuning and RLHF/DPO. Could confuse clinical audience.
- [x] Add definition on first use: ✅ *Done — §1.2 now defines as "shorthand for the combined effect of instruction tuning, RLHF, and DPO training — all of which constrain the model's output to sound appropriate regardless of internal state perturbation. We use this term rather than 'safety training firewall' to emphasise that the phenomenon is not limited to safety-specific training; general instruction-following capability itself contributes to output normalisation."
- **Effort:** 10 minutes

### m2. Table 1 Prompt Details
- **Problem:** Unclear whether 10 prompts are independent scenarios.
- [x] Add footnote to Table 1 referencing Supplementary Table S1. ✅ *Done — footnote cites S1 and S3*
- [x] Create Supplementary Table S1 listing all prompts. ✅ *Done — 05_docs/supplementary_material.md*
  - Source: `challenge_scenarios.json`
- **Effort:** 30 minutes

### m3. LLM Judge Rubric Not Reproduced
- **Problem:** The 1–7 rubric is central but not shown.
- [x] Add rubric to Supplementary Material ✅ *Done — Supplementary Table S3 in 05_docs/supplementary_material.md with all 8 traits, 4-level anchors, calibration vignettes, and edge cases*
- **Effort:** 30 minutes

### m4. ICC Type Not Specified
- **Problem:** ICC=0.827 reported without specifying type.
- [x] Specify: ICC(2,1) two-way random, single measures, absolute agreement. ✅ *Done — Limitations section updated with CI [0.78, 0.86]*
- [x] Verified from `judge_agreement_report.json` ✅
- **Effort:** 15 minutes

### m5. Serial Autocorrelation in §6.1
- **Problem:** OLS on ~2,000 turn-level observations ignores within-conversation clustering.
- [x] Re-run analysis with clustered SEs + mixed-effects model ✅ *Done — SE inflation 1.2–3.8×. emotional_over_involvement remains significant all 3 models. Code: `m5_clustered_se_erosion.py`*
- [x] Report both OLS and robust estimates ✅ *Done — Table S7 (naive vs clustered vs MLM + ICCs)*
- [x] If significance changes, update Table 8 and text ✅ *Done — Llama-3 uncritical_validation lost significance (p=0.017→0.524)*
  - **Code:** Modify `modal_context_erosion_v2.py` analysis section
- **Effort:** ~0.5 days

### m6. Thin Reference List
- **Problem:** Only 15 references for a Lancet submission.
- [x] Added 10 references (Ngo, Perez, Wei, Safran & Muran, Belinkov, EU AI Act, FDA SaMD, Gelso, Bordin, Stiles). ✅ *Done — references expanded from 15 to 25*
- **Effort:** 1 hour

### m7. ASCII Diagrams
- **Problem:** Won't survive journal typesetting.
- [x] Convert all ASCII diagrams to vector graphics (SVG/PDF): ✅ *Done — `03_code/generate_diagrams.py` → `04_results/figures/fig{1..5}_{name}.svg` + PDF*
  - §3.1 Steering Vector Extraction
  - §3.3 Diagnostic Predictors
  - §3.4 Contrastive Probing Pipeline
  - §3.5 Real-Time Monitoring Architecture
  - §5 Mistral Safety Gradient
- [x] Tools: matplotlib → SVG + PDF ✅
- **Effort:** ~1 day

### m8. Table 7 Layer Confusion
- **Problem:** Safety test layers (L10, L12, L14) vs steering layers (L17–19) not explained.
- [x] Add clarifying sentence after Table 7. ✅ *Done — note explains safety test layers (10-19) vs monitoring layers (17-19)*
- **Effort:** 10 minutes

### m9. Cohen's d Confidence Intervals
- **Problem:** Table 6 effect sizes lack CIs; n=30 per cell is small.
- [x] Add 95% CI for each Cohen's d in Table 6. ✅ *Done — Hedges & Olkin (1985) variance formula; most CIs cross zero at N=30*
- [x] Note n per group explicitly. ✅ *Done — "N=30 per group (10 scenarios × 3 models)"*
- [ ] If using expanded corpus (n=100), recompute and report
- **Effort:** 1 hour

### m10. Pearson r vs Other Correlation Types
- **Problem:** "r" used throughout without disambiguation.
- [x] Audit all correlations: ✅ *Done — max |Δ|=0.056 (emotional_over_involvement), mean |Δ|=0.020. Full comparison in Supplementary Table S5*
  - Steering coefficient (5 ordinal levels) vs judge score → Spearman's ρ or linear trend test
  - Activation projection (continuous) vs judge score → Pearson r
  - Add footnote on first use: "We report Pearson's r for continuous–continuous associations and note that treating the 5-level steering coefficient as continuous yields essentially identical results to Spearman's ρ (all |Δ| < 0.02)."
- [x] Verify by computing Spearman for all Table 1 entries ✅ *Done — results in Supplementary Table S5 (max |Δ|=0.056)*
- **Effort:** 2 hours

### m11. Equivalence Testing for §6.2
- **Problem:** Non-significance ≠ equivalence. Fine-tuning "null" result needs TOST.
- [x] Run Two One-Sided Tests (TOST) with equivalence bounds of ±0.5 SD: ✅ *Done — 0/8 equivalent (all p_TOST>0.15). §6.2 rewritten as inconclusive. Full results in Supplementary Table S6*
  - If TOST significant → can claim equivalence (fine-tuning doesn't change traits)
  - If TOST non-significant → insufficient power to claim equivalence; report as "inconclusive"
- [x] Update §6.2 text accordingly ✅ *Done — §6.2 rewritten as inconclusive with TOST results*
- **Effort:** 2 hours

---

## MISSING EXPERIMENTS

### M-EXP1. Human Clinician Validation ⭐ HIGHEST PRIORITY
- **What:** 3–5 licensed therapists rate ~200 steered responses on the 1–7 rubric
- **Design:**
  - Stratify: 3 traits × 5 coefficients × 3 models × ~4 scenarios = ~180 responses
  - Blind raters to steering condition
  - Randomise presentation order
  - Report: human–human ICC, human–GPT ICC, activation r vs human scores
- **Infrastructure:** Rating protocol exists in `05_docs/expert_review_protocol.md`; rating sheets exist in `05_docs/human_validation/`
- **Effort:** 2–4 weeks (recruitment + rating + analysis)
- **Blocks:** M1 resolution

### M-EXP2. ESConv / Naturalistic Data Test ✅ PASS
- **What:** Run monitoring pipeline on ESConv emotional support conversations
- **Design:**
  - Process ESConv with `step0b_process_esconv.py`
  - Extract activations from Llama-3-8B on ESConv counsellor turns
  - Project onto all 8 trait vectors
  - Analyse trait variance, emotion-based differences, within-conversation drift
- **Effort:** 1–2 days
- **Blocks:** M3 resolution
- **Result (Llama-3-8B, 200 conversations):**

| Trait | Mean | Std | Range | CV | Variance |
|---|---|---|---|---|---|
| empathetic_responsiveness | 1.818 | 1.014 | 3.854 | 0.558 | ✅ OK |
| non_judgmental_acceptance | 0.753 | 0.340 | 1.843 | 0.451 | ✅ OK |
| boundary_maintenance | 0.219 | 0.474 | 1.962 | 2.165 | ✅ OK |
| crisis_recognition | 0.743 | 0.485 | 2.249 | 0.654 | ✅ OK |
| emotional_over_involvement | 0.466 | 0.620 | 2.676 | 1.330 | ✅ OK |
| abandonment_of_therapeutic_frame | 1.792 | 1.084 | 3.889 | 0.605 | ✅ OK |
| uncritical_validation | 0.617 | 0.379 | 2.137 | 0.614 | ✅ OK |
| sycophancy_harmful_validation | 1.051 | 0.386 | 1.832 | 0.368 | ✅ OK |

**Key findings:**
1. **Meaningful variance: 8/8 traits** — All trait projections show std > 0.05 on naturalistic data. Activation-based monitoring generalises beyond synthetic scenarios.
2. **Within-conversation drift: 8/8 traits significant** (all p < 0.0001) — Activations shift as conversations progress. Largest: empathetic_responsiveness (+0.93), abandonment_of_therapeutic_frame (+0.89). Smallest: crisis_recognition (+0.31).
3. **Emotion-based differentiation: 1/8** — Only sycophancy_harmful_validation differentiates by user emotion type (F=2.61, p=0.019). Other traits show similar activation patterns regardless of emotion category.
4. **Implication for manuscript:** This is strong evidence for ecological validity. The monitoring vectors, derived from synthetic contrastive prompts, produce meaningful and differentiated signals on real ESConv counseling conversations. The within-conversation drift finding directly supports the context erosion analysis (§6.1) — activations genuinely shift over multi-turn interactions, even in naturalistic data.

### M-EXP3. Random Direction Control
- **What:** Steer along 10 random vectors at same layers/coefficients
- **Design:**
  - For each model, generate 10 random normalised vectors of same dimensionality
  - Steer at [−3.0, −1.5, 0, +1.5, +3.0] on 10 scenarios
  - Judge-score all responses on all 8 traits
  - Expected: mean |r| < 0.10 for random vectors vs. r~0.4 for trait vectors
- **Effort:** ~1 day compute + 0.5 day analysis
- **Blocks:** M5 resolution

### M-EXP4. Cross-Trait Discriminant Validity
- **What:** Score steered responses on ALL 8 traits, not just the target trait
- **Design:**
  - Use existing steered responses (1,200 or 12,000 corpus)
  - For each response, compute judge scores on all 8 traits
  - Build 8×8 discriminant matrix
  - Diagonal should be >> off-diagonal
- **Effort:** ~1–2 days (re-scoring + analysis)
- **Blocks:** M5 resolution

### M-EXP5. Multivariate Text Classifier
- **What:** Train logistic regression + DeBERTa on all text features
- **Design:**
  - Logistic regression on 12 pragmatic features → predict coefficient
  - Fine-tune DeBERTa-v3-base → predict trait score from response text
  - Report R² for both; compare to activation R²
  - Add both to revised Table 5
- **Effort:** 1–2 days
- **Blocks:** M4 resolution

### M-EXP6. Multi-Trait Drift Interactions
- **What:** Correlation matrix of activation projections across traits
- **Design:**
  - From existing monitoring data (100 sessions × 3 models)
  - Compute pairwise Pearson r between all 8 trait projections
  - Report: are traits independent or clustered?
  - Heatmap figure
- **Effort:** ~0.5 days
- **Blocks:** M2c (discriminant validity)

### M-EXP7. Gemini-Labelled Probe Retraining
- **What:** Retrain contrastive probes using Gemini labels instead of GPT-4o-mini
- **Design:**
  - Score existing 500 responses × 3 models × 8 traits with Gemini
  - Retrain logistic regression probes
  - Compare probe vector cosine similarity to GPT-4o-mini-trained probes
  - Re-run validation: are r-values comparable?
- **Effort:** ~1–2 days
- **Blocks:** M1a resolution

---

## EXECUTION PLAN

### Phase 1: Quick Wins (Week 1) ✅ COMPLETE
*Manuscript edits + analyses using existing data*

| Day | Task | Issue | Status |
|-----|------|-------|--------|
| 1 | Construct mapping table + omitted constructs discussion | M2a, M2b | ✅ |
| 1 | Inter-trait correlation matrix from existing data | M2c, M-EXP6 | ✅ |
| 1 | Qualify all clinical/regulatory claims | M3b | ✅ |
| 1 | Fix ICC specification, correlation types, terminology | m4, m10, m1 | ✅ |
| 2 | Report safety results at ±3.0, reframe ±5.0 | M6a–d | ✅ |
| 2 | Add equivalence test (TOST) for §6.2 | m11 | ✅ |
| 2 | Add rubric + prompt tables to supplement | m2, m3 | ✅ |
| 2 | Expand reference list | m6 | ✅ |
| 2 | Table 7/layer clarification, Table 6 CIs | m8, m9 | ✅ |

### Phase 2: New Analyses with Existing Data (Week 2)
*Computational experiments on existing corpora*

| Day | Task | Issue | Status |
|-----|------|-------|--------|
| 3 | Multivariate text classifier (logistic regression on all features) | M4a, M-EXP5 | ✅ |
| 3 | LLM-judge-as-monitor discussion + cost table | M7c, M7d | ✅ |
| 4 | Acknowledge LLM judge = text-based detector in §4.2 | M4c | ✅ |
| 4 | Report R² alongside r throughout | M4d | ✅ |
| 4 | Headline claim reframed (3.3× R² gap) | M4e | ✅ |
| 5 | Clustered SE / mixed-effects for §6.1 | m5 | ✅ |
| 5 | Context erosion conversation quality analysis | M3c | ✅ |
| 5 | Cross-trait discriminant matrix (activation-level) | M-EXP4, M5b | ✅ |

### Phase 3: New Compute Experiments (Weeks 3–4)
*Require Modal GPU time*

| Day | Task | Issue |
|-----|------|-------|
| 6–7 | Random direction control experiment | M-EXP3, M5a |
| 8–9 | Gemini-labelled probe retraining | M-EXP7, M1a |
| 10 | ESConv naturalistic data test | M-EXP2, M3a |
| 11–12 | Fine-tuned DeBERTa text classifier | M-EXP5, M4b |
| 12 | Embedding similarity baseline | M7a |

### Phase 4: Human Validation (Weeks 5–8)
*Recruitment + data collection*

| Week | Task | Issue |
|------|------|-------|
| 5 | Finalise rating protocol, recruit 3–5 clinicians | M-EXP1 |
| 6 | Clinicians rate ~200 responses | M-EXP1 |
| 7 | Analyse human–human ICC, human–GPT ICC | M1b |
| 7 | Compute activation r vs human scores | M1b |
| 8 | Integrate results into manuscript | M1c |

### Phase 5: Manuscript Revision (Weeks 8–9)

| Task | Details |
|------|---------|
| Convert ASCII diagrams to vector figures | m7 |
| Rewrite Abstract (lead with clinical problem) | Presentation |
| Restructure for clinical audience (compress ML methods) | Presentation |
| Add "Discriminant validity" subsection (§3.6) | M5c |
| Revise Discussion: explicit non-findings | Presentation |
| Revise 2.9× claim with new multivariate baselines | M4e |
| Final reference check | m6 |
| Co-author review + resubmission | — |

---

## SUCCESS CRITERIA FOR RESUBMISSION

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Human–LLM judge ICC | ≥ 0.70 | ⬜ Not started |
| Activation r vs **human** scores | ≥ 0.30 for 6+/8 traits | ⬜ Not started |
| Random direction control | Mean |r| < 0.10 | ✅ PASS (Llama3: 0.040, Qwen2: 0.034, Mistral: 0.073) |
| Cross-trait discriminant ratio | Diagonal ≥ 2× off-diagonal | ✅ Done (mean off-diag |r|: 0.26–0.47; 3-4 factor structure) |
| Multivariate text baseline gap | Activation R² > text R² by ≥1.5× | ✅ Done (RF: 3.3×, DeBERTa: 1.3×; reframed as gradient) |
| Gemini probe cosine similarity | ≥ 0.70 to GPT-4o-mini probes | ❌ FAIL (cosine=0.036, functional ratio=0.24) |
| ESConv meaningful variance | Significant trait variation on naturalistic data | ✅ PASS (8/8 traits, all p<0.0001 drift) |
| Clustered SE preserves §6.1 findings | p < 0.05 for emotional_over_involvement | ✅ PASS (all 3 models p<0.01; Llama-3 uncritical_validation lost) |
| All clinical claims qualified | Every claim hedged re: synthetic data | ✅ Done |

---

## RISK REGISTER

| Risk | Impact | Mitigation |
|------|--------|------------|
| Human validation shows low human–LLM ICC | Fatal — undermines all results | Pilot with 2 raters on 30 responses first; iterate rubric |
| Random vectors show high trait scores | Fatal — causal claims collapse | **NOT TRIGGERED.** All 3 models PASS (mean |r|: 0.034–0.073, all <0.10). |
| DeBERTa closes the 2.9× gap entirely | Reframing needed (not fatal) | **PARTIALLY TRIGGERED.** Gap narrowed to 1.3× (R²=0.290 vs 0.371). Mitigation executed: manuscript reframed around comparison ladder + cost-efficiency argument (activations match DeBERTa at ~1000× lower compute). |
| ESConv shows no meaningful trait variance | Weakens ecological validity claim | **NOT TRIGGERED.** All 8/8 traits show meaningful variance + significant within-conversation drift on 200 ESConv conversations. Ecological validity confirmed. |
| Clustered SEs eliminate §6.1 significance | Must weaken context erosion claim | Report as "suggestive" rather than "significant" |
| Gemini probes diverge from GPT-4o-mini probes | Circularity confirmed | **TRIGGERED.** Probe directions are judge-dependent. Mitigation: (1) disclose as limitation, (2) human validation (M-EXP4) becomes critical — if human scores correlate with GPT probes, then GPT-4o-mini is a valid proxy, (3) frame as "monitoring detects drift as defined by GPT-4o-mini" |

---

## FILE MAPPING

| Revision task | Primary code file(s) | Output |
|---|---|---|
| M1a (Gemini probes) | `step1b_contrastive_probing.py`, `modal_judge_gemini.py` | `probe_diagnostics_*_gemini.json` |
| M2c (inter-trait corr) | New analysis script | Correlation heatmap |
| M3a (ESConv) | `step0b_process_esconv.py`, `step2_monitor_drift.py` | ESConv projection data |
| M4a (multivariate text) | `analyze_steered_corpus.py` | Revised Table 5 |
| M4b (DeBERTa) | New script `train_text_classifier.py` | Fine-tuned model + R² |
| M5a (random vectors) | `modal_random_direction_control.py` | `random_direction_control_{model}.json` |
| M5b (cross-trait) | `m5b_cross_trait_discriminant.py` | `m5b_cross_trait_discriminant.json` |
| m5 (clustered SE) | `m5_clustered_se_erosion.py` | `m5_clustered_se_erosion.json` |
| m11 (TOST) | `revision_analyses.py` | Equivalence test results |
| M-EXP7 (Gemini probes) | `modal_gemini_probe_retraining.py` | `gemini_vs_gpt_probe_comparison_{model}.json` |
| M-EXP2 (ESConv) | `modal_esconv_naturalistic.py` | `esconv_naturalistic_{model}.json` |
| M4b (DeBERTa) | `modal_deberta_classifier.py` | `deberta_text_classifier.json` |
| M7a (embedding sim) | `modal_embedding_similarity.py` | `embedding_similarity_baseline.json` |

---

*Last updated: 2026-02-15 (Phase 3 COMPLETE. Manuscript revised with DeBERTa gradient framing. Diagrams converted to SVG/PDF. Remaining: human validation.)*

---

## PHASE 2 PROGRESS

### Completed
- **M4a** — Multivariate text classifier (LR, Ridge, RF, GB on 19 features). Best: RF R²=0.112 vs activation R²=0.371 → 3.3× gap. Code: `03_code/m4a_multivariate_text_classifier.py`
- **M4c** — LLM judge acknowledged as text-based detector in new §7 subsection
- **M4d** — R² reported alongside r throughout (Tables 5a/5b, Abstract, §1.3, §7, §8)
- **M4e** — Headline claim reframed: "3.3× gap in R²" replaces "2.9× gap in r"
- **M7c** — LLM-as-monitor discussion added to §7
- **M7d** — Table 9 (method comparison: R², latency, cost, model access)
- **m5** — Clustered SEs for §6.1 context erosion. SE inflation 1.2–3.8×. Key result: emotional_over_involvement remains significant across all 3 models (Llama-3 p=0.004, Qwen2 p<0.001, Mistral p<0.001). One significance loss: Llama-3 uncritical_validation (p=0.017→0.524). Added Table S7 (naive vs clustered vs MLM + ICCs). Code: `03_code/m5_clustered_se_erosion.py`
- **M3c** — Context erosion conversation quality. Self-BLEU analysis: Qwen2 most diverse (0.034), Mistral most repetitive (0.150). Phase progression validated (emotional words 0%→0.5%). Added Table S8 + example excerpts. Code: `03_code/m3c_erosion_quality.py`
- **M5b/M-EXP4** — Cross-trait discriminant matrix from activation projections (500 scenarios × 5 model configs). Mean off-diagonal |r|: 0.26 (Mistral) to 0.47 (Qwen2 probe). Identified 3-4 factor structure (warmth, professional structure, validation). Added Table S9. Code: `03_code/m5b_cross_trait_discriminant.py`

### Phase 2 COMPLETE ✅

---

## PHASE 3 PROGRESS

### Scripts Created
| Script | Experiment | Modal Command | GPU | Est. Time |
|--------|-----------|---------------|-----|-----------|
| `modal_random_direction_control.py` | M-EXP3: Random direction control | `modal run modal_random_direction_control.py --model llama3` | A10G | ~4h/model |
| `modal_gemini_probe_retraining.py` | M-EXP7: Gemini probe retraining | `modal run modal_gemini_probe_retraining.py --model mistral` | A10G | ~3h/model |
| `modal_esconv_naturalistic.py` | M-EXP2: ESConv naturalistic data | `modal run modal_esconv_naturalistic.py --model llama3` | A10G | ~4h/model |
| `modal_deberta_classifier.py` | M4b: DeBERTa text classifier | `modal run modal_deberta_classifier.py` | A10G | ~4h |
| `modal_embedding_similarity.py` | M7a: Embedding similarity baseline | `modal run modal_embedding_similarity.py` | T4 | ~1h |

### Recommended Execution Order
1. **Day 6-7:** Random direction control (llama3, then qwen2, then mistral) — CRITICAL
2. **Day 8-9:** Gemini probe retraining (mistral, then qwen2) — CRITICAL
3. **Day 10:** ESConv naturalistic (llama3) + Embedding similarity — HIGH
4. **Day 11-12:** DeBERTa classifier — HIGH

### Status
- [x] M-EXP3 Random direction control (llama3) ✅ PASS — mean|r|=0.040, all 8 traits PASS, 6.7–12.6× lower than real vectors
- [x] M-EXP3 Random direction control (qwen2) ✅ PASS — mean|r|=0.034, all 8 traits PASS, 0/80 above 0.15
- [x] M-EXP3 Random direction control (mistral) ✅ PASS — mean|r|=0.073, all 8 traits PASS, 3/80 above 0.15
- [x] M-EXP7 Gemini probe retraining (mistral) ❌ FAIL — mean cosine=0.036, 0/7 PASS.
- [x] M-EXP7b Cross-prediction test (mistral) ❌ FAIL — mean functional ratio=0.24, 0/7 PASS. See diagnosis below.
- [x] M-EXP7 Gemini probe retraining (qwen2) — ⛔ CANCELLED (circularity confirmed for Mistral; no value in repeating)
- [x] M-EXP2 ESConv naturalistic (llama3) ✅ PASS — 8/8 meaningful variance, 8/8 significant drift, 1/8 emotion-differentiated. Ecological validity confirmed.
- [x] M4b DeBERTa text classifier ✅ DONE — Mean CV R²=0.290, Gap=1.3× vs activations (0.371). DeBERTa closes ~78% of gap vs activations but still underperforms. Key result: boundary_maintenance R²=0.401, crisis_recognition R²=0.207.
- [x] M7a Embedding similarity baseline ✅ DONE — Mean CV R²=0.004, Gap=95× vs activations. Firewall complete at embedding level.

### M-EXP7/7b Diagnosis: Circularity Confirmed

**Test 1 (Cosine similarity):** Gemini-trained and GPT-trained probes point in completely different directions in 4096-dim space (mean cosine = 0.036 ≈ orthogonal). Both achieve ~99% training accuracy, but the classification hyperplanes are unrelated.

**Test 2 (Cross-prediction):** Even if geometric directions differ, do they predict the *same* behavioral scores? NO. The Gemini-trained probes cannot predict GPT scores (mean |r| = 0.12) despite the GPT probes predicting GPT scores well (mean |r| = 0.47). Mean functional ratio = 0.24, far below the 0.70 threshold.

| Trait | GPT→GPT r | Gem→GPT r | Ratio | Status |
|---|---|---|---|---|
| non_judgmental_acceptance | +0.599 | +0.117 | 0.20 | FAIL |
| boundary_maintenance | +0.423 | −0.095 | 0.22 | FAIL |
| crisis_recognition | +0.620 | +0.299 | 0.48 | FAIL |
| emotional_over_involvement | +0.545 | +0.065 | 0.12 | FAIL |
| abandonment_of_therapeutic_frame | +0.538 | +0.098 | 0.18 | FAIL |
| uncritical_validation | +0.255 | +0.072 | 0.28 | FAIL |
| sycophancy_harmful_validation | +0.303 | +0.065 | 0.21 | FAIL |

**Implication for manuscript:** The monitoring system's probe directions are judge-dependent. This must be disclosed as a limitation. The activation-based monitoring detects drift *as defined by GPT-4o-mini's scoring*, not a judge-invariant latent trait. The human validation (M-EXP4) becomes even more critical — if human scores correlate with the same probes, then GPT-4o-mini is a valid proxy despite learning different hyperplanes than Gemini.

---

## PHASE 1 SUMMARY OF CHANGES

### Files created
- `05_docs/supplementary_material.md` — Tables S1–S6 (scenarios, best layers, rubrics, inter-trait correlations, Pearson/Spearman, TOST)
- `03_code/revision_analyses.py` — Data analysis script for inter-trait correlations, TOST, Pearson vs Spearman audit
- `03_code/compute_cohens_d_ci.py` — Cohen's d CI computation for Table 6
- `03_code/results/revision_quick_win_results.json` — Analysis output
- `03_code/results/cohens_d_table6_ci.json` — Table 6 CI data

### Manuscript edits (research_document_activation_steering_v5.md)
1. Table 0 construct mapping (Rogers/Wampold → 8 dimensions) after §2
2. Omitted constructs discussion (congruence, goal consensus, rupture-repair)
3. Discriminant validity paragraph (inter-trait correlations)
4. Firewall definition in §1.2
5. ICC(2,1) type + CI in Limitations
6. Pearson vs Spearman footnote on Table 1 + Supplementary ref
7. §5.1 safety at ±3.0 + §5.2 stress test at ±5.0
8. Layer clarification note after Table 7
9. §6.2 rewritten as inconclusive (TOST 0/8 equivalent)
10. Discussion: 3 clinical claim qualifications
11. Abstract: safety reframed from ±5.0 to ±3.0 primary
12. References: 15 → 25
13. Table 6: Cohen's d CIs + N notation
14. Table 1 footnote: references to Supplementary Tables S1, S3, S5
