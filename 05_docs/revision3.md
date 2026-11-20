# Revision 3 Plan — "The Instruction Tuning Firewall" (v5 → v6)

**Date created:** 2026-02-13  
**Source:** Structured peer review (Lancet Digital Health simulation)  
**Prior revision plan:** `revision_plan_v5.md` (Phases 1–3 complete, Phase 4 partial)  
**Verdict:** Major Revisions  

---

## Status Summary

The peer review raised **7 major issues**, **10 minor issues**, **7 missing experiments**, and **5 presentation concerns**. Cross-referencing against `revision_plan_v5.md`, many computational experiments are already complete. The **single biggest blocker** is the human validation study (only 2 raters completed; reviewer demands ≥5). The **second biggest concern** is confirmed circularity (Gemini probes failed, functional ratio = 0.24) — human validation is the only remaining mitigation.

### Legend
- ✅ DONE — Completed in prior revision rounds
- 🔄 PARTIAL — Started but insufficient for reviewer
- ⬜ TODO — Not yet addressed
- ❌ BLOCKED — Cannot proceed without prerequisite

---

## MAJOR ISSUES

---

### M1. Circularity in LLM-Judge Evaluation Pipeline

**Reviewer concern (§3.1, §3.4, §7):** GPT-4o-mini scores contrastive probing training labels AND evaluates steering outcomes AND scores the monitoring validation AND assesses harmfulness. Gemini cross-validation (ICC=0.827) demonstrates inter-judge reliability, not validity. Human validation (N=2) is underpowered and post hoc.

**What we already have:**
- ✅ Gemini cross-validation ICC=0.827 (necessary but insufficient)
- ✅ Claude Sonnet 4 cross-validation (3 independent judges)
- ❌ Gemini probe retraining FAILED — cosine similarity=0.036, functional ratio=0.24. **Circularity confirmed:** probe directions are judge-dependent
- 🔄 Human validation with 2 clinical psychologists: ICC(2,1)=0.659 (inter-rater), mean Human–LLM ICC=0.716 (GPT-4o), all 8/8 traits r≥0.50

**What the reviewer demands:**
1. Pre-registered human validation with ≥5 clinicians
2. Sensitivity analysis: main results with human scores replacing LLM judge scores
3. Contrastive probes trained on human-labelled data (circularity-free baseline)

#### Action Items

| ID | Task | Priority | Effort | Status |
|----|------|----------|--------|--------|
| M1.1 | Recruit 3 additional clinical psychologists (total ≥5) | CRITICAL | 2–3 weeks | ⬜ TODO |
| M1.2 | Design stratified rating set that overcomes floor effects: oversample steered responses at coeff ±3.0 on failure-mode traits (sycophancy, uncritical validation, abandonment) | CRITICAL | 2 days | ⬜ TODO |
| M1.3 | Pre-register the validation protocol on OSF (sample size, ICC thresholds, per-trait analysis plan) | CRITICAL | 1 day | ⬜ TODO |
| M1.4 | Each rater scores ~200 responses on all 8 traits (not 2 traits as in current design) | CRITICAL | 3 weeks collection | ⬜ TODO |
| M1.5 | Analysis: (a) Krippendorff's α across ≥5 raters, (b) per-trait ICC(2,1), (c) human-mean vs each LLM judge ICC, (d) Bland-Altman plots | HIGH | 3 days | ⬜ TODO |
| M1.6 | Sensitivity analysis: recompute Tables 1–3 using human clinician scores as ground truth instead of GPT-4o-mini. Report Δr for each trait. | HIGH | 2 days | ⬜ TODO |
| M1.7 | Train contrastive probes on human-labelled activation data for Llama-3-8B (at least 1 model, circularity-free baseline). Compare probe cosine similarity to GPT-trained probes. | HIGH | 3 days | ⬜ TODO |
| M1.8 | Add "Independence of Training and Evaluation" subsection to Methods | MEDIUM | 1 day | ⬜ TODO |
| M1.9 | Disclose Gemini probe failure (functional ratio=0.24) as a confirmed limitation. Frame: "monitoring detects drift as operationalised by GPT-4o-mini scoring; human validation confirms this operationalisation captures clinically meaningful variation" | MEDIUM | 0.5 day | ✅ DONE |

**Dependency chain:** M1.1 → M1.4 → M1.5 → M1.6 → M1.7 → M1.8/M1.9

**Risk:** If human–LLM ICC < 0.60 or human-trained probes diverge from GPT-trained probes, the paper's validity is fundamentally undermined. **Mitigation:** Pilot with 1 additional rater on 30 responses before full study. Current N=2 data (ICC=0.716) is encouraging.

---

### M2. Construct Validity of Eight Dimensions

**Reviewer concern (§2):** Mapping to Rogers/Wampold contains inferential leaps. Emotional over-involvement ≠ inverse of congruence. Inter-trait correlations (r=0.74–0.76) suggest only 3–4 independent dimensions. Needs factor analysis.

**What we already have:**
- ✅ Table 0 construct mapping (Rogers/Wampold/Source columns)
- ✅ Omitted constructs discussion (congruence, goal consensus, rupture-repair)
- ✅ Inter-trait correlation matrix (mean off-diagonal |r|=0.26–0.47)
- ✅ Activation-level discriminant validity (Table S9)
- ✅ Identification of 3–4 factor structure (warmth, professional structure, validation)

**What the reviewer demands:**
1. Exploratory factor analysis (EFA) on 8×8 correlation matrix with eigenvalue structure
2. Drop claim that emotional over-involvement = inverse of congruence
3. Reframe taxonomy as pragmatic, not direct operationalisation

#### Action Items

| ID | Task | Priority | Effort | Status |
|----|------|----------|--------|--------|
| M2.1 | Run EFA on the 8×8 inter-trait correlation matrix (500 scenarios × 5 model configs). Report eigenvalues, scree plot, rotated factor loadings (varimax). Determine whether 3, 4, or 5 factors best fit. | HIGH | 1 day | ✅ DONE |
| M2.2 | If 3–4 factors emerge, discuss whether a reduced trait set is more defensible. Add paragraph to §2. | HIGH | 0.5 day | ✅ DONE |
| M2.3 | Revise Table 0: change emotional_over_involvement mapping from "(Inverse of) Congruence" to "Inferential — related to therapist over-identification" or similar. Remove the direct inverse claim. | MEDIUM | 0.5 day | ✅ DONE |
| M2.4 | Add framing sentence: "We present the eight dimensions as a pragmatic monitoring framework informed by, but not a direct operationalisation of, Rogers (1957) and Wampold (2015)." | MEDIUM | 10 min | ✅ DONE |

**Script needed:** New analysis in `revision_analyses.py` or standalone `factor_analysis.py` using `sklearn.decomposition.FactorAnalysis` or `factor_analyzer` package. Input: existing `m5b_cross_trait_discriminant.json` correlation matrices.

---

### M3. Ecological Validity — Synthetic-Only Evaluation

**Reviewer concern (§§3–6):** All experiments use synthetic scenarios. Clinical and regulatory recommendations are not warranted by synthetic-only evidence.

**What we already have:**
- ✅ ESConv naturalistic evaluation (Llama-3-8B only): 8/8 traits show meaningful variance, 8/8 significant within-conversation drift
- ✅ All clinical/regulatory claims hedged with "pending validation on naturalistic data"
- ✅ Context erosion conversation quality verified (self-BLEU, phase progression)

**What the reviewer demands:**
1. At least one evaluation on naturalistic clinical data (ESConv done ✅)
2. Soften or remove regulatory recommendations
3. Naturalistic evaluation on Qwen2/Mistral (not just Llama-3)
4. Clinician review of GPT-generated user messages from §6.1

#### Action Items

| ID | Task | Priority | Effort | Status |
|----|------|----------|--------|--------|
| M3.1 | Run ESConv naturalistic evaluation on Qwen2-7B and Mistral-7B | HIGH | 2 days compute | ⬜ TODO |
| M3.2 | Report ESConv results in manuscript §3.6 or new §3.7. Include cross-model comparison table. | HIGH | 0.5 day | ⬜ TODO |
| M3.3 | Have 1 clinician review a sample of 20 GPT-generated user messages from context erosion (§6.1) and rate clinical realism on 1–5 scale | MEDIUM | 1 day | ⬜ TODO |
| M3.4 | Further soften regulatory language in Discussion. Replace "could address these requirements" with "could contribute to, but not substitute for, the monitoring infrastructure required by..." | MEDIUM | 0.5 day | ⬜ TODO |
| M3.5 | Add ESConv results to Abstract (1 sentence) | LOW | 10 min | ⬜ TODO |

**Scripts:** `modal_esconv_naturalistic.py` already exists and works for Llama-3. Extend to `--model qwen2` and `--model mistral`.

---

### M4. Causal Interpretation / Discriminant Validity

**Reviewer concern (§3.1, §7):** Steering may perturb the model generically; LLM judge may respond to any perturbation. High off-diagonal correlations (mean |r|=0.26–0.36) undermine specificity. The r=0.899 diagnostic (Figure 2) is circular because "behavioural difference" is measured by the same LLM judge.

**What we already have:**
- ✅ Random direction control: all 3 models PASS (mean |r|=0.034–0.073, all <0.10)
- ✅ Cross-trait discriminant matrix (Table S9)
- ✅ 3–4 factor structure identified

**What the reviewer demands:**
1. Demonstrate that steering trait A changes *only* trait A (current data suggests this fails)
2. Random direction control is done ✅
3. Acknowledge that "any perturbation → judge scores change" is ruled out, but "steering changes a cluster of correlated traits" is NOT ruled out

#### Action Items

| ID | Task | Priority | Effort | Status |
|----|------|----------|--------|--------|
| M4.1 | Compute full 8×8 cross-steering matrix: for each of 8 steered traits, report judge scores on ALL 8 traits. Build specificity matrix: diagonal = target effect, off-diagonal = bleed. | HIGH | 2 days | ⬜ TODO |
| M4.2 | If off-diagonal effects are large, discuss honestly: "Steering modulates a cluster of related traits rather than a single dimension. The 3–4 factor structure means that steering empathetic_responsiveness also shifts non_judgmental_acceptance (shared warmth factor)." | HIGH | 0.5 day | ⬜ TODO |
| M4.3 | Reframe the causal claim: from "eight independent manipulable directions" to "3–4 therapeutic factors that can be individually targeted, with expected cross-trait bleed on conceptually related dimensions" | MEDIUM | 0.5 day | ✅ DONE |
| M4.4 | Acknowledge circularity in the r=0.899 diagnostic explicitly: "The behavioural difference metric is itself LLM-judged, meaning this analysis demonstrates internal consistency of the steering→judge pipeline, not independent causal validation." | MEDIUM | 10 min | ✅ DONE |

**Data source:** The 12,000-response expanded corpus already has responses steered on each trait. Need to score each response on ALL 8 traits via GPT-4o-mini (currently only scored on the target trait). This requires ~96,000 additional judge calls (12,000 × 7 off-target traits). Alternatively, use a subset (1,200 responses × 8 traits = 9,600 calls).

---

### M5. The Sensitivity Gap — Shifting Goalposts

**Reviewer concern (§4.2):** The paper reports 7.9× (univariate R²), 3.3× (RF), and 1.3× (DeBERTa) gaps. DeBERTa exceeds activation monitoring on 3/8 traits. The actual finding is much weaker than the title suggests.

**What we already have:**
- ✅ Full comparison ladder: embedding (R²=0.004) → RF (0.112) → DeBERTa (0.290) → activation (0.371)
- ✅ Reframed headline from "2.9× gap" to "comparison ladder"
- ✅ Table 9 method comparison with latency/cost

**What the reviewer demands:**
1. Choose ONE primary comparison (recommend DeBERTa)
2. Revise title and abstract to reflect selective advantage, not general superiority
3. Separate accuracy argument from cost argument

#### Action Items

| ID | Task | Priority | Effort | Status |
|----|------|----------|--------|--------|
| M5.1 | Lead with DeBERTa comparison (Table 5c) as the primary fairest comparison. Move univariate (Table 5a) to supplementary. | HIGH | 1 day | ⬜ TODO |
| M5.2 | Revise title. Options: (a) "Activation Monitoring for Therapeutic Persona Drift in Mental Health Chatbots: A Cross-Architecture Validation Study" (b) "Internal Monitoring Outperforms Text Analysis on Safety-Critical Therapeutic Traits" (c) Keep "Instruction Tuning Firewall" but add subtitle specifying it's about the gradient. **Decision needed from authors.** | HIGH | 0.5 day | ✅ DONE (kept title + added cross-architecture subtitle) |
| M5.3 | Revise abstract: separate the accuracy finding ("activation monitoring retains a selective advantage on safety-critical traits") from the cost finding ("at ~1000× lower computational cost than DeBERTa"). Currently these are conflated. | HIGH | 0.5 day | ✅ DONE |
| M5.4 | Add explicit paragraph in §4.2 acknowledging: "DeBERTa exceeds activation monitoring on 3/8 traits (boundary maintenance, emotional over-involvement, uncritical validation). The advantage of activation monitoring is concentrated on the two most safety-critical traits: crisis recognition (2.6×) and abandonment of therapeutic frame (2.5×)." This is already partially present but should be more prominent. | MEDIUM | 0.5 day | 🔄 PARTIAL |
| M5.5 | Create a 2×2 framing: (accuracy × cost) showing that activation monitoring wins on (high accuracy + low cost) for safety traits, while DeBERTa wins on (higher accuracy + high cost) for 3 non-safety traits | MEDIUM | 0.5 day | ⬜ TODO |

---

### M6. Sample Size and Statistical Power

**Reviewer concern (§3.1, §4.2):** N=50 per trait is small. Power analysis MDE=r≥0.49 at N=30 means the design is underpowered for weaker traits. Boundary maintenance (r=0.302, CI [0.11, 0.47]) may not be robustly validated.

**What we already have:**
- ✅ Power analysis script with power curves
- ✅ Expanded corpus (N=12,000) for text analysis
- ✅ Wide CIs acknowledged for boundary maintenance

**What the reviewer demands:**
1. Post-hoc power for each trait in Table 1
2. Flag boundary_maintenance as marginal

#### Action Items

| ID | Task | Priority | Effort | Status |
|----|------|----------|--------|--------|
| M6.1 | Compute and report post-hoc observed power for each trait in Table 1 (N=50, observed r, α=0.05 two-sided). Add as column to Table 1 or footnote. | HIGH | 0.5 day | ✅ DONE |
| M6.2 | Flag boundary_maintenance explicitly: "Boundary maintenance (r=0.302) is the weakest validated trait, with a 95% CI lower bound of 0.11. While statistically significant, this effect should be considered marginal and in need of replication at larger N." | MEDIUM | 10 min | ✅ DONE |
| M6.3 | Consider re-running boundary_maintenance validation at N=100 (10 scenarios × 10 prompts × 1 model) for a tighter CI | LOW | 1 day compute | ⬜ TODO |

---

### M7. Missing Baselines — Linear Probes

**Reviewer concern (§4):** No comparison to linear probes trained on activation data (Belinkov 2022). Do they outperform directional projection? No logistic regression on activation snapshots.

**What we already have:**
- ✅ Comparison ladder on the text side (embedding → RF → DeBERTa)
- ✅ Contrastive probing (which IS a form of linear probe, but unsupervised)
- ❌ No supervised linear probe baseline on the activation side

#### Action Items

| ID | Task | Priority | Effort | Status |
|----|------|----------|--------|--------|
| M7.1 | Train supervised linear probes (logistic regression) on activation data using LLM judge labels as targets. For each trait × model: extract hidden states at best layer, fit LR to predict judge score, 5-fold CV grouped by scenario. Report R². | HIGH | 2 days | ⬜ TODO |
| M7.2 | Train supervised linear probes on human clinician labels (after M1.4 completes). This is the circularity-free activation baseline. | HIGH | 1 day (after M1.4) | ❌ BLOCKED on M1.4 |
| M7.3 | Compare: (a) directional projection R² vs (b) supervised linear probe R² vs (c) DeBERTa R². If supervised probe ≈ directional projection, this validates the unsupervised approach. If supervised probe >> directional projection, the current method is suboptimal. | HIGH | 0.5 day | ⬜ TODO |
| M7.4 | Add supervised probe results to Table 9 (method comparison) | MEDIUM | 0.5 day | ⬜ TODO |

**Script needed:** New `modal_linear_probe_baseline.py`. Use `sklearn.linear_model.Ridge` or `LogisticRegression` on mean-pooled or last-token activations. Input: existing hidden state extractions from monitoring runs.

---

## MINOR ISSUES

---

| ID | Issue | Reviewer Section | Status | Action | Effort |
|----|-------|-----------------|--------|--------|--------|
| m1 | "Instruction Tuning Firewall" terminology — implies binary barrier, evidence shows gradient | §1.2 | ✅ DONE (definition added) | Consider renaming to "instruction tuning filter" or keeping with stronger caveat. Author decision needed. | 10 min |
| m2 | Notation inconsistency — switches between r, R², |r|, "equivalent r" without clarifying | Throughout | ✅ DONE | Added notation conventions table before §3 | 1 hour |
| m3 | Table 9 estimated value — LLM judge R² "estimated from ICC=0.827" is not measured | Table 9 | ✅ DONE | Footnote clarified: “upper bound... reflects inter-judge consistency, not detection accuracy against ground truth” | 0.5 day |
| m4 | Figures are SVG references, not embedded | Throughout | ✅ DONE (SVGs generated) | Ensure figures are embedded in the final PDF submission. Convert to 300dpi PNG or vector EPS per journal requirements. | 1 hour |
| m5 | ESConv corpus in codebase but not in manuscript | `step0b_process_esconv.py` | 🔄 PARTIAL | ESConv results exist (M3.1) but not yet written into manuscript. Write §3.7 or §6.5 reporting ESConv findings. | 0.5 day |
| m6 | Reference gaps — Burns et al. (2023) CCS, Marks et al. (2024) SAEs, Hubinger et al. (2024) sleeper agents | References | ✅ DONE | Added as refs 26–28 | 1 hour |
| m7 | "Model welfare" statement in Ethics — unusual for clinical journal | Ethics | ✅ DONE | Removed paragraph | 5 min |
| m8 | System prompt not provided | Methods | ✅ DONE | Added full/short/none system prompts to Supplementary Table S12 + reference in Methods Summary | 30 min |
| m9 | Floor effects limit human validation to 5/8 traits | §7 | ✅ DONE | Already well-documented in manuscript (§7 + Limitations + Conclusions). Abstract now references floor effects. | 30 min |
| m10 | Context erosion user messages — clinical realism not verified | §6.1 | ⬜ TODO | Same as M3.3: have 1 clinician review sample of 20 GPT-generated user messages and rate realism. | 1 day |

---

## MISSING EXPERIMENTS

---

### EXP1. Expanded Human Clinician Validation ⭐ HIGHEST PRIORITY

**What:** ≥5 licensed therapists rate ~200 steered responses on all 8 traits using 1–7 anchored scale  
**Why:** Only external anchor for validity. Current N=2 is insufficient for stable per-trait ICC. Reviewer demands pre-registration.

| Sub-task | Detail | Effort |
|----------|--------|--------|
| EXP1.1 | Recruit 3 additional clinical psychologists (diverse training backgrounds: CBT, psychodynamic, humanistic) | 2 weeks |
| EXP1.2 | Redesign rating set: 200 responses, stratified to oversample extreme coefficients (±3.0) and failure-mode traits. Include 20% overlap for intra-rater reliability. | 2 days |
| EXP1.3 | Pre-register on OSF: hypotheses (human–LLM ICC ≥ 0.60, per-trait r ≥ 0.40 for ≥6/8 traits), sample size justification, analysis plan | 1 day |
| EXP1.4 | Raters score independently (blinded to model, trait, coefficient) | 3 weeks |
| EXP1.5 | Analyse: Krippendorff's α (≥5 raters), ICC(2,1) per trait, human-mean vs each LLM judge, Bland-Altman | 3 days |
| EXP1.6 | Sensitivity analysis: recompute Tables 1–3 with human ground truth | 2 days |

**Success criteria:** Human–LLM ICC ≥ 0.60, per-trait r ≥ 0.40 for ≥6/8 traits  
**Blocks:** M1.5, M1.6, M1.7, M7.2

---

### EXP2. ESConv Naturalistic Evaluation (Qwen2 + Mistral)

**What:** Extend ESConv evaluation from Llama-3 only → all 3 models  
**Why:** Ecological validity currently demonstrated for 1/3 models

| Sub-task | Detail | Effort |
|----------|--------|--------|
| EXP2.1 | Run `modal_esconv_naturalistic.py --model qwen2` | 4h compute |
| EXP2.2 | Run `modal_esconv_naturalistic.py --model mistral` | 4h compute |
| EXP2.3 | Cross-model comparison table: variance, drift significance, emotion differentiation | 0.5 day |

**Success criteria:** ≥6/8 traits show meaningful variance on all 3 models  
**Script:** `modal_esconv_naturalistic.py` (exists, tested on Llama-3)

---

### EXP3. Cross-Steering Specificity Matrix

**What:** Score each steered response on ALL 8 traits (not just the target)  
**Why:** Required for discriminant validity (M4.1). Currently each response is only scored on its target trait.

| Sub-task | Detail | Effort |
|----------|--------|--------|
| EXP3.1 | Select subset: 1,200 responses (primary corpus) × 7 off-target traits = 8,400 judge calls | 0.5 day compute |
| EXP3.2 | Build 8×8 specificity matrix: rows = steered trait, columns = judged trait, cells = r(coefficient, judge_score) | 0.5 day |
| EXP3.3 | Report: diagonal mean r, off-diagonal mean r, ratio | 0.5 day |

**Success criteria:** Diagonal mean r ≥ 2× off-diagonal mean r  
**Risk:** Given inter-trait correlations (r=0.74 for sycophancy×uncritical_validation), ratio may be <2× for correlated trait pairs. This is acceptable if explained.

---

### EXP4. Supervised Linear Probe Baseline

**What:** Train supervised linear probes on activation data to compare against unsupervised directional projection  
**Why:** Reviewer M7 concern — is the directional projection method optimal?

| Sub-task | Detail | Effort |
|----------|--------|--------|
| EXP4.1 | Extract hidden states at best layer for all 12,000 responses (may already exist in monitoring runs) | 0.5 day |
| EXP4.2 | Fit Ridge regression: activation → steering coefficient, 5-fold CV grouped by scenario | 0.5 day |
| EXP4.3 | Fit Ridge regression: activation → LLM judge score, 5-fold CV | 0.5 day |
| EXP4.4 | Compare R²: directional projection vs supervised probe vs DeBERTa | 0.5 day |

**Success criteria:** Directional projection R² ≥ 0.80 × supervised probe R² (validates unsupervised approach)  
**Script needed:** New `modal_linear_probe_baseline.py`

---

### EXP5. Exploratory Factor Analysis

**What:** EFA on 8×8 inter-trait correlation matrix  
**Why:** Reviewer M2 demands empirical dimensionality assessment

| Sub-task | Detail | Effort |
|----------|--------|--------|
| EXP5.1 | Compute EFA using existing correlation matrices from `m5b_cross_trait_discriminant.json` | 0.5 day | ✅ DONE |
| EXP5.2 | Report eigenvalues, scree plot, varimax-rotated loadings for each model config | 0.5 day | ✅ DONE |
| EXP5.3 | Determine optimal factor count (parallel analysis, MAP test) | 0.5 day | ✅ DONE |

**Success criteria:** Clear 3–4 factor structure consistent with qualitative clustering already reported  
**Script needed:** New `factor_analysis.py` using `factor_analyzer` package

---

### EXP6. Post-Hoc Power Analysis for Table 1

**What:** Compute observed power for each trait at N=50

| Sub-task | Detail | Effort |
|----------|--------|--------|
| EXP6.1 | For each trait in Table 1, compute post-hoc power at α=0.05 two-sided using observed r and N=50 | 2 hours | ✅ DONE |
| EXP6.2 | Add power column to Table 1 or report in footnote | 30 min | ✅ DONE |

**Script:** Extend `power_analysis.py`

---

### EXP7. Larger Models / Closed-Source APIs

**What:** Test whether findings generalise to 70B+ models and/or GPT-4/Claude  
**Why:** All 3 tested models are 7–8B. Scale effects unknown.

| Sub-task | Detail | Effort |
|----------|--------|--------|
| EXP7.1 | Test activation steering on Llama-3-70B (requires A100 GPU or quantisation to fit in memory) | 2–3 days |
| EXP7.2 | For closed-source models (no activation access): test DeBERTa and pragmatic text features as monitoring proxies | 1 day |
| EXP7.3 | Discuss scale limitations explicitly in Limitations section | 0.5 day |

**Priority:** MEDIUM — important for generalisability but not blocking  
**Risk:** 70B models may require different optimal layers, negating the specific layer selections. This is acknowledged in the current manuscript.

---

## PRESENTATION & FRAMING

---

| ID | Issue | Action | Effort |
|----|-------|--------|--------|
| P1 | Paper reads as ML paper with clinical motivation; needs stronger clinical framing for Lancet audience | Restructure Introduction: lead with patient safety case (what happens when a suicidal user gets a drifted response?). Compress ML methods into a Methods section with clinical implications foregrounded. | 1 day |
| P2 | Abstract is too dense and technical | Rewrite abstract: (1) problem (1 sentence), (2) method (1 sentence), (3) key finding 1: activation monitoring validated across 3 architectures (1 sentence), (4) key finding 2: text analysis cannot match activation monitoring on safety-critical traits even with DeBERTa (1 sentence), (5) limitation: synthetic data, need clinical validation (1 sentence) | 2 hours |
| P3 | Title may be misleading — "firewall" implies impenetrability, but evidence shows a gradient | Author decision: (a) keep current title with explicit subtitle, (b) change to clinical framing (e.g., "Activation Monitoring Detects Therapeutic Persona Drift That Text Analysis Cannot: A Cross-Architecture Validation"), (c) keep "firewall" but add "gradient" | 30 min |
| P4 | Discussion buries limitations after 2 pages of clinical recommendations | Move limitations paragraph to immediately after main findings: "Before discussing implications, we note three critical caveats: all experiments use synthetic scenarios, the LLM judge evaluation contains confirmed circularity (§M1.9), and the human validation sample is small." | 1 hour |
| P5 | Figures not embedded in manuscript | Embed all 5 SVG/PDF figures in final submission. Verify journal format requirements (EPS? TIFF? 300dpi?). | 1 hour |

---

## EXECUTION PLAN

---

### Phase A: Quick Analytical Wins (Week 1)
*No new compute required. Use existing data.*

| Day | Tasks | Issues Addressed |
|-----|-------|-----------------|
| 1 | EXP5 (factor analysis on existing correlation matrices) | M2.1, M2.2 |
| 1 | EXP6 (post-hoc power for Table 1) | M6.1, M6.2 |
| 1 | Notation table, fix Table 9 estimated value, reference additions | m2, m3, m6 |
| 2 | Revise title/abstract/framing per P1–P4 | M5.1–M5.5, P1–P4 |
| 2 | Acknowledge circularity (M1.9), reframe causal claims (M4.3, M4.4) | M1.9, M4.3, M4.4 |
| 2 | Floor effects disclosure, system prompt to supplementary, model welfare edit | m7, m8, m9 |

### Phase B: New Compute Experiments (Weeks 2–3)
*Require Modal GPU time*

| Day | Tasks | Issues Addressed | GPU |
|-----|-------|-----------------|-----|
| 3–4 | EXP3: Cross-steering specificity matrix (8,400 judge calls) | M4.1, M4.2 | None (API calls) |
| 5 | EXP2: ESConv on Qwen2 + Mistral | M3.1, M3.2 | 2× A10G |
| 6–7 | EXP4: Supervised linear probe baseline | M7.1, M7.3 | A10G |
| 8 | M3.3: Clinician reviews 20 context erosion user messages | m10 | None |
| 8 | Table 9 LLM judge R² — compute actual R² | m3 | None (API calls) |

### Phase C: Human Validation (Weeks 4–8) ⭐ CRITICAL PATH
*Recruitment + data collection*

| Week | Tasks | Issues Addressed |
|------|-------|-----------------|
| 4 | EXP1.1–1.3: Recruit raters, design rating set, pre-register on OSF | M1.1–M1.3 |
| 5–6 | EXP1.4: Raters score ~200 responses | M1.4 |
| 7 | EXP1.5: Analyse inter-rater reliability, human–LLM agreement | M1.5 |
| 7 | EXP1.6: Sensitivity analysis with human ground truth | M1.6 |
| 8 | EXP1.7/M7.2: Train human-labelled contrastive probes + supervised probes | M1.7, M7.2 |

### Phase D: Final Manuscript Revision (Weeks 9–10)

| Task | Details |
|------|---------|
| Integrate all new results into manuscript v6 | Tables 1–3 updated, new Tables for EFA, specificity matrix, linear probes |
| Write new "Independence of Training and Evaluation" subsection | M1.8 |
| Restructure for clinical audience | P1 |
| Rewrite abstract | P2 |
| Final reference check + formatting | m6, P5 |
| Co-author review cycle | 1 week |

---

## SUCCESS CRITERIA FOR RESUBMISSION

| Criterion | Threshold | Current Status | Phase |
|-----------|-----------|---------------|-------|
| Human–LLM ICC (≥5 raters) | ≥ 0.60 | 🔄 0.716 (N=2, need ≥5) | C |
| Per-trait human–LLM r | ≥ 0.40 for ≥6/8 traits | 🔄 r≥0.50 all 8 (N=2) | C |
| Sensitivity analysis Δr | Mean Δr < 0.10 (human vs LLM ground truth) | ⬜ Not started | C |
| Factor analysis | Clear 3–4 factor structure | ✅ DONE (2–3 factors, 55–76% var) | A |
| Cross-steering specificity | Diagonal mean r ≥ 1.5× off-diagonal | ⬜ Not started | B |
| Supervised probe comparison | Directional R² ≥ 0.80 × supervised R² | ⬜ Not started | B |
| ESConv 3 models | ≥6/8 traits variance on each model | 🔄 Llama-3 done (8/8) | B |
| Post-hoc power reported | All traits in Table 1 | ✅ DONE (4/8 adequate, 4/8 underpowered) | A |
| Title/abstract revised | Reflects DeBERTa comparison honestly | ✅ DONE | A |
| Pre-registered validation | OSF registration | ⬜ Not started | C |

---

## RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Human validation ICC < 0.60 with ≥5 raters | LOW (current N=2 is 0.716) | FATAL | Pilot with 1 additional rater first; iterate rubric anchors if needed |
| Cross-steering specificity ratio < 1.5× | MEDIUM (given r=0.74 sycophancy×uncritical_validation) | HIGH | Reframe: "monitoring targets 3–4 therapeutic factors, not 8 independent traits" — this is honest and still useful clinically |
| Supervised probes greatly outperform directional projection | LOW | MEDIUM | Pivot: present supervised probes as the recommended method and directional projection as a no-training-required alternative |
| ESConv fails on Qwen2/Mistral | LOW (works on Llama-3) | MEDIUM | Report Llama-3 only with cross-model as limitation |
| Factor analysis yields 2 factors (not 3–4) | MEDIUM | MEDIUM | Reduce framework to 2 meta-traits (warmth + professional structure) with sub-dimensions |
| 70B model shows different layer structure | HIGH | LOW | Acknowledge as limitation; not blocking for resubmission |

---

## FILE MAPPING

| Revision Task | Script | Input Data | Output |
|--------------|--------|-----------|--------|
| M2.1 (EFA) | New: `factor_analysis.py` | `m5b_cross_trait_discriminant.json` | Scree plot + loadings table |
| M4.1 (Cross-steering) | New: `modal_cross_steering_specificity.py` | Existing 1,200 corpus | 8×8 specificity matrix |
| M6.1 (Post-hoc power) | Extend: `power_analysis.py` | Table 1 values | Power column for Table 1 |
| M7.1 (Linear probes) | New: `modal_linear_probe_baseline.py` | Hidden states from monitoring | Probe R² per trait |
| EXP1 (Human validation) | Extend: `analyze_human_llm_icc.py` | New clinician ratings | ICC, Krippendorff's α |
| EXP2 (ESConv Qwen2/Mistral) | Existing: `modal_esconv_naturalistic.py` | ESConv dataset | `esconv_naturalistic_{model}.json` |
| m3 (Table 9 R²) | New analysis in `revision_analyses.py` | Judge scores + steering coefficients | Measured LLM judge R² |
| m8 (System prompt) | Extract from `step1_validate_traits.py` | Code | Supplementary text |

---

## TIMELINE SUMMARY

```
Week 1:  Phase A — Analytical quick wins (EFA, power, framing, notation)
Week 2-3: Phase B — New compute (cross-steering, ESConv×3, linear probes)
Week 4:  Phase C start — Recruit raters, pre-register, design rating set
Week 5-6: Phase C — Clinician ratings collected
Week 7:  Phase C — Analysis of human validation
Week 8:  Phase C — Human-labelled probes, sensitivity analysis
Week 9-10: Phase D — Final manuscript revision + co-author review
Week 11: Resubmission
```

**Total estimated time: 11 weeks**  
**Critical path: Human validation (Weeks 4–8)**  
**Estimated compute cost: ~$50–100 Modal credits (ESConv × 2 models + linear probes + cross-steering judge calls)**

---

*This plan addresses all 7 major issues, 10 minor issues, 7 missing experiments, and 5 presentation concerns from the peer review. The single non-negotiable item is the expanded human validation (EXP1/M1). Everything else can be completed in parallel.*
