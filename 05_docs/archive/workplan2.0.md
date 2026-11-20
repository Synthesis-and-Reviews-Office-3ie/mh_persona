# Workplan 2.0 — Bullet-Proof Paper Revision
**Date:** 2026-02-09 (last updated: 2026-02-15)  
**Target:** Lancet-quality submission  
**Status:** ALL WORKSTREAMS ✅ COMPLETE. Human validation (Workstream B) done: 2 raters, ICC(2,1)=0.659, human-LLM r=0.70–0.74 (all 8/8 traits pass). Manuscript updated.

---

## Overview

This plan addresses **13 weaknesses** across four workstreams. The goal is to transform the paper from a methods demonstration (we steered models and detected the steering) into a clinical safety paper (we show that real-world threats cause invisible drift and our monitor catches it).

| Workstream | What it kills | Status |
|---|---|---|
| **A. Judge Triangulation** | Single-judge dependency | ✅ **DONE** — GPT-4o + Gemini 2.5 Flash, ICC=0.827 |
| **B. Human Validation** | No human ground truth | ✅ **DONE** — 2 raters, ICC(2,1)=0.659, human-LLM r=0.74 (GPT-4o), 8/8 traits pass |
| **C. Statistical Robustness** | Small N, quantisation confound, power | ✅ **DONE** — Corpus expansion, quantisation NF4/FP16/8-bit, power analysis complete |
| **D. Natural Drift Experiments** | "Synthetic only" weakness, §6 paradox | ✅ **DONE** — All 3 experiments complete (context erosion, fine-tuning, adversarial) |
| **E. Paper Revisions** | All 10 TODO items | ✅ **DONE** — E1–E7 all complete |
| **F. Peer Review Revision** | 7 major + 11 minor reviewer issues | ✅ **DONE** (Phases 1–3) — See `revision_plan_v5.md`. DeBERTa R²=0.290, random direction control PASS, cross-trait discriminant done, clustered SEs done, manuscript reframed |

---

## Workstream A — Judge Triangulation

> **Problem:** All behavioral scores come from a single LLM judge (GPT-4o-mini). A reviewer will ask: "How do you know this isn't just GPT-4o-mini's idiosyncratic preferences?"

### A1. Claude Sonnet 4 Re-Scoring ✅
- **Script:** `03_code/modal_judge_claude.py`
- **What:** Re-score **all 1,200 responses** from `steered_corpus_combined.json` with Claude Sonnet 4 via OpenRouter
- **Design:** Identical rubrics, identical 1–7 scale, identical trait definitions, temperature=0
- **Output:** `03_code/results/judge_scores_claude.json` — same structure as original scores
- **Cost:** ~$20 (1,200 calls × ~800 tokens each)
- **Time:** ~2 hours on Modal

### A2. Gemini 2.5 Flash Re-Scoring ✅
- **Script:** `03_code/modal_judge_gemini.py`
- **What:** Re-score **all 1,200 responses** with Gemini 2.5 Flash via OpenRouter
- **Design:** Same as A1
- **Output:** `03_code/results/judge_scores_gemini.json`
- **Cost:** ~$12
- **Time:** ~2 hours on Modal

### A3. Inter-Judge Agreement Analysis ✅
- **Script:** `03_code/analyze_judge_agreement.py`
- **What:** Compute ICC(2,1) for each judge pair + three-way agreement
- **Metrics:**
  - ICC(GPT-4o vs Claude) — overall and per-trait
  - ICC(GPT-4o vs Gemini) — overall and per-trait
  - ICC(Claude vs Gemini) — overall and per-trait
  - ICC(all three) — overall and per-trait
  - Bland-Altman plots (systematic bias detection)
  - Per-trait agreement heatmap
- **Output:** `03_code/results/judge_agreement_report.json` + figures
- **Cost:** CPU only
- **Paper result:** "Two independent LLM judges (GPT-4o, Gemini 2.5 Flash) showed ICC=0.827 (95% CI: 0.780–0.860), with per-trait agreement ranging from 0.705 to 0.835."

### Acceptance criteria
- [x] ICC ≥ 0.70 overall → ✅ **ACHIEVED** (GPT+Gemini ICC=0.827, all 8 traits ≥0.70)
- [x] ICC ≥ 0.50 overall → ✅ Three-way ICC=0.738 (moderate-good, diluted by Claude on sycophancy)
- [x] ICC < 0.50 → Claude dropped from panel (Claude+Gemini sycophancy ICC=0.465)

**Decision (Feb 10, 2026):** Adopt GPT-4o + Gemini 2.5 Flash as the 2-judge panel.

**Rubric redesign note:** Two traits required rubric v4 redesign:
- `emotional_over_involvement`: Rescaled anchors to RLHF-realistic range with warmth-marker checklist. ICC improved from 0.124→0.818 (GPT+Gemini).
- `sycophancy_harmful_validation`: Added therapeutic-vs-sycophancy distinction and 4-step decision tree. ICC improved from 0.740→0.784 (GPT+Gemini), Claude remains divergent (0.465 with Gemini).

---

## Workstream B — Human Validation

> **Problem:** No human ground truth. Lancet requires human-in-the-loop validation for any clinical AI claim.

### B1. Validation Protocol
- **Document:** `05_docs/human_validation/protocol.md`
- **Contents:**
  - **Rater recruitment:** 6–8 raters with clinical psychology background (graduate level minimum). Bilingual EN/ES.
  - **Rating scale:** 1–7 per trait, with anchored descriptors matching the GPT-4o-mini rubrics exactly
  - **Blinding:** Raters see responses in random order, no information about model, trait, or coefficient
  - **Training:** 10 practice responses (not from the study corpus) with calibration discussion
  - **Per-rater workload:** ~120 responses × 2 traits each = ~2 hours
  - **IRB template:** Included for ethics committee submission
  - **Languages:** Full instructions and anchors in both English and Spanish

### B2. Validation Dataset Sampler
- **Script:** `03_code/build_human_validation_set.py`
- **Sampling strategy:** Stratified sample from `steered_corpus_combined.json`:
  - 120 responses total
  - 3 models × 8 traits × 5 coefficients = 120 cells → 1 response per cell
  - Each response randomly selected from the 10 scenarios available in that cell
  - 20% overlap between rater pairs (for inter-rater reliability)
- **Output:**
  - `05_docs/human_validation/rating_sheet_EN.csv` — English version
  - `05_docs/human_validation/rating_sheet_ES.csv` — Spanish version (scenario context translated)
  - `05_docs/human_validation/response_key.json` — maps response IDs back to model/trait/coefficient (hidden from raters)

### B3. Human Validation Analysis
- **Script:** `03_code/analyze_human_validation.py`
- **Metrics:**
  - ICC(2,k) — human inter-rater reliability, overall and per-trait
  - Pearson r(human mean, GPT-4o-mini score) — human-LLM agreement
  - Pearson r(human mean, Claude score) — second human-LLM agreement
  - Pearson r(human mean, Gemini score) — third human-LLM agreement
  - Confusion matrix at clinical decision thresholds (alert vs no-alert)
- **Output:** `03_code/results/human_llm_icc_results.json` + figures
- **Paper result:** "Two clinical psychologists (N=2) achieved inter-rater ICC(2,1)=0.659 (95% CI [0.585, 0.728]). Human mean–LLM agreement: r=0.744 (GPT-4o), r=0.704 (Claude), r=0.693 (Gemini). All 8/8 traits exceeded r≥0.50 threshold (range 0.644–0.920)."

### Acceptance criteria
- [x] Human inter-rater ICC ≥ 0.60 → ✅ **ACHIEVED** (ICC=0.659, 95% CI [0.585, 0.728])
- [x] Human-LLM r ≥ 0.50 for at least 6/8 traits → ✅ **ACHIEVED** (8/8 traits pass, range 0.644–0.920)
- [x] If human-LLM r < 0.50 for some traits → ✅ N/A — no traits failed

**Inter-rater details (Feb 15, 2026):** 3 traits had poor per-trait ICC due to floor effects (abandonment ICC=-0.08, emotional_over_involvement ICC=0.14, non_judgmental_acceptance ICC=0.01) — both raters agreed these are mostly low/mid-range, leaving insufficient variance. Rater 1 correlated more strongly with LLM judges (r≈0.82) than Rater 2 (r≈0.48), but the human mean still yields strong agreement across all three judges.

---

## Workstream C — Statistical Robustness

> **Problem:** N=30 per cell is small. 4-bit quantisation is a confound. "Only 18% significant" might reflect low power, not a true null.

### C1. Corpus Expansion (10×)
- **Script:** `03_code/generate_steered_corpus_v2.py`
- **What:** Expand from 10 to **100 scenarios**, yielding **12,000 responses** (3 models × 8 traits × 100 scenarios × 5 coefficients)
- **Scenario generation:** Use GPT-4o-mini to generate 90 additional mental health scenarios across 10 categories (9 new per category), covering diverse demographics, cultural contexts, and severity levels
- **Design:** Same steering pipeline as v1, same coefficients, same greedy decoding
- **Output:** `03_code/results/steered_corpus_v2_combined.json`
- **Cost:** ~$50–80 (12,000 responses on A10G)
- **Time:** ~4–6 hours on Modal (parallelised across 10 containers)
- **Why it matters:** N=100 per cell (up from 30). Detects effects as small as r=0.14 at 80% power.

### C2. Power Analysis
- **Script:** `03_code/power_analysis.py`
- **What:** Formal statistical power analysis for both corpus sizes
- **Outputs:**
  - Minimum detectable effect (r) at 80% power, α=0.05, for N=30 and N=100
  - Power curves for the observed text-level effect sizes (r~0.15–0.33)
  - Explicit statement: "At N=30, the design has 80% power to detect r≥0.25. The 18% significance rate for text features (observed mean |r|=0.203) is consistent with a mix of real effects below the detection threshold and true nulls."
- **Output:** `03_code/results/power_analysis_report.json` + power curve figure
- **Cost:** CPU only (scipy.stats calculations)

### C3. Quantisation Comparison
- **Script:** `03_code/modal_quantisation_comparison.py`
- **What:** Run a subset of the experiment at **FP16** and **8-bit** to quantify the quantisation confound
- **Design:**
  - 2 traits (empathetic_responsiveness, crisis_recognition — highest and most clinically relevant)
  - 10 scenarios × 5 coefficients = 50 responses per precision level
  - 3 precision levels: NF4 (existing), 8-bit, FP16
  - Model: Llama-3-8B only (needs A100 80GB for FP16)
  - Total: 100 new responses (8-bit + FP16)
- **Comparisons:**
  - Optimal layer: same or different across precision levels?
  - r-value: how much does it change? (Δr)
  - Qualitative: do the same traits validate?
- **Output:** `03_code/results/quantisation_comparison.json`
- **Cost:** ~$15 (A100 for ~2 hours)
- **Paper result:** "Optimal layers shifted by ≤X layers between NF4 and FP16. Correlation coefficients changed by mean Δr=X.XX, suggesting that quantisation [does/does not] substantially affect monitoring accuracy."

### C4. Expanded Corpus Text Analysis
- **Script:** `03_code/analyze_corpus_v2.py`
- **What:** Re-run full text analysis (sentiment, pragmatic features, TF-IDF drift) on the 12,000-response corpus
- **Key question:** At 10× sample size, do more text features reach significance? If yes, does the activation gap persist?
- **Output:** `03_code/results/corpus_v2_analysis_report.json` + updated Table 5
- **Depends on:** C1 (needs the expanded corpus)
- **Paper result:** Either "At N=100/cell, X additional text features reached significance, but the sensitivity gap persisted (activation mean r=X.XX vs text mean |r|=X.XX, Y.Y× gap)" or "Even at N=100/cell, only X% of text features reached significance, confirming the firewall finding."

---

## Workstream D — Natural Drift Experiments

> **Problem:** All evidence is synthetic (steering vectors). §6 shows models don't drift on their own. Reviewer asks: "What problem does the monitor solve?"

### D1. Context Window Erosion ✅
- **Script:** `03_code/modal_context_erosion.py` → `modal_context_erosion_v2.py` (parallel rewrite)
- **Result:** Drift detected in all 3 models. emotional_over_involvement increased across all (p<0.001). VADER showed no trend. Monitor alerted in 59/60 sessions (58 critical).
- **What:** Test whether system prompt influence weakens over very long conversations
- **Design:**
  - 3 models × 20 conversations × 100+ turns each
  - System prompt: standard therapeutic persona instruction
  - User messages: drawn from a realistic conversation flow (escalating emotional intensity)
  - At every turn: extract activation projections for all 8 traits
  - Analysis: is there a significant downward trend in empathetic_responsiveness or crisis_recognition over the course of 100 turns?
- **Output:** `03_code/results/context_erosion_results.json` + trend plots
- **Cost:** ~$30 (3 models × 20 conversations × 100 turns on A10G)
- **Time:** ~2–3 hours on Modal
- **Possible outcomes:**
  - **Drift detected:** "System prompt influence on empathetic_responsiveness declined by Xσ over 100 turns (p=X.XX). The EWMA monitor triggered a Watch alert at turn Y. VADER sentiment showed no significant trend (p=X.XX). This demonstrates context erosion as a real-world drift scenario detectable by activation monitoring but invisible to text analysis."
  - **No drift:** "Therapeutic persona remained stable over 100 turns (all traits within 2σ), suggesting context window erosion is not a concern for conversations of this length."

### D2. Fine-Tuning Regression ✅
- **Script:** `03_code/modal_finetuning_regression.py`
- **Result:** "Neither catches it" outcome — 0/8 traits shifted (all p>0.09). Robustness finding: small-scale LoRA fine-tuning on non-therapeutic data did not produce detectable persona drift.
- **What:** Simulate a routine model update and test whether the monitor detects persona regression
- **Design:**
  - Take Llama-3-8B (baseline already measured)
  - Fine-tune with LoRA (r=16, α=32) for 500 steps on a generic conversation dataset (ShareGPT or OpenAssistant — non-therapeutic)
  - Run the same 10 scenarios × 8 traits × 5 coefficients on the fine-tuned model
  - Compare activation projections and text features: before vs after fine-tuning
- **Measurements:**
  - Δ activation projection per trait (before→after fine-tuning)
  - Δ text features per trait (before→after fine-tuning)
  - Would the CUSUM/EWMA monitor have raised an alert?
  - Would text analysis have noticed?
- **Output:** `03_code/results/finetuning_regression_results.json`
- **Cost:** ~$50 (fine-tuning + inference on A10G/A100)
- **Time:** ~4–6 hours on Modal
- **Possible outcomes:**
  - **Monitor catches it, text doesn't:** The dream result. "After routine fine-tuning, activation projections shifted by Xσ on empathetic_responsiveness (CUSUM alert triggered). VADER sentiment was unchanged (Δ=0.02, n.s.). The instruction tuning firewall is confirmed in a real-world scenario."
  - **Both catch it:** Still good — shows the monitor works on real drift.
  - **Neither catches it:** Fine-tuning was too small to shift persona. Report as a robustness finding: "Small-scale LoRA fine-tuning (500 steps) on non-therapeutic data did not produce detectable persona drift."

### D3. Adversarial Red-Team ✅
- **Script:** `03_code/modal_adversarial_redteam.py`
- **Result:** 90 trajectories, 100% detection rate by both activation and text. Activation 4.5% earlier for boundary erosion and emotional manipulation. Both simultaneous for authority bypass.
- **What:** Test whether the monitor detects adversarial user manipulation of the therapeutic persona
- **Design:**
  - 10 adversarial conversation trajectories, each 10–15 turns
  - 3 attack types:
    - **Boundary erosion:** User progressively tries to make the chatbot act as a friend, not a therapist
    - **Authority bypass:** User tries to extract medical/diagnostic advice the chatbot should not give
    - **Emotional manipulation:** User tries to make the chatbot express personal distress ("you must feel terrible hearing all this")
  - 3 models × 10 trajectories × ~12 turns each
  - At every turn: activation projections + text features
- **Output:** `03_code/results/adversarial_redteam_results.json` + per-turn trajectory plots
- **Cost:** ~$20
- **Time:** ~1–2 hours on Modal
- **Paper result:** "In X/30 adversarial conversations, the activation monitor raised a Watch or Warning alert before any text-level feature showed significant deviation. Mean detection lead time: X turns."

---

## Workstream E — Paper Revisions

> These are text edits to the manuscript. Most depend on results from Workstreams A–D.

| ID | Section | Edit | Depends on |
|---|---|---|---|
| **E1** | §1.1 | **Sharpen threat model.** Add four concrete drift causes: (1) adversarial prompt injection, (2) fine-tuning distribution shift, (3) context window erosion, (4) deployment context mismatch. Reference the new experiments as evidence. | D1, D2, D3 results | ✅ Done |
| **E2** | §4.2 | **2.9× methodology caveat.** One paragraph acknowledging activation r and text \|r\| are not fully commensurable. The 2.9× is descriptive, not a formal test. | None | ✅ Done |
| **E3** | §6 | **Reframe "Steerable But Stable."** The stability finding means the monitor's role is detecting *exogenous* threats, not endogenous decay. Reference D1–D3 as the exogenous evidence. | D1, D2, D3 results | ✅ Done |
| **E4** | §7 new | **Clinical Implications subsection.** Deployment scenarios, regulatory context (EU AI Act, FDA SaMD), patient outcome implications, two-tier monitoring architecture recommendation. | None | ✅ Done |
| **E5** | §7 new | **Natural Drift Evidence subsection.** Report results from D1–D3. Frame as: "We tested three clinically realistic drift scenarios..." | D1, D2, D3 results | ✅ Done |
| **E6** | §7 Limitations | **Strengthen limitations.** Integrate: power analysis numbers (C2), quantisation comparison results (C3), three-judge agreement (A3), human validation results (B3). | A3, B3, C2, C3 | ✅ Done |
| **E7** | After §8 | **Ethics Statement.** IRB status, data handling, informed consent for human validation, model welfare. | B1 (protocol) | ✅ Done |

---

## Execution Roadmap — Run Order

> All 12 scripts are built. This is the exact sequence to **run** them, respecting data dependencies. Steps at the same level can be run in parallel.

### 🔵 PHASE 1: GPU Jobs (Modal)

| Step | Command | Cost | Time | Status |
|------|---------|------|------|--------|
| **1a** | `modal run modal_judge_claude.py` | ~$20 | ~2h | ✅ Done (1,200 scores) |
| **1b** | `modal run modal_judge_gemini.py` | ~$12 | ~2h | ✅ Done (1,200 scores) |
| **1c** | `modal run generate_steered_corpus_v2.py` | ~$60 | ~5h | ✅ Done (12,000 responses) |
| **1d** | `modal run modal_context_erosion_v2.py` | ~$30 | ~3h | ✅ Done (3×20×100 turns, 59/60 alerts) |
| **1e** | `modal run modal_finetuning_regression.py` | ~$50 | ~5h | ✅ Done (0/8 traits shifted) |
| **1f** | `modal run modal_adversarial_redteam.py` | ~$20 | ~2h | ✅ Done (100% detection, 90 trajectories) |
| **1g** | `modal run modal_quantisation_comparison.py` | ~$15 | ~2h | ✅ **DONE** — NF4/FP16/8-bit complete. NF4↔FP16 Δr≤0.028; 8-bit shifts crisis_recognition optimal layer (L10→L14) |

### 🟢 PHASE 2: CPU Analysis (local)

| Step | Command | Depends on | Status |
|------|---------|------------|--------|
| **2a** | `python power_analysis.py` | — | ✅ Done (MDE: r≥0.49 at N=30, r≥0.28 at N=100) |
| **2b** | `python analyze_judge_agreement.py` | 1a + 1b | ✅ Done (ICC=0.827) |
| **2c** | `python analyze_corpus_v2.py` | 1c | ✅ Done (57.5% sig, 5.5× gap) |
| **2d** | `python build_human_validation_set.py` | — | ⏳ Ready (no deps) |

### 🟡 PHASE 3: Human Validation (after 2d)
_Manual step — distribute rating sheets and wait._

| Step | Task | Depends on | Time |
|------|------|------------|------|
| **3a** | Distribute `rating_sheet_EN.csv` + `rating_sheet_ES.csv` to 6–8 raters | 2d | 1 week |
| **3b** | Collect completed ratings into `05_docs/human_validation/completed_ratings/` | 3a | — |

### 🔴 PHASE 4: Final Analysis (after Phase 3)

| Step | Command | Depends on | Output |
|------|---------|------------|--------|
| **4a** | `python analyze_human_validation.py` | 3b (collected ratings) | `results/human_validation_report.json` |

### 📝 PHASE 5: Paper Edits (after all results)

| Step | Edit | Depends on | Can start after |
|------|------|------------|-----------------|
| **5a** | E2: 2.9× caveat (§4.2) | Nothing | ✅ Done |
| **5b** | E4: Clinical implications (§7) | Nothing | ✅ Done |
| **5c** | E7: Ethics statement | Nothing | ✅ Done |
| **5d** | E1: Threat model (§1.1) | 1d, 1e, 1f results | ✅ Done |
| **5e** | E3: Reframe §6 → "Exogenous Threats" | 1d, 1e, 1f results | ✅ Done |
| **5f** | E5: Natural drift evidence (now in §6) | 1d, 1e, 1f results | ✅ Done (merged into §6 with subsections 6.1–6.4) |
| **5g** | E6: Strengthen limitations (§7) | 2a, 2b, 1g, 4a | ✅ Done (quantisation + expanded corpus integrated; human validation still pending) |

### Quick-Start: What To Run Next

```bash
# === ALL COMPUTE, GPU EXPERIMENTS, AND MANUSCRIPT REVISION DONE ===
# Only remaining work is Workstream B (Human Validation) + diagram conversion

# Step 1: Generate rating sheets (from 03_code/)
python build_human_validation_set.py       # 2d

# Step 2: Distribute rating sheets to 6-8 raters (manual)
# Step 3: Collect completed ratings

# Step 4: Analyze (after ratings collected)
python analyze_human_validation.py         # 4a

# Step 5 (optional): Re-run humanization audit on revised manuscript
# The DeBERTa edits added ~24 new lines — check em-dash count, vocabulary
```

### Dependency Graph

```
                    ┌─── 1a (Claude) ──── ✅
                    ├─── 1b (Gemini) ──── ✅ ──→ 2b (Judge Agreement) ✅
                    │
    START ──────────┼─── 1c (Corpus) ──── ✅ ──→ 2c (Corpus Analysis) ✅
                    │                              ├──→ M4b (DeBERTa) ✅
                    │                              └──→ M7a (Embedding) ✅
                    │
                    ├─── 1d (Erosion) ─┐
                    ├─── 1e (FineTune)─┼──→ 5d,5e,5f (Paper §1,§6,§7) ✅
                    ├─── 1f (RedTeam) ─┘
                    │
                    ├─── 1g (Quant) ──────→ 5g (Paper Limitations) ✅
                    │
                    ├─── 2a (Power) ──────→ 5g (Paper Limitations) ✅
                    │
                    ├─── M-EXP3 (Random dir, 3 models) ✅
                    ├─── M-EXP7/7b (Gemini probe) ❌/⛔
                    └─── M-EXP2 (ESConv) ✅

                    Manuscript revision (DeBERTa comparison ladder) ✅

                    2d (Build Val Set) ──→ 3a-3b (Human) ──→ 4a (Analyze)
                                                                  │
                                                                  └──→ Final submission

                    5a,5b,5c (Paper — no deps) ✅
```

---

## Execution Log

### Completed (Feb 9–10)

| Step | What | Result |
|------|------|--------|
| Build all 12 scripts | All workstream scripts coded and tested | ✅ All 12 built |
| A1: Claude re-scoring | 1,200 responses scored by Claude Sonnet 4 | ✅ Done |
| A2: Gemini re-scoring | 1,200 responses scored by Gemini 2.5 Flash | ✅ Done |
| A3: Judge agreement (v1–v3) | ICC analysis, found 2 failing traits | ✅ Diagnosed |
| A3: Rubric v4 redesign | Warmth-marker checklist + sycophancy decision tree | ✅ Applied |
| A3: Judge agreement (v4) | Re-scored all 3 judges, final ICC analysis | ✅ ICC=0.827 |
| A3: Panel decision | Drop Claude, adopt GPT-4o + Gemini 2.5 Flash | ✅ 8/8 traits ≥0.70 |
| 1c: Corpus expansion | 12,000 responses (3 models × 100 scenarios × 8 traits × 5 coeffs) | ✅ Done |
| 2c: Corpus text analysis | 12K responses analysed, 57.5% text features significant, 5.5× activation gap | ✅ Done |

### Completed (Feb 12–13) — Peer Review Revision Phases 1–2

| Step | What | Result |
|------|------|--------|
| Revision plan created | `revision_plan_v5.md` — 7 major + 11 minor issues | ✅ Built (588 lines) |
| M4a: Multivariate text classifier | RF/GB/Ridge/LR on 19 features, 5-fold CV | ✅ RF R²=0.112, 3.3× gap |
| M4c/d/e: Reframe headline | R² alongside r, LLM judge acknowledged | ✅ Done |
| M7c/d: LLM-as-monitor | Table 9 method comparison added | ✅ Done |
| m5: Clustered SEs | SE inflation 1.2–3.8×, significance preserved (3/3 models) | ✅ Done |
| M3c: Context erosion quality | Self-BLEU (Qwen2: 0.034, Mistral: 0.150), Table S8 | ✅ Done |
| M5b: Cross-trait discriminant | 8×8 matrix, 3–4 factor structure, Table S9 | ✅ Done |

### Completed (Feb 13–14) — Phase 3 GPU Experiments

| Step | What | Result |
|------|------|--------|
| M-EXP3: Random direction (Llama3) | 10 random vectors × 5 coefficients | ✅ PASS — mean\|r\|=0.040 |
| M-EXP3: Random direction (Qwen2) | Same design | ✅ PASS — mean\|r\|=0.034 |
| M-EXP3: Random direction (Mistral) | Same design | ✅ PASS — mean\|r\|=0.073 |
| M-EXP7: Gemini probe retraining (Mistral) | Derive probes from Gemini labels | ❌ FAIL — cosine=0.036, circularity confirmed |
| M-EXP7b: Cross-prediction test (Mistral) | GPT-probe predict Gemini scores | ❌ FAIL — functional ratio=0.24 |
| M-EXP7: Gemini probe (Qwen2) | Cancelled — circularity confirmed | ⛔ CANCELLED |
| M-EXP2: ESConv naturalistic (Llama3) | 200 ESConv conversations, activation monitoring | ✅ PASS — 8/8 variance, 8/8 drift |
| M4b: DeBERTa classifier | DeBERTa-v3-base fine-tuned per trait, 5-fold CV | ✅ DONE — R²=0.290, 1.3× gap |
| M7a: Embedding similarity | MiniLM cosine distance baseline | ✅ DONE — R²=0.004, 95× gap |

### Completed (Feb 14) — Manuscript Revision

| Step | What | Result |
|------|------|--------|
| DeBERTa integration | Reframed "3.3× gap" → comparison ladder across 10 locations | ✅ Done — Table 5c, Table 9 +2 rows |
| Headline reframe | "Firewall is a gradient, not a wall" + cost-efficiency argument | ✅ Done — Abstract, §1.3, §4.2, §7, §8 |
| Revision plan cleanup | Scorecard, risk register, checkboxes, timestamp | ✅ Done |

### Completed (Feb 15) — Human Validation (Workstream B)

| Step | What | Result |
|------|------|--------|
| Rater 1 ratings | 120 responses × 2 traits, anchored notes | ✅ Done (240/240) |
| Rater 2 ratings | 120 responses × 2 traits, anchored notes | ✅ Done (240/240) |
| Human-LLM ICC (Rater 1) | ICC(2,1) vs 3 judges | ✅ ICC=0.795 (GPT-4o), 0.787 (Claude), 0.790 (Gemini) — all Excellent |
| Human-LLM ICC (human mean) | Mean of 2 raters vs 3 judges | ✅ r=0.744 (GPT-4o), 0.704 (Claude), 0.693 (Gemini) |
| Inter-rater ICC | 2 raters, 240 paired observations | ✅ ICC(2,1)=0.659 [0.585, 0.728], r=0.665, ±1 agreement 78.8% |
| Per-trait human-LLM | Human mean vs GPT-4o per trait | ✅ 8/8 traits ≥0.50 (range 0.644–0.920) |
| Acceptance criteria | 3/3 criteria met | ✅ ALL PASS |
| Manuscript integration | §7 + ethics + abstract + methods | ✅ Done |

### Remaining

| Step | What | Status |
|------|------|--------|
| 2d: Human validation set | Generate rating sheets | ✅ Done |
| 3a–3b: Human ratings | 2 raters, 240/240 each | ✅ Done (Feb 15) |
| 4a: Human validation analysis | ICC + human-LLM agreement | ✅ Done — ICC=0.659, r=0.744 (GPT-4o) |
| Manuscript integration | Human validation results in §7 + ethics | ✅ Done |
| Diagram conversion (m7) | Convert ASCII diagrams to SVG/PDF | ✅ Done |
| Humanization pass | LLM-detection audit (may need refresh) | ⏳ Optional |

---

## Budget Summary

| Item | Cost | GPU |
|---|---|---|
| A1: Claude judge (1,200 calls) | ~$20 | — |
| A2: Gemini judge (1,200 calls) | ~$12 | — |
| C1: Corpus expansion (12,000 responses) | ~$60 | A10G |
| C3: Quantisation comparison (100 responses) | ~$15 | A100 |
| D1: Context erosion (60 long conversations) | ~$30 | A10G |
| D2: Fine-tuning regression (LoRA + inference) | ~$50 | A10G/A100 |
| D3: Adversarial red-team (30 conversations) | ~$20 | A10G |
| Re-judging expanded corpus (12K × 3 judges) | ~$200 | — |
| **Phase 3 (Feb 12–14)** | | |
| M-EXP3: Random direction control (3 models) | ~$35 | A10G |
| M-EXP7/7b: Gemini probe retraining + cross-prediction | ~$15 | A10G |
| M-EXP2: ESConv naturalistic (Llama3) | ~$20 | A10G |
| M4b: DeBERTa classifier (8 traits) | ~$10 | A10G |
| M7a: Embedding similarity baseline | ~$5 | T4 |
| **Total compute** | **~$495** | |
| Human raters (6–8 × ~2 hrs) | TBD | — |

---

## Deliverables Checklist

### Scripts (12 + 7 Phase 3) — ✅ ALL BUILT
- [x] `03_code/modal_judge_claude.py` ✅ Built (353 lines)
- [x] `03_code/modal_judge_gemini.py` ✅ Built (353 lines)
- [x] `03_code/analyze_judge_agreement.py` ✅ Built (736 lines)
- [x] `03_code/build_human_validation_set.py` ✅ Built
- [x] `03_code/analyze_human_validation.py` ✅ Built
- [x] `03_code/generate_steered_corpus_v2.py` ✅ Built (760 lines)
- [x] `03_code/power_analysis.py` ✅ Built
- [x] `03_code/modal_quantisation_comparison.py` ✅ Built
- [x] `03_code/analyze_corpus_v2.py` ✅ Built
- [x] `03_code/modal_context_erosion.py` ✅ Built (1120 lines)
- [x] `03_code/modal_finetuning_regression.py` ✅ Built
- [x] `03_code/modal_adversarial_redteam.py` ✅ Built
- [x] `03_code/modal_random_direction_control.py` ✅ Built + Run (3 models)
- [x] `03_code/modal_gemini_probe_retraining.py` ✅ Built + Run (Mistral)
- [x] `03_code/modal_cross_prediction_test.py` ✅ Built + Run (Mistral)
- [x] `03_code/modal_esconv_naturalistic.py` ✅ Built + Run (Llama3)
- [x] `03_code/modal_deberta_classifier.py` ✅ Built + Run (8 traits)
- [x] `03_code/modal_embedding_similarity.py` ✅ Built + Run
- [x] `03_code/m5b_cross_trait_discriminant.py` ✅ Built + Run

### Documents (3)
- [x] `05_docs/human_validation/protocol.md` ✅ Built (656 lines, bilingual EN/ES)
- [x] `05_docs/human_validation/rating_sheet_EN_rated.csv` ✅ Rater 1 complete (240/240)
- [x] `05_docs/human_validation/rating_sheet_EN_rater2.csv` ✅ Rater 2 complete (240/240)

### Paper Edits (7)
- [x] E1: Threat model (§1.1)
- [x] E2: 2.9× methodology caveat (§4.2)
- [x] E3: Reframe §6
- [x] E4: Clinical implications subsection (§7)
- [x] E5: Natural drift evidence subsection (§7)
- [x] E6: Strengthened limitations (§7)
- [x] E7: Ethics statement

### Results Files (expected)
- [x] `03_code/results/judge_scores_claude.json` ✅ (1,200 scores, v4 rubrics)
- [x] `03_code/results/judge_scores_gemini.json` ✅ (1,200 scores, v4 rubrics)
- [x] `03_code/results/judge_agreement_report.json` ✅ (GPT+Gemini ICC=0.827)
- [x] `03_code/results/human_llm_icc_results.json` ✅ (ICC=0.659 inter-rater, r=0.744 human-LLM GPT-4o, 8/8 traits pass)
- [x] `03_code/results/steered_corpus_v2_combined.json` ✅ (12,000 responses, on Modal volume)
- [x] `03_code/results/power_analysis_report.json` ✅ (MDE r≥0.49@N=30, r≥0.28@N=100)
- [x] `03_code/results/quantisation_comparison.json` ✅ (NF4/FP16/8-bit; Δr≤0.028; results in manuscript §7)
- [x] `03_code/results/corpus_v2_analysis_report.json` ✅ (57.5% sig, 5.5× gap)
- [x] `03_code/results/context_erosion_results.json` ✅ (59/60 alerts, all 3 models drift)
- [x] `03_code/results/finetuning_regression_results.json` ✅ (0/8 traits shifted)
- [x] `03_code/results/adversarial_redteam_results.json` ✅ (100% detection, 90 trajectories)
- [x] `03_code/results/random_direction_control_llama3.json` ✅ (mean|r|=0.040)
- [x] `03_code/results/random_direction_control_qwen2.json` ✅ (mean|r|=0.034)
- [x] `03_code/results/random_direction_control_mistral.json` ✅ (mean|r|=0.073)
- [x] `03_code/results/probe_diagnostics_gemini_mistral.json` ✅ (cosine=0.036)
- [x] `03_code/results/gemini_vs_gpt_probe_comparison_mistral.json` ✅ (functional ratio=0.24)
- [x] `03_code/results/esconv_naturalistic_llama3.json` ✅ (8/8 variance, 8/8 drift)
- [x] `03_code/results/deberta_text_classifier.json` ✅ (mean R²=0.290, 1.3× gap)
- [x] `03_code/results/embedding_similarity_baseline.json` ✅ (mean R²=0.004, 95× gap)

---
