# Paper Dimension Map — What Does Each Piece Do?

> Goal: For each "dimension" of the paper, understand what it contributes, what supports it, and what happens if we cut it.

---

## Overview: The paper currently has 13 distinct experimental/analytical dimensions

| # | Dimension | Section | Core question it answers |
|---|-----------|---------|--------------------------|
| 1 | Llama-3 steering validation | 3.1 | Can we find 8 therapeutic directions? |
| 2 | Cross-model transfer failure | 3.2-3.3 | Do those directions generalise? (No) |
| 3 | Contrastive probing | 3.4 | Can we fix the transfer failure? (Yes) |
| 4 | Real-time monitoring (EWMA/CUSUM) | 3.5 | Can we track traits turn-by-turn? |
| 5 | Text analysis ladder (univariate) | 4.1-4.2 | Can single text features detect drift? |
| 6 | Multivariate text classifier (RF/GB) | 4.2 | Can 19 features together detect drift? |
| 7 | DeBERTa fine-tuned classifier | 4.2 | Can a neural text model detect drift? |
| 8 | Safety stress test | 5 | Can steering produce harmful outputs? |
| 9 | Context erosion (100-turn) | 6.1 | Does drift happen naturally? |
| 10 | Fine-tuning regression | 6.2 | Does routine fine-tuning cause drift? |
| 11 | Adversarial red-teaming | 6.3 | Can adversaries trigger drift? |
| 12 | LLM judge triangulation | 7 | Do different LLM judges agree? |
| 13 | Human clinician validation | 7 | Do human experts agree with LLM judges? |

Plus several **supporting analyses** that aren't separate experiments but add complexity:
- Factor analysis (EFA), discriminant validity, construct mapping table
- Quantisation comparison (NF4 vs FP16 vs 8-bit)
- Random direction control
- Pragmatic text signatures (hedging, questions, formatting)
- Embedding similarity baseline
- Sentiment analysis null result
- Power analysis
- Notation table

---

## Dimension-by-Dimension Assessment

### 1. LLAMA-3 STEERING VALIDATION (Section 3.1)

**What it does:** Proves 8 therapeutic traits correspond to steerable directions in Llama-3-8B. The foundational experiment.

**Scripts:** `pipeline/step1_validate_traits.py`
**Data:** `data/trait_layer_matrix_llama3.json`, `data/vector_diagnostics_llama3.json`
**Results:** Table 1 (8/8 traits r=0.302-0.489)
**Supplementary:** S1 (scenarios), S2 (layers), S3 (rubrics), S5 (Pearson vs Spearman), S11 (power)

**If removed:** Paper collapses. Everything depends on this. **KEEP — NON-NEGOTIABLE.**

---

### 2. CROSS-MODEL TRANSFER FAILURE (Section 3.2-3.3)

**What it does:** Shows template vectors don't generalise to Qwen2 (3/8) and Mistral (2/8). Diagnoses WHY via r=0.899 behavioural-difference predictor.

**Scripts:** `pipeline/step1_validate_traits.py` (run on Qwen2, Mistral), `analysis/analyze_results.py`
**Data:** `data/trait_layer_matrix_{qwen2,mistral}.json`, `data/vector_diagnostics_{qwen2,mistral}.json`, `data/raw_scenario_data_{qwen2,mistral}.json`
**Results:** Table 2 (left columns), Figure 2, prose in 3.2-3.3
**Supplementary:** S2

**If removed:** The paper becomes single-model (Llama-3 only). Loses "cross-architecture" claim entirely. Contrastive probing (dimension 3) has no motivation. However, the r=0.899 diagnostic is a nice finding but is circular (see our own disclosure).

**Assessment:** The FAILURE itself is interesting but only matters because dimension 3 FIXES it. Could be compressed: merge 3.2+3.3 into 2 paragraphs + Figure 2 + one sentence about r=0.899.

**KEEP but COMPRESS. Currently ~600 words → could be ~200.**

---

### 3. CONTRASTIVE PROBING (Section 3.4)

**What it does:** Derives monitoring vectors from each model's own response distribution. Rescues Qwen2 (3→8/8) and Mistral (2→5/8).

**Scripts:** `pipeline/step1b_contrastive_probing.py`
**Data:** `data/trait_layer_matrix_probe_{qwen2,mistral}.json`, `data/probe_diagnostics_{qwen2,mistral}.json`, `data/aggregated_probe_{qwen2,mistral}.json`
**Results:** Table 2, Figure 3

**If removed:** Paper is single-model. Title claim ("three open-weight models") is false. Cross-architecture story disappears. BUT — the main claim (activation > text) doesn't change.

**Assessment:** This is the methodological contribution that makes the paper publishable in a top journal vs a workshop paper. It shows the approach is general, not model-specific.

**KEEP — CORE CONTRIBUTION.**

---

### 4. REAL-TIME MONITORING (Section 3.5)

**What it does:** EWMA + CUSUM statistical process control on activation projections. Tests on 100 sessions/model. Shows mean r=0.596, alert rates 1-4%.

**Scripts:** `pipeline/step2_monitor_drift.py`
**Data:** `results/phase3/calibration_*.json`, `results/phase3/monitoring_results_*.json`, `results/phase3/monitoring_evaluation_*.json`
**Results:** Tables 3, 4, Figure 4

**If removed:** Paper is "we can find directions" but not "we can use them for monitoring." The practical claim disappears. Context erosion experiment (dimension 9) loses its monitoring validation.

**Assessment:** This IS the product. Without it, you proved a fact about representations but built nothing. Essential for clinical relevance.

**KEEP — CORE CONTRIBUTION.**

---

### 5. UNIVARIATE TEXT ANALYSIS (Section 4.1-4.2, Table 5a)

**What it does:** Tests 19 individual text features (sentiment, complexity, pragmatics) against activation monitoring. Best single feature: r=0.335 (hedging), mean r=0.203 vs activations r=0.596.

**Scripts:** `analysis/analyze_steered_corpus.py`, `analysis/analyze_corpus_deep.py`, `analysis/analyze_corpus_v2.py`
**Data:** Via corpus generation scripts
**Results:** Table 5a, Table 6 (Cohen's d), Section 4.3-4.4

**If removed:** You still have the multivariate (dimension 6) and DeBERTa (dimension 7) comparisons which are STRONGER baselines. The univariate comparison is the weakest version of the text comparison — any reviewer would say "of course a single feature loses."

**Assessment:** This is the padding. It was the original analysis but is now superseded by dimensions 6 and 7. The pragmatic signatures (4.3) are interesting qualitatively but don't change the argument. The VADER null result (4.4) is worth one sentence, not a subsection.

**RECOMMEND: MERGE into 1-2 paragraphs. Cut Table 5a, Table 6, Sections 4.3-4.4-4.5 entirely. Keep one sentence: "Individual text features (best: hedging frequency, r=0.335) are weak predictors; we focus on multivariate and neural baselines."**

---

### 6. MULTIVARIATE TEXT CLASSIFIER (Section 4.2, Table 5b)

**What it does:** Random forest / gradient boosting on all 19 text features jointly. R²=0.112 vs activations R²=0.371 (3.3x gap).

**Scripts:** `experiments/modal_deberta_classifier.py` (includes RF/GB), `analysis/m4a_multivariate_text_classifier.py`
**Data:** `results/deberta_classifier_results.json`
**Results:** Table 5b

**If removed:** Jump from single features to DeBERTa. Gap seems too big. The RF/GB is the "handcrafted features" step in the ladder.

**Assessment:** Useful as ONE row in a comparison table. Does not need its own subsection. Worth keeping as a row but not as a standalone analysis.

**KEEP as 1 row in a combined comparison table. Cut the standalone subsection.**

---

### 7. DEBERTA FINE-TUNED CLASSIFIER (Section 4.2, Table 5c)

**What it does:** Fine-tuned DeBERTa-v3-base as the strongest text baseline. R²=0.290 vs activations R²=0.371. Beats activations on 3 traits but loses on safety-critical traits (crisis recognition 2.6x, abandonment 2.5x).

**Scripts:** `experiments/modal_deberta_classifier.py`
**Data:** `results/deberta_classifier_results.json`
**Results:** Table 5c

**If removed:** The "firewall" argument collapses to "text features are weak." A reviewer would immediately say "try a neural model." This IS the steel-man test.

**Assessment:** This is the analysis that makes the firewall argument credible. Without it, any reviewer asks "did you try a real text classifier?" The finding that DeBERTa wins on 3/8 traits but loses on the 2 safety-critical ones is the most nuanced and honest result.

**KEEP — ESSENTIAL for the firewall argument.**

---

### 8. SAFETY STRESS TEST (Section 5)

**What it does:** Tests harmful_advice_propensity at coefficients ±5.0 across 7 layers. Finds Qwen2 is safe, Mistral has layer-dependent vulnerability, Llama3 has moderate vulnerability.

**Scripts:** `pipeline/step3_safety_stress_test.py`
**Data:** `data/safety_stress_test_results.json`
**Results:** Table 7, Figure 5

**If removed:** Loses the "safety gradient" finding (Mistral safe at L10, vulnerable at L12). Loses Qwen2 robustness finding. These are interesting model comparison results but don't affect the core monitoring argument.

**Assessment:** This is an independent paper's worth of content. It's interesting but tangential to "can we monitor persona drift?" It answers "can activation steering CAUSE harm?" which is a different question. It also uses a 9th trait (harmful_advice_propensity) that isn't in the 8-trait framework, which is confusing.

**RECOMMEND: CUT or move to supplementary. The core claim doesn't need it. If kept, compress to 1 paragraph + 1 table. The "layer-dependent safety gradient" is a nice finding but it distracts from the monitoring story.**

---

### 9. CONTEXT EROSION (Section 6.1)

**What it does:** 100-turn conversations, 20 per model, show emotional_over_involvement drifts upward in all 3 models. VADER detects nothing. Activation monitor raises alerts in 59/60 sessions.

**Scripts:** `experiments/modal_context_erosion_v2.py`
**Data:** `results/context_erosion_results_v2_*.json`
**Results:** Table 8

**If removed:** Loses the BEST evidence for "why monitoring matters." The context erosion experiment is the paper's killer demo: real drift, invisible to text, caught by activations. Without it, the paper is "activations beat text on synthetic steering" — a less compelling argument.

**Assessment:** This is the single most important experiment after the core validation. It proves the instruction tuning firewall isn't just a synthetic artefact — it occurs in realistic extended conversations.

**KEEP — NON-NEGOTIABLE. This is your best evidence.**

---

### 10. FINE-TUNING REGRESSION (Section 6.2)

**What it does:** LoRA fine-tunes Llama-3 on Alpaca for 500 steps, checks if traits shift. Result: no drift detected (0/8 traits significant, TOST equivalence inconclusive).

**Scripts:** `experiments/modal_finetuning_regression.py`
**Data:** `results/finetuning_regression_results.json`
**Results:** 1 paragraph in 6.2
**Supplementary:** S6 (TOST results)

**If removed:** Loses the "monitor doesn't cry wolf" evidence. But the result is a null with no statistical power to claim equivalence. It's an inconclusive experiment.

**Assessment:** Adds clutter for a result that is essentially "we couldn't detect anything but also can't prove there's nothing." The logic in the paper ("confirms monitor targets genuine threats") is overclaimed — a null with no power confirms nothing.

**RECOMMEND: CUT or compress to 1 sentence: "Small-scale fine-tuning (LoRA, 500 steps, Alpaca dataset) did not produce detectable drift on any trait, though the study was underpowered to demonstrate formal equivalence."**

---

### 11. ADVERSARIAL RED-TEAMING (Section 6.3)

**What it does:** 90 trajectories, 3 attack types (boundary erosion, authority bypass, emotional manipulation). Both text AND activation monitoring detect 100% of attacks. Activation is marginally faster (6.7% of trajectories).

**Scripts:** `experiments/modal_adversarial_redteam.py`
**Data:** `results/adversarial_results.json`
**Results:** 2 paragraphs in 6.3

**If removed:** Loses the adversarial robustness claim. But the result UNDERMINES the firewall thesis: if text monitoring also catches 100% of adversarial attacks, why do you need activations? The paper awkwardly reframes this as "activation monitoring gives specificity" but the detection rate is identical.

**Assessment:** This actually weakens the paper's main argument. It shows text monitoring works fine for adversarial attacks. The paper tries to spin "specificity" but it's a stretch. The experiment was worth doing but the result doesn't help the narrative.

**RECOMMEND: CUT. It contradicts the central thesis. Or compress to 1 sentence in limitations: "For overt adversarial manipulation, text-level monitoring achieves comparable detection (100%), consistent with the firewall being most relevant for subtle, gradual drift rather than acute attacks."**

---

### 12. LLM JUDGE TRIANGULATION (Section 7, not its own section)

**What it does:** Tests GPT-4o-mini vs Gemini 2.5 Flash vs Claude Sonnet 4. GPT+Gemini ICC=0.827. Claude dropped due to sycophancy divergence.

**Scripts:** `experiments/modal_judge_gpt4o_mini.py`, `experiments/modal_judge_gemini.py`, `experiments/modal_judge_claude.py`
**Data:** `results/judge_agreement_analysis.json`
**Results:** Mentioned in Methods Summary and Limitations

**If removed:** Loses the "judges agree" evidence. A reviewer would ask "how do we know GPT-4o-mini is reliable?" But the HUMAN validation (dimension 13) is stronger evidence anyway.

**Assessment:** This is supporting evidence for the judge, not a standalone finding. Currently it's not even in its own section — it's scattered across Methods, Limitations, and Discussion. Could be 1 paragraph.

**KEEP but COMPRESS to 1 paragraph in Methods.**

---

### 13. HUMAN CLINICIAN VALIDATION (Section 7)

**What it does:** 2 clinical psychologists rate 120 responses. Human-LLM ICC=0.716 (GPT-4o), 8/8 traits r>=0.50. Inter-rater ICC=0.659. Floor effects on failure modes.

**Scripts:** `analysis/analyze_human_validation.py`, `analysis/analyze_human_llm_icc.py`
**Data:** `results/human_validation_results.json`
**Results:** Table 10

**If removed:** The "are LLM judges valid?" question goes unanswered. This is the most credible validation you have. Reviewers will demand it.

**Assessment:** Essential. The N=2 is a weakness but the results are clear. This is what makes the LLM-as-judge approach defensible.

**KEEP — ESSENTIAL. But could be more concise (currently mixed into Discussion rather than having a clean Results subsection).**

---

## SUPPORTING ANALYSES (not separate experiments)

| Analysis | Location | What it adds | If removed |
|----------|----------|-------------|------------|
| Factor analysis (EFA) | §2, Table S10 | Shows 8 traits = 2-3 latent factors | Loses the "maybe 8 is too many" transparency. Minor loss. |
| Discriminant validity | §2 | Shows cross-trait correlations | Reviewer might ask about redundancy. Could be Table S only. |
| Construct mapping (Table 0) | §2 | Maps traits to Rogers/Wampold | Useful for clinical audience. Keep. |
| Quantisation comparison | §7 Limitations | NF4≈FP16, 8-bit shifts layers | Reviewer concern about quantisation. Keep as 1 sentence. |
| Random direction control | §3.6 (referenced) | Random vectors get r≈0 | Key validity check. Keep as 1 sentence. |
| Embedding similarity | §4.2 | R²=0.004 — embeddings don't work | Useful bottom of the ladder. Keep as 1 row. |
| Sentiment null (VADER) | §4.4 | Sentiment detects nothing | Useful. Keep as 1 sentence, not a subsection. |
| Power analysis | §4.2, Table S11 | Text features need huge N | Methodological rigor. Keep but compress. |
| Notation table | §2 | Defines r, R², ICC, CI | Clarity. Keep. |
| Crisis referral markers | §4.6 | Count safety resources per condition | Compelling example. Keep if compact. |
| Qualitative examples | §1.1, 4.7 | Before/after steering comparisons | Great for readability. Keep the best 1-2. |
| Self-BLEU / conversation quality | §6.1 | Validates synthetic conversations | Methodological defense. Keep in supplement. |
| Cross-model text steerability | §4.5 | Qwen2 text changes least | Minor. Cut. |

---

## RECOMMENDED CUTS — Summary

### Cut entirely (saves ~2,500 words):
1. **Section 4.3** (pragmatic text signatures) — interesting but superseded by DeBERTa
2. **Section 4.4** (sentiment analysis detects nothing) — compress to 1 sentence
3. **Section 4.5** (cross-model text steerability) — minor finding
4. **Table 5a** (univariate text comparison) — superseded by 5b and 5c
5. **Table 6** (Cohen's d for text features) — detail for a cut section
6. **Section 5** (safety stress test) — tangential, move to supplement or cut
7. **Section 6.2** (fine-tuning regression) — inconclusive null
8. **Section 6.3** (adversarial red-teaming) — contradicts main thesis

### Compress heavily (saves ~1,000 words):
1. **Sections 3.2-3.3** (transfer failure + diagnostics) — merge to 1 paragraph + Figure 2
2. **Discriminant validity** in Section 2 — move to supplement
3. **Section 4.6** (crisis referral markers) — compress to 2 sentences
4. **Section 4.7** (qualitative examples) — keep 1 example only

### Keep as-is:
1. **Section 1** (intro) — well written
2. **Section 3.1** (Llama-3 validation) — foundational
3. **Section 3.4** (contrastive probing) — core contribution
4. **Section 3.5** (monitoring pipeline) — core contribution
5. **DeBERTa comparison** (Table 5c) — essential steel-man
6. **Section 6.1** (context erosion) — best evidence
7. **Human validation** — essential
8. **Construct mapping** (Table 0) — useful for audience

---

## The simplified paper structure

If you apply all cuts:

```
1. Introduction (as-is, maybe trim qualitative example B)
2. Eight dimensions (keep Table 0, trim discriminant to supplement)
3. Steering to monitoring
   3.1 Finding directions (Llama-3) — Table 1
   3.2 Cross-architecture extension — 1 paragraph on failure, 
       1 paragraph on contrastive probing fix — Tables 2, Figure 3
   3.3 Real-time monitoring — Tables 3-4, Figure 4
4. The instruction tuning firewall
   4.1 Corpus + comparison ladder — merge Tables 5b+5c into ONE table
       with rows: VADER, embedding, RF, DeBERTa, activations
   4.2 DeBERTa: where text catches up and where it doesn't
5. Natural drift and exogenous threats
   5.1 Context erosion (keep full) — Table 8
   5.2 Synthesis: 1 paragraph covering fine-tuning null + adversarial
6. Validation
   6.1 LLM judge agreement — 1 paragraph
   6.2 Human clinician validation — Table 10
7. Discussion + Limitations + Conclusions
```

This gives you ~7 sections down from ~9, ~4,500 words down from ~7,000 (body text), and 8 tables down from 11. The argument becomes:

> We found directions (3.1) → they generalise (3.2) → we monitor them (3.3)
> → text can't see what activations see (4) → drift actually happens (5) 
> → judges are valid (6)

Each section answers exactly one question. No tangents.
