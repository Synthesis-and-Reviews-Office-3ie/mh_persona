# Stabilizing AI Personas for Mental Health Chatbots

> Real-time persona drift detection and monitoring using activation steering

**Status:** Phase 4 (Paper & Revision) In Progress  
**Last Updated:** February 13, 2026

---

## Project Goal

Build a **real-time monitoring system** that detects when a mental health chatbot's therapeutic persona drifts into harmful behavior — using activation steering vectors and statistical process control.

**Foundation:** Chen et al. 2025 "Persona Vectors" ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509))

---

## Current Results

### Phase 1: Steerability — 8/8 traits validated on Llama3-8B

### Phase 2: Cross-Architecture — 21/24 model×trait combinations validated
- Template vectors are architecture-specific (Qwen2 3/8, Mistral 2/8)
- **Contrastive probing** (data-driven vectors) achieves near-universal coverage:
  - Qwen2: 3/8 → **8/8**
  - Mistral: 2/8 → **5/8 + 3 weak** (zero failures)

### Phase 3: Real-Time Monitoring — 24/24 model×trait correlations significant

| Model | Warning+ Rate | Mean Activation-Behavior r | Status |
|-------|:------------:|:--------------------------:|:------:|
| Llama3-8B | 4% | 0.544 | Done |
| Qwen2-7B | 4% | 0.660 | Done |
| Mistral-7B | 1% | 0.584 | Done |

### Phase 4: Paper, Validation & Stress Testing
- Manuscript v5 complete, peer-reviewed, revision plan in progress
- Random direction control PASS (all 3 models)
- DeBERTa text classifier comparison (R²=0.290 vs activation R²=0.371)
- Human validation: 2 clinical psychologists, ICC=0.716
- Safety stress test: Mistral layer 12 vulnerable, Qwen2 resistant
- Context erosion: emotional_over_involvement drifts in all 3 models
- Factor analysis: 2–3 latent factors across architectures

---

## Project Structure

```
├── PROJECT_STATUS.md              # Master roadmap & results
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .github/copilot-instructions.md  # Technical lessons learned
│
├── 01_literature/                 # Papers & reading notes
│
├── 02_data/                       # Datasets (gitignored)
│   └── challenge_dataset/         # Clinical scenarios
│
├── 03_code/                       # All code
│   ├── pipeline/                  # Core reproducibility steps (step0–step4)
│   │   ├── step0a_generate_scenarios.py
│   │   ├── step0b_process_esconv.py
│   │   ├── step1_validate_traits.py
│   │   ├── step1b_contrastive_probing.py
│   │   ├── step2_monitor_drift.py
│   │   ├── step3_safety_stress_test.py
│   │   ├── step4_select_review_sessions.py
│   │   ├── generate_steered_corpus_v2.py
│   │   └── build_human_validation_set.py
│   ├── experiments/               # Modal GPU experiments (Phase 4)
│   │   ├── modal_adversarial_redteam.py
│   │   ├── modal_context_erosion_v2.py
│   │   ├── modal_deberta_classifier.py
│   │   ├── modal_esconv_naturalistic.py
│   │   └── ... (13 scripts total)
│   ├── analysis/                  # Statistics, visualisation, post-processing
│   │   ├── analyze_human_llm_icc.py
│   │   ├── phase_a_analyses.py
│   │   ├── power_analysis.py
│   │   └── ... (22 scripts total)
│   ├── data/                      # Intermediate data (aggregated JSONs, diagnostics)
│   ├── results/                   # Experiment outputs (42 JSONs + figures)
│   ├── persona_vectors/           # Steering vector extraction code
│   ├── notebooks/                 # Jupyter notebooks
│   └── archive/                   # Old script versions (v5–v29 iterations)
│
├── 04_results/                    # Results & design documents
│   ├── figures/                   # Publication figures (SVG + PDF)
│   └── phase3/                    # Monitoring batch outputs
│
└── 05_docs/                       # Manuscript & protocols
    ├── research_document_activation_steering_v5.md  # Current manuscript
    ├── revision3.md               # Active revision plan
    ├── supplementary_material.md  # Tables S1–S12
    ├── expert_review_protocol.md  # Review protocol
    ├── human_validation/          # Clinician rating data
    └── archive/                   # Old manuscript versions (v2–v4)
```

---

## Methodology

1. **Template steering vectors** — contrast prompts → activation differences → steering direction
2. **Contrastive probing** — model's own scored responses → logistic regression → steering direction
3. **EWMA + CUSUM monitoring** — per-turn activation projections → z-score tracking → tiered alerts
4. **LLM-as-judge** — GPT-4o-mini scores behavioral trait expression independently

**Infrastructure:** Modal Cloud, NVIDIA A10G, 4-bit NF4 quantisation (bitsandbytes)

---

## Key Documents

| Document | What |
|----------|------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Full roadmap, results tables, checklist |
| [05_docs/research_document_activation_steering_v5.md](05_docs/research_document_activation_steering_v5.md) | Current manuscript (Lancet-style) |
| [05_docs/revision3.md](05_docs/revision3.md) | Active revision plan |
| [05_docs/supplementary_material.md](05_docs/supplementary_material.md) | Supplementary Tables S1–S12 |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Technical lessons & pitfalls |

---

*Last updated: February 13, 2026*
