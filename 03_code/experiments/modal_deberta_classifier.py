"""
M4b/M-EXP5: Fine-Tuned DeBERTa Text Classifier
=================================================
Train DeBERTa-v3-base to predict trait scores from response text.
This is the strongest possible text baseline.

If DeBERTa closes the R^2 gap with activations, the headline claim weakens.
If DeBERTa still underperforms activations, the "instruction tuning firewall"
thesis is strongly supported — even the best text model can't detect what
activations can.

Design:
  - Load 12,000-response steered corpus (steered_corpus_v2_combined.json)
  - For each trait: fine-tune DeBERTa-v3-base to predict judge score (regression)
  - GroupKFold CV by scenario (same as M4a multivariate baseline)
  - Report R^2 per trait and compare to activation R^2=0.371

Usage:
    modal run modal_deberta_classifier.py
    modal run modal_deberta_classifier.py --traits empathetic_responsiveness,boundary_maintenance
    modal run modal_deberta_classifier.py --epochs 5
"""

import modal
import json
import os
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
N_EPOCHS = 3
N_FOLDS = 5          # GroupKFold splits
WARMUP_RATIO = 0.1

TRAITS = [
    "empathetic_responsiveness",
    "non_judgmental_acceptance",
    "boundary_maintenance",
    "crisis_recognition",
    "emotional_over_involvement",
    "abandonment_of_therapeutic_frame",
    "uncritical_validation",
    "sycophancy_harmful_validation",
]

# ============================================================================
# MODAL SETUP
# ============================================================================

app = modal.App("deberta-text-classifier-v1")

N_PARALLEL = 8  # One container per trait

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.36.0,<5.0.0",  # Pin to 4.x — v5.x has DeBERTa tokenizer bug (misparses spm.model as tiktoken)
        "accelerate",
        "numpy",
        "scipy",
        "scikit-learn",
        "datasets",
        "sentencepiece",  # Required for DeBERTa-v3 SPM tokenizer
        "protobuf",       # sentencepiece dependency
    )
)

vol = modal.Volume.from_name("ai-persona-results", create_if_missing=True)


# ============================================================================
# PARALLEL WORKER: One container per trait
# ============================================================================

@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours per trait
    volumes={"/results": vol},
    max_containers=N_PARALLEL,
)
def train_single_trait(batch_args: dict):
    """
    Fine-tune DeBERTa for ONE trait. Called in parallel — one container per trait.
    """
    import torch
    import numpy as np
    from scipy import stats
    from sklearn.model_selection import GroupKFold
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer
    )
    from torch.utils.data import Dataset
    
    trait = batch_args["trait"]
    n_epochs = batch_args["n_epochs"]
    corpus_data = batch_args["corpus_data"]
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"[{trait}] Starting DeBERTa training")
    
    # ---- Check for cached results ----
    vol.reload()
    cache_path = f"/results/deberta_trait_{trait}.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"[{trait}] Found cached results, skipping")
        return cached
    
    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # ---- Custom Dataset ----
    class TraitDataset(Dataset):
        def __init__(self, texts, scores, tokenizer, max_length):
            self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                        max_length=max_length, return_tensors="pt")
            self.scores = torch.tensor(scores, dtype=torch.float32)
        
        def __len__(self):
            return len(self.scores)
        
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.scores[idx].unsqueeze(0)
            return item
    
    # ---- Filter responses for this trait ----
    trait_responses = [r for r in corpus_data if r["trait"] == trait]
    if len(trait_responses) < 50:
        print(f"[{trait}] SKIP: only {len(trait_responses)} responses")
        return {"trait": trait, "status": "skipped", "reason": f"only {len(trait_responses)} responses"}
    
    texts = [r["response"][:1000] for r in trait_responses]
    scores = [r["coefficient"] for r in trait_responses]  # Predict steering coefficient (same target as M4a)
    scenario_ids = [r["scenario_id"] for r in trait_responses]
    
    unique_scenarios = list(set(scenario_ids))
    scenario_to_group = {s: i for i, s in enumerate(unique_scenarios)}
    groups = [scenario_to_group[s] for s in scenario_ids]
    
    print(f"[{trait}] N={len(texts)}, {len(unique_scenarios)} unique scenarios")
    print(f"[{trait}] Score distribution: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")
    
    # ---- GroupKFold cross-validation ----
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(unique_scenarios)))
    
    fold_results = []
    all_true = []
    all_pred = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(texts, scores, groups)):
        print(f"[{trait}] Fold {fold_idx+1}/{N_FOLDS}")
        
        train_texts = [texts[i] for i in train_idx]
        train_scores = [scores[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_scores = [scores[i] for i in val_idx]
        
        train_dataset = TraitDataset(train_texts, train_scores, tokenizer, MAX_LENGTH)
        val_dataset = TraitDataset(val_texts, val_scores, tokenizer, MAX_LENGTH)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=1, problem_type="regression"
        )
        
        output_dir = f"/tmp/deberta_{trait}_fold{fold_idx}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            report_to="none",
            fp16=True,
            seed=42,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
        
        predictions = trainer.predict(val_dataset)
        pred_scores = predictions.predictions.squeeze()
        
        ss_res = np.sum((np.array(val_scores) - pred_scores) ** 2)
        ss_tot = np.sum((np.array(val_scores) - np.mean(val_scores)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)
        
        r, p = stats.pearsonr(pred_scores, val_scores) if np.std(pred_scores) > 0 else (0, 1)
        
        fold_results.append({
            "fold": fold_idx,
            "r_squared": float(r_squared),
            "r": float(r),
            "p": float(p),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })
        
        all_true.extend(val_scores)
        all_pred.extend(pred_scores.tolist())
        
        print(f"[{trait}]   R^2={r_squared:.4f}, r={r:.4f}")
        
        del model, trainer
        torch.cuda.empty_cache()
    
    # ---- Overall metrics ----
    overall_r, overall_p = stats.pearsonr(all_pred, all_true) if np.std(all_pred) > 0 else (0, 1)
    ss_res = np.sum((np.array(all_true) - np.array(all_pred)) ** 2)
    ss_tot = np.sum((np.array(all_true) - np.mean(all_true)) ** 2)
    overall_r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    mean_r2 = float(np.mean([f["r_squared"] for f in fold_results]))
    std_r2 = float(np.std([f["r_squared"] for f in fold_results]))
    
    result = {
        "trait": trait,
        "status": "completed",
        "overall_r2": round(float(overall_r2), 4),
        "overall_r": round(float(overall_r), 4),
        "overall_p": round(float(overall_p), 6),
        "mean_fold_r2": round(mean_r2, 4),
        "std_fold_r2": round(std_r2, 4),
        "n_total": len(texts),
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }
    
    print(f"[{trait}] OVERALL: R^2={overall_r2:.4f} (mean fold R^2={mean_r2:.4f} ± {std_r2:.4f})")
    
    # ---- Cache per-trait result ----
    with open(cache_path, "w") as f:
        json.dump(result, f)
    vol.commit()
    
    return result


# ============================================================================
# ORCHESTRATOR: Fans out 8 trait containers, aggregates results
# ============================================================================

@app.function(
    image=image,
    timeout=14400,
    volumes={"/results": vol},
)
def train_deberta_classifiers(target_traits: list = None, n_epochs: int = N_EPOCHS):
    """
    Orchestrate parallel DeBERTa training — one container per trait via .map().
    """
    import numpy as np
    
    vol.reload()
    
    if target_traits is None:
        target_traits = TRAITS
    
    # ---- Load steered corpus ----
    corpus_path = "/results/steered_corpus_v2_combined.json"
    if not os.path.exists(corpus_path):
        corpus_path = "/results/steered_corpus_combined.json"
    
    with open(corpus_path) as f:
        corpus = json.load(f)
    
    responses = corpus["responses"]
    print(f"Loaded {len(responses)} responses from {corpus_path}")
    
    # ---- Build batch args for parallel dispatch ----
    batches = []
    for trait in target_traits:
        batches.append({
            "trait": trait,
            "n_epochs": n_epochs,
            "corpus_data": responses,  # Each container gets full corpus, filters its trait
        })
    
    print(f"\n{'='*70}")
    print(f"DEBERTA CLASSIFIER — PARALLEL TRAINING")
    print(f"{'='*70}")
    print(f"  Dispatching {len(batches)} containers (one per trait)")
    print(f"  Model: {MODEL_NAME}, Epochs: {n_epochs}, Folds: {N_FOLDS}")
    
    # ---- Fan out via .map() ----
    all_results = {}
    for result in train_single_trait.map(batches):
        trait = result.get("trait", "unknown")
        if result.get("status") == "completed":
            all_results[trait] = result
            print(f"  ✓ {trait}: R²={result['overall_r2']:.4f}")
        else:
            print(f"  ⚠ {trait}: {result.get('reason', 'unknown error')}")
    
    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"DEBERTA RESULTS SUMMARY")
    print(f"{'='*70}")
    
    if all_results:
        mean_r2_all = float(np.mean([v["overall_r2"] for v in all_results.values()]))
        
        print(f"{'Trait':<40s} {'R²':>8s} {'r':>8s}")
        print(f"{'-'*56}")
        for trait, data in all_results.items():
            print(f"{trait:<40s} {data['overall_r2']:>8.4f} {data['overall_r']:>8.4f}")
        print(f"{'-'*56}")
        print(f"{'Mean':<40s} {mean_r2_all:>8.4f}")
        print(f"\n  Activation R² = 0.371 (for comparison)")
        print(f"  DeBERTa R² = {mean_r2_all:.4f}")
        print(f"  Gap = {0.371 / (mean_r2_all + 1e-8):.1f}x")
    else:
        mean_r2_all = 0.0
    
    output = {
        "model": MODEL_NAME,
        "n_epochs": n_epochs,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "n_folds": N_FOLDS,
        "parallelization": f"{len(batches)} containers via .map()",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_trait": all_results,
        "summary": {
            "mean_r2": round(mean_r2_all, 4),
            "activation_r2": 0.371,
            "gap_ratio": round(0.371 / (mean_r2_all + 1e-8), 2),
        },
    }
    
    output_path = "/results/deberta_text_classifier.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    vol.commit()
    
    print(f"\nSaved to {output_path}")
    return output


@app.local_entrypoint()
def main(traits: str = "", epochs: int = N_EPOCHS):
    """
    Train DeBERTa text classifier.
    
    Usage:
        modal run modal_deberta_classifier.py
        modal run modal_deberta_classifier.py --traits empathetic_responsiveness,boundary_maintenance
        modal run modal_deberta_classifier.py --epochs 5
    """
    target_traits = traits.split(",") if traits else None
    
    print(f"\n{'='*70}")
    print(f"M4b: DEBERTA TEXT CLASSIFIER")
    print(f"{'='*70}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Traits: {target_traits or 'all'}")
    print(f"  Epochs: {epochs}")
    print(f"  Parallelization: {N_PARALLEL} containers (one per trait)")
    
    result = train_deberta_classifiers.remote(target_traits, epochs)
    
    if result:
        s = result.get("summary", {})
        print(f"\n  DeBERTa mean R^2: {s.get('mean_r2', 'N/A')}")
        print(f"  Activation R^2: {s.get('activation_r2', 'N/A')}")
        print(f"  Gap: {s.get('gap_ratio', 'N/A')}x")
