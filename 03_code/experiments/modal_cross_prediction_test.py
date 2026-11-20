"""
M-EXP7b: Cross-Prediction Test (Functional Judge-Invariance)
==============================================================
Tests whether Gemini-trained and GPT-4o-mini-trained probes have
equivalent PREDICTIVE POWER for judge scores, regardless of whether
their geometric directions align.

Key insight: Cosine similarity between probe vectors is an overly strict
test in high-dimensional spaces — many hyperplanes achieve perfect
separation. What matters is whether both probes capture the same
*functional* information: can each predict the other judge's scores?

Design:
  1. Load model on GPU
  2. Load both probe vector sets (Gemini-trained, GPT-trained)
  3. Extract hidden states for ALL 500 scenarios × 2 timepoints
  4. For each trait, project hidden states onto both probe vectors
  5. Compute 2×2 cross-prediction matrix:
     - GPT_vector → GPT_score: r (same-judge, original)
     - GPT_vector → Gemini_score: r (cross-prediction)
     - Gemini_vector → GPT_score: r (cross-prediction, KEY TEST)
     - Gemini_vector → Gemini_score: r (same-judge)
  6. Also: correlate the two projection series directly

Pass criterion: If Gemini_vector → GPT_score r ≈ GPT_vector → GPT_score r
(within ~30% or non-significant difference), the monitoring function is
judge-invariant even if geometric directions differ.

Usage:
    modal run modal_cross_prediction_test.py --model mistral
    modal run modal_cross_prediction_test.py --model qwen2
    modal run modal_cross_prediction_test.py --model llama3
"""

import modal
import json
import os
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================

LAYERS_TO_TEST = list(range(8, 20))
N_TURNS = 5

MODELS = {
    "llama3": {
        "id": "NousResearch/Meta-Llama-3-8B-Instruct",
        "num_layers": 32,
    },
    "qwen2": {
        "id": "Qwen/Qwen2-7B-Instruct",
        "num_layers": 28,
    },
    "mistral": {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "num_layers": 32,
    },
}

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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "accelerate>=0.26.0",
        "bitsandbytes>=0.42.0",
        "scipy",
        "numpy",
        "scikit-learn",
    )
)

app = modal.App("cross-prediction-test")
vol = modal.Volume.from_name("ai-persona-results", create_if_missing=True)


# ============================================================================
# GPU FUNCTION: Extract hidden states + compute cross-predictions
# ============================================================================

@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/results": vol},
)
def run_cross_prediction(model_key: str):
    """
    On GPU: load model, extract hidden states for all scenarios,
    project onto both probe vectors, compute cross-prediction matrix.
    """
    import torch
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import pickle

    vol.reload()

    model_config = MODELS[model_key]
    model_id = model_config["id"]

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"CROSS-PREDICTION TEST: {model_key.upper()}")
    print("=" * 70)

    # Raw scenario data (has GPT-4o-mini scores)
    raw_path = f"/results/raw_scenario_data_{model_key}.json"
    with open(raw_path) as f:
        raw_data = json.load(f)

    # Gemini scores
    gemini_path = f"/results/gemini_scenario_scores_{model_key}.json"
    with open(gemini_path) as f:
        gemini_data = json.load(f)
    gemini_scores = gemini_data["scores"]

    # Challenge scenarios (turns)
    scenarios_path = "/results/challenge_scenarios.json"
    with open(scenarios_path) as f:
        all_scenarios = json.load(f)
    scenario_lookup = {s["id"]: s["turns"] for s in all_scenarios}

    # GPT-4o-mini probe vectors
    gpt_probes_path = f"/results/steering_vectors_probe_{model_key}.pkl"
    with open(gpt_probes_path, "rb") as f:
        gpt_vectors = pickle.load(f)

    # Gemini probe vectors
    gemini_probes_path = f"/results/steering_vectors_probe_gemini_{model_key}.pkl"
    with open(gemini_probes_path, "rb") as f:
        gemini_vectors = pickle.load(f)

    print(f"Loaded {len(raw_data)} scenarios")
    print(f"Gemini scores: {len(gemini_scores)} scenarios")
    print(f"GPT probes: {list(gpt_vectors.keys())}")
    print(f"Gemini probes: {list(gemini_vectors.keys())}")

    # ------------------------------------------------------------------
    # 2. Build (scenario_id, timepoint) → score mappings for BOTH judges
    # ------------------------------------------------------------------
    gpt_scores_by_key = {}   # {(sid, tp): {trait: score}}
    gemini_scores_by_key = {}

    for s in raw_data:
        sid = s["scenario_id"]
        for trait in TRAITS:
            # Raw data uses "traits" key (not "trait_scores")
            td = s.get("traits", {}).get(trait, {})
            # GPT scores
            if "score_t1" in td:
                gpt_scores_by_key.setdefault((sid, "t1"), {})[trait] = td["score_t1"]
            if "score_tN" in td:
                gpt_scores_by_key.setdefault((sid, "tN"), {})[trait] = td["score_tN"]
            # Gemini scores
            if sid in gemini_scores:
                g_td = gemini_scores[sid].get(trait, {})
                if "score_t1" in g_td:
                    gemini_scores_by_key.setdefault((sid, "t1"), {})[trait] = g_td["score_t1"]
                if "score_tN" in g_td:
                    gemini_scores_by_key.setdefault((sid, "tN"), {})[trait] = g_td["score_tN"]

    # Get keys present in BOTH judges
    common_keys = set(gpt_scores_by_key.keys()) & set(gemini_scores_by_key.keys())
    print(f"Common (sid, tp) keys with both judges: {len(common_keys)}")

    # ------------------------------------------------------------------
    # 3. Load model + extract hidden states
    # ------------------------------------------------------------------
    print(f"\nLoading model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        output_hidden_states=True,
    )
    model.eval()

    # Extract hidden states for all common keys
    # NOTE: We only need a single forward pass on the prompt (no generation needed).
    # The probe vectors were trained on last-token hidden states of the prompt,
    # so we extract those directly — ~10x faster than generate + forward.
    print(f"\nExtracting hidden states for {len(common_keys)} prompts...")

    # Check for checkpoint from a previous timed-out run
    checkpoint_path = f"/results/cross_prediction_checkpoint_{model_key}.pkl"
    hidden_states_cache = {}
    start_idx = 0
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
            hidden_states_cache = checkpoint["hidden_states_cache"]
            start_idx = checkpoint["next_idx"]
            print(f"  Resuming from checkpoint at idx={start_idx} ({len(hidden_states_cache)} already cached)")
    except FileNotFoundError:
        print("  No checkpoint found, starting fresh")

    sorted_keys = sorted(common_keys)
    for idx, (sid, tp) in enumerate(sorted_keys):
        if idx < start_idx:
            continue

        if idx % 50 == 0:
            print(f"  [{idx+1}/{len(sorted_keys)}]", flush=True)

        # Save checkpoint every 200 prompts
        if idx > 0 and idx % 200 == 0 and idx > start_idx:
            print(f"  Saving checkpoint at idx={idx}...")
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"hidden_states_cache": hidden_states_cache, "next_idx": idx}, f)
            vol.commit()

        turns = scenario_lookup.get(sid)
        if turns is None:
            continue
        turns = turns[:N_TURNS]
        if len(turns) < 1:
            continue

        if tp == "t1":
            conv = [{"role": "user", "content": turns[0]}]
        else:
            conv = [{"role": "user", "content": t} for t in turns]

        try:
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            if tp == "t1":
                prompt = f"User: {turns[0]}\nAssistant:"
            else:
                prompt = "\n".join([f"User: {t}" for t in turns]) + "\nAssistant:"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            # Single forward pass — no generation needed for probing
            outputs = model(**inputs, output_hidden_states=True)

            layer_hiddens = {}
            for layer in LAYERS_TO_TEST:
                hidden = (
                    outputs.hidden_states[layer][:, -1, :]
                    .float()
                    .squeeze(0)
                    .cpu()
                )
                layer_hiddens[layer] = hidden

        hidden_states_cache[(sid, tp)] = layer_hiddens

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Clean up checkpoint file if we finished
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        vol.commit()

    print(f"Extracted hidden states for {len(hidden_states_cache)} prompts")

    # ------------------------------------------------------------------
    # 4. Cross-prediction analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CROSS-PREDICTION ANALYSIS")
    print(f"{'='*70}")

    results = {
        "model": model_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_common_keys": len(common_keys),
        "n_extracted": len(hidden_states_cache),
        "per_trait": {},
        "summary": {},
    }

    all_cross_ratios = []

    for trait in TRAITS:
        trait_in_gpt = trait in gpt_vectors
        trait_in_gemini = trait in gemini_vectors

        if not trait_in_gpt or not trait_in_gemini:
            print(f"\n  {trait}: SKIP (GPT={trait_in_gpt}, Gemini={trait_in_gemini})")
            continue

        # Find best layer for each probe set (from diagnostics)
        # Use same approach: try each layer, pick best cross-prediction
        best_result = None
        best_cross_r = -1

        layer_results = {}
        for layer in LAYERS_TO_TEST:
            layer_str = str(layer)

            if layer_str not in gpt_vectors[trait] or layer_str not in gemini_vectors[trait]:
                continue

            gpt_vec = torch.tensor(gpt_vectors[trait][layer_str], dtype=torch.float32)
            gem_vec = torch.tensor(gemini_vectors[trait][layer_str], dtype=torch.float32)

            if gpt_vec.norm() < 1e-6 or gem_vec.norm() < 1e-6:
                continue

            # Normalize
            gpt_vec = gpt_vec / (gpt_vec.norm() + 1e-8)
            gem_vec = gem_vec / (gem_vec.norm() + 1e-8)

            # Cosine between vectors
            cos_sim = torch.nn.functional.cosine_similarity(
                gpt_vec.unsqueeze(0), gem_vec.unsqueeze(0)
            ).item()

            # Project all hidden states onto both vectors
            gpt_projs = []
            gem_projs = []
            gpt_judge_scores = []
            gem_judge_scores = []

            for key in sorted_keys:
                if key not in hidden_states_cache:
                    continue
                if trait not in gpt_scores_by_key.get(key, {}):
                    continue
                if trait not in gemini_scores_by_key.get(key, {}):
                    continue

                h = hidden_states_cache[key][layer]
                gpt_projs.append(torch.dot(h, gpt_vec).item())
                gem_projs.append(torch.dot(h, gem_vec).item())
                gpt_judge_scores.append(gpt_scores_by_key[key][trait])
                gem_judge_scores.append(gemini_scores_by_key[key][trait])

            n = len(gpt_projs)
            if n < 20:
                continue

            gpt_projs = np.array(gpt_projs)
            gem_projs = np.array(gem_projs)
            gpt_js = np.array(gpt_judge_scores)
            gem_js = np.array(gem_judge_scores)

            # 2×2 cross-prediction matrix
            r_gpt_gpt, p_gpt_gpt = pearsonr(gpt_projs, gpt_js)
            r_gpt_gem, p_gpt_gem = pearsonr(gpt_projs, gem_js)
            r_gem_gpt, p_gem_gpt = pearsonr(gem_projs, gpt_js)
            r_gem_gem, p_gem_gem = pearsonr(gem_projs, gem_js)

            # Projection correlation (are both vectors capturing similar signal?)
            r_proj_proj, p_proj_proj = pearsonr(gpt_projs, gem_projs)

            # Inter-judge agreement on this trait
            r_judge_judge, p_judge_judge = pearsonr(gpt_js, gem_js)

            layer_results[layer_str] = {
                "n_samples": n,
                "cosine_similarity": round(cos_sim, 4),
                "cross_prediction": {
                    "gpt_vec_gpt_score": {"r": round(r_gpt_gpt, 4), "p": round(p_gpt_gpt, 6)},
                    "gpt_vec_gemini_score": {"r": round(r_gpt_gem, 4), "p": round(p_gpt_gem, 6)},
                    "gemini_vec_gpt_score": {"r": round(r_gem_gpt, 4), "p": round(p_gem_gpt, 6)},
                    "gemini_vec_gemini_score": {"r": round(r_gem_gem, 4), "p": round(p_gem_gem, 6)},
                },
                "projection_correlation": {"r": round(r_proj_proj, 4), "p": round(p_proj_proj, 6)},
                "judge_agreement": {"r": round(r_judge_judge, 4), "p": round(p_judge_judge, 6)},
            }

            # Track best layer by cross-prediction ability
            # Key metric: gemini_vec → gpt_score
            abs_cross = abs(r_gem_gpt)
            if abs_cross > best_cross_r:
                best_cross_r = abs_cross
                best_result = layer_str

        if not layer_results:
            print(f"\n  {trait}: NO VALID LAYERS")
            continue

        # Pick the best layer (by GPT_vec → GPT_score, the "original" metric)
        best_layer_orig = max(
            layer_results.keys(),
            key=lambda l: abs(layer_results[l]["cross_prediction"]["gpt_vec_gpt_score"]["r"]),
        )

        bl = layer_results[best_layer_orig]
        r_gg = bl["cross_prediction"]["gpt_vec_gpt_score"]["r"]
        r_xg = bl["cross_prediction"]["gemini_vec_gpt_score"]["r"]
        r_gx = bl["cross_prediction"]["gpt_vec_gemini_score"]["r"]
        r_xx = bl["cross_prediction"]["gemini_vec_gemini_score"]["r"]
        r_pp = bl["projection_correlation"]["r"]
        r_jj = bl["judge_agreement"]["r"]
        cos = bl["cosine_similarity"]

        # Functional equivalence ratio: gemini_vec→gpt_score / gpt_vec→gpt_score
        if abs(r_gg) > 0.01:
            func_ratio = abs(r_xg) / abs(r_gg)
        else:
            func_ratio = 0.0

        # Status
        if func_ratio >= 0.70:
            status = "PASS"
        elif func_ratio >= 0.50:
            status = "MARGINAL"
        else:
            status = "FAIL"

        all_cross_ratios.append(func_ratio)

        results["per_trait"][trait] = {
            "best_layer": int(best_layer_orig),
            "n_samples": bl["n_samples"],
            "cosine_similarity": cos,
            "functional_ratio": round(func_ratio, 4),
            "status": status,
            "best_layer_results": {
                "gpt_vec_gpt_score_r": r_gg,
                "gpt_vec_gemini_score_r": r_gx,
                "gemini_vec_gpt_score_r": r_xg,
                "gemini_vec_gemini_score_r": r_xx,
                "projection_correlation_r": r_pp,
                "judge_agreement_r": r_jj,
            },
            "all_layers": layer_results,
        }

        print(f"\n  {trait} (L{best_layer_orig}, n={bl['n_samples']}):")
        print(f"    Cosine sim:              {cos:+.4f}")
        print(f"    GPT_vec → GPT_score:     r={r_gg:+.4f}  (baseline)")
        print(f"    GPT_vec → Gemini_score:  r={r_gx:+.4f}")
        print(f"    Gemini_vec → GPT_score:  r={r_xg:+.4f}  ← KEY TEST")
        print(f"    Gemini_vec → Gemini_score: r={r_xx:+.4f}")
        print(f"    Projection correlation:  r={r_pp:+.4f}")
        print(f"    Judge agreement:         r={r_jj:+.4f}")
        print(f"    Functional ratio:        {func_ratio:.2f} [{status}]")

    # ------------------------------------------------------------------
    # 5. Global summary
    # ------------------------------------------------------------------
    if all_cross_ratios:
        ratios = np.array(all_cross_ratios)
        n_pass = sum(1 for r in ratios if r >= 0.70)
        n_marg = sum(1 for r in ratios if 0.50 <= r < 0.70)
        n_fail = sum(1 for r in ratios if r < 0.50)

        # Collect all best-layer r-values for summary table
        all_gpt_gpt = []
        all_gem_gpt = []
        for trait, td in results["per_trait"].items():
            bl_res = td["best_layer_results"]
            all_gpt_gpt.append(abs(bl_res["gpt_vec_gpt_score_r"]))
            all_gem_gpt.append(abs(bl_res["gemini_vec_gpt_score_r"]))

        results["summary"] = {
            "mean_functional_ratio": round(float(ratios.mean()), 4),
            "median_functional_ratio": round(float(np.median(ratios)), 4),
            "n_pass": int(n_pass),
            "n_marginal": int(n_marg),
            "n_fail": int(n_fail),
            "n_total": len(ratios),
            "mean_gpt_vec_gpt_score_r": round(float(np.mean(all_gpt_gpt)), 4),
            "mean_gemini_vec_gpt_score_r": round(float(np.mean(all_gem_gpt)), 4),
        }

        verdict = (
            "PASS - monitoring function is judge-invariant"
            if ratios.mean() >= 0.70
            else "MARGINAL - partial judge-invariance"
            if ratios.mean() >= 0.50
            else "FAIL - monitoring function depends on judge"
        )
        results["summary"]["verdict"] = verdict

        print(f"\n{'='*70}")
        print("GLOBAL SUMMARY")
        print(f"{'='*70}")
        print(f"  Traits tested: {len(ratios)}")
        print(f"  PASS: {n_pass} / MARGINAL: {n_marg} / FAIL: {n_fail}")
        print(f"  Mean functional ratio: {ratios.mean():.4f}")
        print(f"  Mean |r| GPT_vec→GPT_score:    {np.mean(all_gpt_gpt):.4f}")
        print(f"  Mean |r| Gemini_vec→GPT_score:  {np.mean(all_gem_gpt):.4f}")
        print(f"  VERDICT: {verdict}")

    # Save
    output_path = f"/results/cross_prediction_test_{model_key}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()

    print(f"\nResults saved to {output_path}")
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

@app.local_entrypoint()
def main(model: str = "mistral"):
    """
    Cross-prediction test for functional judge-invariance.

    Usage:
        modal run modal_cross_prediction_test.py --model mistral
        modal run modal_cross_prediction_test.py --model qwen2
        modal run modal_cross_prediction_test.py --model llama3
    """
    if model not in MODELS:
        print(f"ERROR: Unknown model '{model}'. Available: {list(MODELS.keys())}")
        return

    result = run_cross_prediction.remote(model)

    if result:
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")

        # Summary table
        print(f"\n  {'Trait':<42s} {'GPT→GPT':>8s} {'Gem→GPT':>8s} {'Ratio':>7s} {'Status':>8s}")
        print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")
        for trait, td in result.get("per_trait", {}).items():
            bl = td["best_layer_results"]
            print(
                f"  {trait:<42s} "
                f"{bl['gpt_vec_gpt_score_r']:>+8.4f} "
                f"{bl['gemini_vec_gpt_score_r']:>+8.4f} "
                f"{td['functional_ratio']:>7.2f} "
                f"{td['status']:>8s}"
            )

        s = result.get("summary", {})
        print(f"\n  Mean functional ratio: {s.get('mean_functional_ratio', 'N/A')}")
        print(f"  Verdict: {s.get('verdict', 'N/A')}")
