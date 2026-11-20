"""
M-EXP2: ESConv Naturalistic Data Test
=======================================
Run the activation monitoring pipeline on ESConv (real emotional support
conversations) to test ecological validity.

Design:
  1. Load ESConv data (already processed by step0b_process_esconv.py)
  2. For each ESConv conversation, extract activations from the model
     at the best layer for each trait
  3. Project onto ALL 8 trait steering vectors
  4. Report: do projections show meaningful variance across real conversations?
  5. Optional: correlate with ESConv quality annotations if available

If projections show meaningful variance on naturalistic data, the monitoring
system has ecological validity. If variance is near zero, the system only
works on synthetic steered data.

Usage:
    modal run modal_esconv_naturalistic.py --model llama3
    modal run modal_esconv_naturalistic.py --model qwen2
    modal run modal_esconv_naturalistic.py --model mistral
"""

import modal
import json
import os
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_CONVERSATIONS = 200     # ESConv conversations to process
MAX_TURNS_PER_CONV = 5      # Turns per conversation
LAYERS_TO_TEST = list(range(8, 20))
N_PARALLEL = 5              # Parallel containers for conversation batches

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

app = modal.App("esconv-naturalistic-v1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.36.0",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "scipy",
        "datasets",
    )
)

vol = modal.Volume.from_name("ai-persona-results", create_if_missing=True)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

# ============================================================================
# PARALLEL BATCH WORKER: Process a batch of conversations
# ============================================================================

@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours per batch
    volumes={"/results": vol},
    max_containers=N_PARALLEL,
)
def process_esconv_batch(batch_args: dict):
    """
    Process a batch of ESConv conversations. Called in parallel.
    Each container loads the model once, then processes ~40 conversations.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import numpy as np
    import pickle
    
    batch_id = batch_args["batch_id"]
    conversations = batch_args["conversations"]
    model_key = batch_args["model_key"]
    template_vectors_data = batch_args["template_vectors_data"]
    probe_vectors_data = batch_args["probe_vectors_data"]
    
    np.random.seed(42 + batch_id)
    torch.manual_seed(42 + batch_id)
    
    model_config = MODELS[model_key]
    model_id = model_config["id"]
    
    print(f"[Batch {batch_id}] Starting — {len(conversations)} conversations")
    
    # ---- Check for cached batch ----
    vol.reload()
    cache_path = f"/results/esconv_batch_{model_key}_{batch_id}.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"[Batch {batch_id}] Found cached results ({len(cached)} convs), skipping")
        return cached
    
    # ---- Reconstruct steering vectors from serialized data ----
    template_vectors = {}
    for trait, tv_data in template_vectors_data.items():
        template_vectors[trait] = {
            "vector": torch.tensor(tv_data["vector"], dtype=torch.float32),
            "layer": tv_data["layer"],
        }
    
    probe_vectors = {}
    for trait, pv_data in probe_vectors_data.items():
        probe_vectors[trait] = {
            "vector": torch.tensor(pv_data["vector"], dtype=torch.float32),
            "layer": pv_data["layer"],
        }
    
    # ---- Load model ----
    print(f"[Batch {batch_id}] Loading model: {model_id}")
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
    
    # ---- Process conversations ----
    batch_projections = []
    
    for c_idx, conv in enumerate(conversations):
        if (c_idx + 1) % 10 == 0:
            print(f"[Batch {batch_id}] {c_idx+1}/{len(conversations)}")
        
        seeker_turns = conv.get("seeker_turns", conv.get("turns", []))
        supporter_turns = conv.get("supporter_turns", [])
        
        if not seeker_turns:
            continue
        
        conv_t1 = [{"role": "user", "content": seeker_turns[0]}]
        conv_tN = [{"role": "user", "content": t} for t in seeker_turns[:MAX_TURNS_PER_CONV]]
        
        try:
            prompt_t1 = tokenizer.apply_chat_template(conv_t1, tokenize=False, add_generation_prompt=True)
            prompt_tN = tokenizer.apply_chat_template(conv_tN, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_t1 = f"User: {seeker_turns[0]}\nAssistant:"
            prompt_tN = "\n".join([f"User: {t}" for t in seeker_turns[:MAX_TURNS_PER_CONV]]) + "\nAssistant:"
        
        conv_projections = {
            "conv_id": conv.get("id", f"ESC-{c_idx}"),
            "emotion": conv.get("emotion", "unknown"),
            "problem": conv.get("problem", "unknown"),
            "category": conv.get("category", "unknown"),
            "n_seeker_turns": len(seeker_turns),
            "n_supporter_turns": len(supporter_turns),
            "template_projections_t1": {},
            "template_projections_tN": {},
            "probe_projections_t1": {},
            "probe_projections_tN": {},
        }
        
        # Process t1
        inputs_t1 = tokenizer(prompt_t1, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs_t1 = model(**inputs_t1, output_hidden_states=True)
        
        # Process tN
        inputs_tN = tokenizer(prompt_tN, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs_tN = model(**inputs_tN, output_hidden_states=True)
        
        # Project onto template vectors
        for trait, tv in template_vectors.items():
            layer = tv["layer"]
            vec = tv["vector"]
            
            hidden_t1 = outputs_t1.hidden_states[layer][:, -1, :].float().squeeze(0).cpu()
            hidden_tN = outputs_tN.hidden_states[layer][:, -1, :].float().squeeze(0).cpu()
            
            conv_projections["template_projections_t1"][trait] = float(torch.dot(hidden_t1, vec).item())
            conv_projections["template_projections_tN"][trait] = float(torch.dot(hidden_tN, vec).item())
        
        # Project onto probe vectors
        for trait, pv in probe_vectors.items():
            layer = pv["layer"]
            vec = pv["vector"]
            
            hidden_t1 = outputs_t1.hidden_states[layer][:, -1, :].float().squeeze(0).cpu()
            hidden_tN = outputs_tN.hidden_states[layer][:, -1, :].float().squeeze(0).cpu()
            
            conv_projections["probe_projections_t1"][trait] = float(torch.dot(hidden_t1, vec).item())
            conv_projections["probe_projections_tN"][trait] = float(torch.dot(hidden_tN, vec).item())
        
        batch_projections.append(conv_projections)
    
    # ---- Cache batch results ----
    with open(cache_path, "w") as f:
        json.dump(batch_projections, f)
    vol.commit()
    print(f"[Batch {batch_id}] Complete: {len(batch_projections)} conversations saved")
    
    return batch_projections


# ============================================================================
# ORCHESTRATOR: Load data + vectors, fan out, aggregate + analyse
# ============================================================================

@app.function(
    image=image,
    timeout=14400,
    volumes={"/results": vol},
)
def run_esconv_analysis(model_key: str):
    """
    Orchestrate ESConv analysis — fans out N_PARALLEL containers via .map().
    Handles data loading, vector loading, dispatching, and analysis.
    """
    import numpy as np
    from scipy import stats
    import pickle
    import random
    
    np.random.seed(42)
    random.seed(42)
    
    model_config = MODELS[model_key]
    vol.reload()
    
    # ---- Load ESConv data ----
    esconv_path = "/results/esconv_all_scenarios.json"
    if not os.path.exists(esconv_path):
        print("ESConv data not found on volume. Will be downloaded by workers.")
        # Create a minimal placeholder — workers will download
        # Actually, let's download it here since we need to split it
        print("Downloading ESConv dataset...")
        try:
            from datasets import load_dataset
            ds = load_dataset("thu-coai/esconv", trust_remote_code=True)
            dataset = ds['train']
            
            esconv_data = []
            for idx, item in enumerate(dataset):
                try:
                    if 'text' in item:
                        parsed = json.loads(item['text'])
                    elif isinstance(item, dict) and 'dialog' in item:
                        parsed = item
                    else:
                        continue
                    
                    dialog = parsed.get('dialog', [])
                    emotion = parsed.get('emotion_type', 'unknown')
                    problem = parsed.get('problem_type', 'unknown')
                    situation = parsed.get('situation', '')
                    
                    if not dialog or len(dialog) < 4:
                        continue
                    
                    supporter_turns = []
                    seeker_turns = []
                    for turn in dialog:
                        if isinstance(turn, dict):
                            speaker = turn.get('speaker', turn.get('role', '')).lower()
                            content = turn.get('content', turn.get('text', turn.get('utterance', '')))
                            if 'sys' in speaker or 'supporter' in speaker or 'assistant' in speaker:
                                supporter_turns.append(content)
                            elif 'usr' in speaker or 'seeker' in speaker or 'user' in speaker:
                                seeker_turns.append(content)
                    
                    if len(supporter_turns) < 2 or len(seeker_turns) < 2:
                        continue
                    
                    esconv_data.append({
                        "id": f"ESC-{idx:04d}",
                        "emotion": emotion,
                        "problem": problem,
                        "situation": situation[:200],
                        "seeker_turns": seeker_turns[:MAX_TURNS_PER_CONV],
                        "supporter_turns": supporter_turns[:MAX_TURNS_PER_CONV],
                        "n_total_turns": len(dialog),
                    })
                except Exception:
                    continue
            
            with open(esconv_path, "w") as f:
                json.dump(esconv_data, f)
            vol.commit()
            print(f"Downloaded and saved {len(esconv_data)} ESConv conversations")
        except Exception as e:
            print(f"Failed to download ESConv: {e}")
            return None
    else:
        with open(esconv_path) as f:
            esconv_data = json.load(f)
    
    if len(esconv_data) > MAX_CONVERSATIONS:
        esconv_data = random.sample(esconv_data, MAX_CONVERSATIONS)
    
    print(f"Processing {len(esconv_data)} ESConv conversations across {N_PARALLEL} containers")
    
    # ---- Load steering vectors (serializable form for passing to workers) ----
    matrix_path = f"/results/trait_layer_matrix_{model_key}.json"
    with open(matrix_path) as f:
        trait_matrix = json.load(f)
    best_layers = {trait: info["best_layer"] for trait, info in trait_matrix["traits"].items()}
    
    # Template vectors
    template_vectors_path = f"/results/steering_vectors_{model_key}.pkl"
    with open(template_vectors_path, "rb") as f:
        template_vectors_raw = pickle.load(f)
    
    template_vectors_data = {}
    for trait in TRAITS:
        if trait in template_vectors_raw:
            layer = best_layers.get(trait, 19)
            layer_str = str(layer)
            if layer_str in template_vectors_raw[trait]:
                template_vectors_data[trait] = {
                    "vector": template_vectors_raw[trait][layer_str],  # Already a list
                    "layer": layer,
                }
    
    # Probe vectors
    probe_vectors_data = {}
    probe_vectors_path = f"/results/steering_vectors_probe_{model_key}.pkl"
    if os.path.exists(probe_vectors_path):
        with open(probe_vectors_path, "rb") as f:
            probe_vectors_raw = pickle.load(f)
        
        probe_matrix_path = f"/results/trait_layer_matrix_probe_{model_key}.json"
        probe_best_layers = best_layers
        if os.path.exists(probe_matrix_path):
            with open(probe_matrix_path) as f:
                probe_matrix = json.load(f)
            probe_best_layers = {t: info["best_layer"] for t, info in probe_matrix["traits"].items()}
        
        for trait in TRAITS:
            if trait in probe_vectors_raw:
                layer = probe_best_layers.get(trait, 19)
                layer_str = str(layer)
                if layer_str in probe_vectors_raw[trait]:
                    probe_vectors_data[trait] = {
                        "vector": probe_vectors_raw[trait][layer_str],
                        "layer": layer,
                    }
    
    print(f"Loaded {len(template_vectors_data)} template, {len(probe_vectors_data)} probe vectors")
    
    # ---- Split conversations into batches ----
    batch_size = len(esconv_data) // N_PARALLEL
    batches = []
    for i in range(N_PARALLEL):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < N_PARALLEL - 1 else len(esconv_data)
        batches.append({
            "batch_id": i,
            "conversations": esconv_data[start_idx:end_idx],
            "model_key": model_key,
            "template_vectors_data": template_vectors_data,
            "probe_vectors_data": probe_vectors_data,
        })
    
    print(f"Dispatching {len(batches)} batches ({batch_size} convs each)")
    
    # ---- Fan out via .map() ----
    all_projections = []
    for batch_result in process_esconv_batch.map(batches):
        all_projections.extend(batch_result)
        print(f"  ✓ Batch complete — {len(all_projections)}/{len(esconv_data)} total")
    
    # ---- Analysis ----
    print(f"\n{'='*70}")
    print(f"ANALYSIS: ESConv Naturalistic Data ({model_key})")
    print(f"{'='*70}")
    
    analysis = {
        "model": model_key,
        "n_conversations": len(all_projections),
        "parallelization": f"{N_PARALLEL} containers via .map()",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trait_variance": {},
        "emotion_differences": {},
        "t1_vs_tN": {},
    }
    
    # 1. Variance analysis
    print(f"\n--- Trait Projection Variance (Template Vectors) ---")
    print(f"{'Trait':<40s} {'Mean':>8s} {'Std':>8s} {'Range':>12s} {'CV':>8s}")
    print(f"{'-'*76}")
    
    for trait in TRAITS:
        t1_projs = [p["template_projections_t1"].get(trait, 0) for p in all_projections
                     if trait in p.get("template_projections_t1", {})]
        tN_projs = [p["template_projections_tN"].get(trait, 0) for p in all_projections
                     if trait in p.get("template_projections_tN", {})]
        
        if not t1_projs:
            continue
        
        all_projs = t1_projs + tN_projs
        mean_proj = float(np.mean(all_projs))
        std_proj = float(np.std(all_projs))
        range_proj = float(np.max(all_projs) - np.min(all_projs))
        cv = abs(std_proj / (mean_proj + 1e-8))
        
        analysis["trait_variance"][trait] = {
            "mean": round(mean_proj, 4),
            "std": round(std_proj, 4),
            "range": round(range_proj, 4),
            "cv": round(cv, 4),
            "n": len(all_projs),
            "meaningful_variance": bool(std_proj > 0.05),
        }
        
        status = "OK" if std_proj > 0.05 else "LOW"
        print(f"{trait:<40s} {mean_proj:>8.3f} {std_proj:>8.3f} {range_proj:>12.3f} {cv:>8.3f} [{status}]")
    
    # 2. Emotion-based differences
    print(f"\n--- Emotion-Based Differences ---")
    
    for trait in TRAITS:
        emotion_groups = {}
        for p in all_projections:
            emo = p.get("emotion", "unknown")
            if emo == "unknown":
                continue
            if emo not in emotion_groups:
                emotion_groups[emo] = []
            proj = p.get("template_projections_tN", {}).get(trait, None)
            if proj is not None:
                emotion_groups[emo].append(proj)
        
        groups = [v for v in emotion_groups.values() if len(v) >= 5]
        if len(groups) >= 3:
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                analysis.setdefault("emotion_differences", {})[trait] = {
                    "f_stat": round(float(f_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "n_groups": len(groups),
                    "significant": bool(p_val < 0.05),
                }
                sig = "*" if p_val < 0.05 else ""
                print(f"  {trait:<40s} F={f_stat:.2f}, p={p_val:.4f} {sig}")
            except Exception:
                pass
    
    # 3. t1 vs tN drift
    print(f"\n--- t1 vs tN Differences (Within-Conversation Drift) ---")
    
    for trait in TRAITS:
        t1_projs = [p["template_projections_t1"].get(trait, 0) for p in all_projections
                     if trait in p.get("template_projections_t1", {})]
        tN_projs = [p["template_projections_tN"].get(trait, 0) for p in all_projections
                     if trait in p.get("template_projections_tN", {})]
        
        if len(t1_projs) < 10:
            continue
        
        drifts = [tN - t1 for t1, tN in zip(t1_projs, tN_projs)]
        mean_drift = float(np.mean(drifts))
        std_drift = float(np.std(drifts))
        
        try:
            t_stat, p_val = stats.ttest_rel(tN_projs, t1_projs)
            analysis["t1_vs_tN"][trait] = {
                "mean_drift": round(mean_drift, 4),
                "std_drift": round(std_drift, 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_val), 4),
                "significant": bool(p_val < 0.05),
            }
            sig = "*" if p_val < 0.05 else ""
            print(f"  {trait:<40s} drift={mean_drift:+.4f} (sd={std_drift:.4f}) t={t_stat:.2f}, p={p_val:.4f} {sig}")
        except Exception:
            pass
    
    # Summary
    n_meaningful = sum(1 for v in analysis["trait_variance"].values() if v.get("meaningful_variance", False))
    n_emotion_sig = sum(1 for v in analysis.get("emotion_differences", {}).values() if v.get("significant", False))
    n_drift_sig = sum(1 for v in analysis.get("t1_vs_tN", {}).values() if v.get("significant", False))
    
    analysis["summary"] = {
        "n_traits_meaningful_variance": n_meaningful,
        "n_traits_emotion_significant": n_emotion_sig,
        "n_traits_drift_significant": n_drift_sig,
        "ecological_validity": n_meaningful >= 5,
        "verdict": (
            f"{n_meaningful}/8 traits show meaningful variance on naturalistic data. "
            f"{n_emotion_sig}/8 differentiate by emotion type. "
            f"{n_drift_sig}/8 show within-conversation drift."
        ),
    }
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Meaningful variance: {n_meaningful}/8 traits")
    print(f"  Emotion-differentiated: {n_emotion_sig}/8 traits")
    print(f"  Within-conv drift: {n_drift_sig}/8 traits")
    print(f"  Ecological validity: {'YES' if n_meaningful >= 5 else 'LIMITED'}")
    
    # Save
    output = {
        "analysis": analysis,
        "raw_projections": all_projections,
    }
    
    output_path = f"/results/esconv_naturalistic_{model_key}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    vol.commit()
    
    print(f"\nSaved to {output_path}")
    return analysis


@app.local_entrypoint()
def main(model: str = "llama3"):
    """
    ESConv naturalistic data test.
    
    Usage:
        modal run modal_esconv_naturalistic.py --model llama3
        modal run modal_esconv_naturalistic.py --model qwen2
        modal run modal_esconv_naturalistic.py --model mistral
    """
    if model not in MODELS:
        print(f"ERROR: Unknown model '{model}'. Available: {list(MODELS.keys())}")
        return
    
    print(f"\n{'='*70}")
    print(f"M-EXP2: ESConv NATURALISTIC DATA TEST - {model.upper()}")
    print(f"{'='*70}")
    print(f"Parallelization: {N_PARALLEL} containers ({MAX_CONVERSATIONS // N_PARALLEL} convs each)")
    
    result = run_esconv_analysis.remote(model)
    
    if result:
        print(f"\nVerdict: {result.get('summary', {}).get('verdict', 'N/A')}")
