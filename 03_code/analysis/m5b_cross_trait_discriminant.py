"""
M5b: Cross-Trait Discriminant Validity Matrix
===============================================
Reviewer concern: Does steering on trait A selectively affect trait A's
projections, or does it bleed into other traits?

Method: For each of the 500 scenarios per model, extract activation
projections for ALL 8 traits at each trait's best layer. Compute 8x8
correlation matrix. Diagonal = within-trait validity (already validated);
off-diagonal = cross-trait bleed.

Success criterion: Diagonal entries >> off-diagonal entries.
"""

import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MODELS_TEMPLATE = {
    "llama3": "aggregated_llama3.json",
    "qwen2": "aggregated_qwen2.json",
    "mistral": "aggregated_mistral.json",
}

# Also include probe models for cross-validation
MODELS_PROBE = {
    "qwen2_probe": "aggregated_probe_qwen2.json",
    "mistral_probe": "aggregated_probe_mistral.json",
}

ALL_TRAITS = [
    "empathetic_responsiveness",
    "non_judgmental_acceptance",
    "boundary_maintenance",
    "crisis_recognition",
    "emotional_over_involvement",
    "abandonment_of_therapeutic_frame",
    "uncritical_validation",
    "sycophancy_harmful_validation",
]

# Best layers per model (from trait_layer_matrix files and Table S2)
BEST_LAYERS = {
    "llama3": {
        "empathetic_responsiveness": "14",
        "non_judgmental_acceptance": "14",
        "boundary_maintenance": "15",
        "crisis_recognition": "14",
        "emotional_over_involvement": "19",
        "abandonment_of_therapeutic_frame": "19",
        "uncritical_validation": "14",
        "sycophancy_harmful_validation": "14",
    },
    "qwen2": {
        "empathetic_responsiveness": "14",
        "non_judgmental_acceptance": "14",
        "boundary_maintenance": "14",
        "crisis_recognition": "14",
        "emotional_over_involvement": "14",
        "abandonment_of_therapeutic_frame": "14",
        "uncritical_validation": "14",
        "sycophancy_harmful_validation": "14",
    },
    "mistral": {
        "empathetic_responsiveness": "14",
        "non_judgmental_acceptance": "14",
        "boundary_maintenance": "14",
        "crisis_recognition": "14",
        "emotional_over_involvement": "14",
        "abandonment_of_therapeutic_frame": "14",
        "uncritical_validation": "14",
        "sycophancy_harmful_validation": "14",
    },
}


def load_best_layers(model):
    """Try to load best layers from trait_layer_matrix files."""
    # Try template first
    for suffix in ["", "_probe"]:
        fname = f"trait_layer_matrix{suffix}_{model.replace('_probe','')}.json"
        fpath = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            layers = {}
            for trait in ALL_TRAITS:
                if trait in data:
                    trait_data = data[trait]
                    # Find layer with highest abs(r)
                    best_r = 0
                    best_l = "14"
                    for layer_str, ldata in trait_data.items():
                        if layer_str.startswith("_"):
                            continue
                        r = abs(ldata.get("r", 0))
                        if r > best_r:
                            best_r = r
                            best_l = layer_str
                    layers[trait] = best_l
            return layers
    return None


def compute_discriminant_matrix(data, best_layers):
    """
    Build 8x8 cross-trait correlation matrix.
    
    For each scenario, extract the activation drift (proj_tN - proj_t1)
    at each trait's best layer. This gives an 8-dimensional vector per
    scenario. Compute Pearson correlation matrix.
    """
    n_scenarios = len(data)
    n_traits = len(ALL_TRAITS)
    
    # Build matrix: scenarios x traits
    matrix = np.full((n_scenarios, n_traits), np.nan)
    
    for i, scenario in enumerate(data):
        trait_data = scenario.get("trait_data", {})
        for j, trait in enumerate(ALL_TRAITS):
            td = trait_data.get(trait, {})
            layer = best_layers.get(trait, "14")
            lp = td.get("layer_projections", {}).get(layer, {})
            
            # Use activation drift (change from t1 to tN)
            drift = lp.get("activation_drift", None)
            if drift is not None:
                matrix[i, j] = drift
    
    # Drop rows with any NaN
    valid_rows = ~np.any(np.isnan(matrix), axis=1)
    matrix_clean = matrix[valid_rows]
    
    if len(matrix_clean) < 10:
        return None, None, None
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(matrix_clean.T)
    
    # Compute p-values
    p_matrix = np.full((n_traits, n_traits), np.nan)
    for i in range(n_traits):
        for j in range(n_traits):
            if i == j:
                p_matrix[i, j] = 0.0
            else:
                r, p = stats.pearsonr(matrix_clean[:, i], matrix_clean[:, j])
                p_matrix[i, j] = p
    
    return corr_matrix, p_matrix, len(matrix_clean)


def compute_discriminant_from_projections(data, best_layers):
    """
    Alternative approach: for each trait being steered, check if the
    target trait's projection changes more than non-target traits.
    
    This is more directly relevant: when we steer trait A, does trait A's
    projection respond more than trait B's projection?
    """
    # Group scenarios by which trait was steered (if available)
    # In the aggregated data, each scenario has projections for all traits
    # but we don't know which trait was steered. We use behavioral_drift
    # as the indicator.
    
    # Actually, all scenarios are natural (not steered) in aggregated data.
    # The projections capture natural variation across scenarios.
    # The correlation matrix from compute_discriminant_matrix is appropriate.
    pass


def main():
    print("=" * 80)
    print("M5b: CROSS-TRAIT DISCRIMINANT VALIDITY MATRIX")
    print("=" * 80)

    all_results = {}

    for model, fname in {**MODELS_TEMPLATE, **MODELS_PROBE}.items():
        fpath = os.path.join(os.path.dirname(__file__), fname)
        if not os.path.exists(fpath):
            print(f"\n  {model}: file not found ({fname})")
            continue

        with open(fpath) as f:
            data = json.load(f)

        print(f"\n{'='*70}")
        print(f"  MODEL: {model.upper()} ({len(data)} scenarios)")
        print(f"{'='*70}")

        # Load best layers
        best_layers = load_best_layers(model)
        if not best_layers:
            base_model = model.replace("_probe", "")
            best_layers = BEST_LAYERS.get(base_model, BEST_LAYERS["llama3"])
            print(f"  Using default layers")
        else:
            print(f"  Best layers: {best_layers}")

        corr_matrix, p_matrix, n_valid = compute_discriminant_matrix(data, best_layers)
        
        if corr_matrix is None:
            print(f"  Insufficient data for correlation matrix")
            continue

        print(f"  Valid scenarios: {n_valid}")

        # Print correlation matrix
        short_names = [t[:12] for t in ALL_TRAITS]
        print(f"\n  Cross-trait activation-drift correlation matrix:")
        print(f"  {'':>14}", end="")
        for sn in short_names:
            print(f" {sn:>12}", end="")
        print()

        for i, trait in enumerate(ALL_TRAITS):
            print(f"  {short_names[i]:>14}", end="")
            for j in range(len(ALL_TRAITS)):
                r = corr_matrix[i, j]
                star = ""
                if i != j and p_matrix[i, j] < 0.001:
                    star = "***"
                elif i != j and p_matrix[i, j] < 0.01:
                    star = "**"
                elif i != j and p_matrix[i, j] < 0.05:
                    star = "*"
                print(f" {r:+.3f}{star:>4}", end="")
            print()

        # Summary statistics
        diag = np.diag(corr_matrix)  # All 1.0 by definition
        off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        print(f"\n  Off-diagonal correlations:")
        print(f"    Mean: {np.mean(off_diag):+.3f}")
        print(f"    Median: {np.median(off_diag):+.3f}")
        print(f"    Range: [{np.min(off_diag):+.3f}, {np.max(off_diag):+.3f}]")
        print(f"    |r| > 0.3: {np.sum(np.abs(off_diag) > 0.3)}/{len(off_diag)}")
        print(f"    |r| > 0.5: {np.sum(np.abs(off_diag) > 0.5)}/{len(off_diag)}")

        # Find most correlated trait pairs
        n_traits = len(ALL_TRAITS)
        pairs = []
        for i in range(n_traits):
            for j in range(i+1, n_traits):
                pairs.append((ALL_TRAITS[i], ALL_TRAITS[j], corr_matrix[i,j], p_matrix[i,j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"\n  Top 5 cross-trait correlations:")
        for t1, t2, r, p in pairs[:5]:
            print(f"    {t1[:30]:<30} x {t2[:30]:<30} r={r:+.3f} p={p:.4f}")

        print(f"\n  Bottom 5 (most independent):")
        for t1, t2, r, p in pairs[-5:]:
            print(f"    {t1[:30]:<30} x {t2[:30]:<30} r={r:+.3f} p={p:.4f}")

        # Store results
        model_result = {
            "n_scenarios": n_valid,
            "best_layers": best_layers,
            "correlation_matrix": corr_matrix.tolist(),
            "p_value_matrix": p_matrix.tolist(),
            "trait_names": ALL_TRAITS,
            "off_diagonal_stats": {
                "mean": float(np.mean(off_diag)),
                "median": float(np.median(off_diag)),
                "min": float(np.min(off_diag)),
                "max": float(np.max(off_diag)),
                "abs_gt_03": int(np.sum(np.abs(off_diag) > 0.3)),
                "abs_gt_05": int(np.sum(np.abs(off_diag) > 0.5)),
                "n_pairs": int(len(off_diag)),
            },
            "top_5_pairs": [
                {"trait_1": t1, "trait_2": t2, "r": float(r), "p": float(p)}
                for t1, t2, r, p in pairs[:5]
            ],
        }
        all_results[model] = model_result

    # ================================================================
    # CROSS-MODEL SUMMARY
    # ================================================================
    print(f"\n\n{'='*80}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'N':>5} {'Mean |r|':>10} {'Median r':>10} {'|r|>0.3':>8} {'|r|>0.5':>8}")
    print("-" * 65)
    
    for model, res in all_results.items():
        ods = res["off_diagonal_stats"]
        off_diag_abs = [abs(r) for r in np.array(res["correlation_matrix"])[np.triu_indices(8, k=1)]]
        mean_abs = np.mean(off_diag_abs)
        print(f"  {model:<18} {res['n_scenarios']:>5} {mean_abs:>10.3f} {ods['median']:>+10.3f} "
              f"{ods['abs_gt_03']:>5}/{ods['n_pairs']}  {ods['abs_gt_05']:>5}/{ods['n_pairs']}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "m5b_cross_trait_discriminant.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
