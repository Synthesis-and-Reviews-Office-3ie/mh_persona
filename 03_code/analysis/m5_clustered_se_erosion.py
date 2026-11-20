"""
m5: Clustered Standard Errors for Context Erosion (Section 6.1)
================================================================
Reviewer concern: OLS on pooled turn-level observations (~2,000/model)
treats turns from the same conversation as independent, inflating
significance. Observations are nested within 20 conversations.

Fix: Re-run OLS with cluster-robust standard errors (CR1) by conversation_id.
Also run mixed-effects model: projection ~ turn + (1 | conversation_id).

Output: Updated p-values and SEs for Table 8, plus comparison table.
"""

import json
import os
import numpy as np
from scipy import stats

# Optional: statsmodels for robust SEs
try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS as SM_OLS
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Optional: mixed-effects
try:
    import statsmodels.formula.api as smf
    HAS_FORMULA = True
except ImportError:
    HAS_FORMULA = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELS = ["llama3", "qwen2", "mistral"]

CONCERNING_TRAITS = {
    "emotional_over_involvement": "increase",
    "sycophancy_harmful_validation": "increase",
    "uncritical_validation": "increase",
    "non_judgmental_acceptance": "decrease",
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


def load_erosion_data(model):
    """Load context erosion JSON and extract turn-level data."""
    fpath = os.path.join(RESULTS_DIR, f"context_erosion_{model}.json")
    with open(fpath, "r") as f:
        data = json.load(f)

    rows = []
    sessions = data.get("sessions", [])
    for s_idx, session in enumerate(sessions):
        conv_id = session.get("session_id", f"conv_{s_idx}")
        turns = session.get("turns", [])
        for turn in turns:
            turn_num = turn.get("turn", 0)
            traits = turn.get("traits", {})
            row = {
                "conv_id": conv_id,
                "turn": turn_num,
            }
            for trait_name, trait_data in traits.items():
                row[trait_name] = trait_data.get("raw_projection", np.nan)
            rows.append(row)
    return rows


def ols_naive(turns, projections):
    """Standard OLS (no clustering)."""
    X = np.column_stack([np.ones(len(turns)), turns])
    y = np.array(projections)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    residuals = y - y_hat
    n = len(y)
    k = 2
    mse = np.sum(residuals**2) / (n - k)
    var_beta = mse * np.linalg.inv(X.T @ X)
    se_slope = np.sqrt(var_beta[1, 1])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    t_stat = beta[1] / se_slope
    p_value = 2 * stats.t.sf(abs(t_stat), n - k)
    return {
        "slope": beta[1],
        "intercept": beta[0],
        "se_naive": se_slope,
        "r_squared": r_squared,
        "t_stat": t_stat,
        "p_value_naive": p_value,
        "n": n,
    }


def ols_clustered(turns, projections, cluster_ids):
    """OLS with cluster-robust (CR1) standard errors."""
    X = np.column_stack([np.ones(len(turns)), turns])
    y = np.array(projections)
    n = len(y)
    k = 2

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta

    # Cluster-robust variance (CR1 / "HC1-cluster")
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    # Bread: (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)

    # Meat: sum of X_g' e_g e_g' X_g
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = cluster_ids == c
        X_g = X[mask]
        e_g = residuals[mask].reshape(-1, 1)
        meat += X_g.T @ e_g @ e_g.T @ X_g

    # CR1 finite-sample correction: G/(G-1) * (n-1)/(n-k)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V_cluster = correction * XtX_inv @ meat @ XtX_inv

    se_cluster = np.sqrt(V_cluster[1, 1])
    t_stat = beta[1] / se_cluster
    # Use G-1 degrees of freedom for cluster-robust t-test
    p_value = 2 * stats.t.sf(abs(t_stat), G - 1)

    return {
        "se_clustered": se_cluster,
        "t_stat_clustered": t_stat,
        "p_value_clustered": p_value,
        "n_clusters": G,
        "df": G - 1,
    }


def mixed_effects_model(turns, projections, cluster_ids):
    """Random-intercept mixed-effects model: projection ~ turn + (1|conv)."""
    if not HAS_FORMULA:
        return {"error": "statsmodels.formula not available"}

    import pandas as pd
    df = pd.DataFrame({
        "projection": projections,
        "turn": turns,
        "conv_id": cluster_ids,
    })

    try:
        model = smf.mixedlm("projection ~ turn", df, groups=df["conv_id"])
        result = model.fit(reml=True)
        slope = result.fe_params["turn"]
        se = result.bse["turn"]
        p_val = result.pvalues["turn"]
        # Random effect variance
        re_var = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, 'iloc') else float(result.cov_re)
        return {
            "slope_mlm": slope,
            "se_mlm": se,
            "p_value_mlm": p_val,
            "re_variance": re_var,
            "residual_variance": result.scale,
            "icc": re_var / (re_var + result.scale) if (re_var + result.scale) > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    all_results = {}
    print("=" * 80)
    print("m5: CLUSTERED STANDARD ERRORS FOR CONTEXT EROSION")
    print("=" * 80)

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model.upper()}")
        print(f"{'='*70}")

        rows = load_erosion_data(model)
        if not rows:
            print(f"  No data found for {model}")
            continue

        # Extract arrays
        turns_all = np.array([r["turn"] for r in rows], dtype=float)
        conv_ids_all = np.array([r["conv_id"] for r in rows])
        n_total = len(rows)
        n_clusters = len(np.unique(conv_ids_all))
        print(f"  Total observations: {n_total}")
        print(f"  Conversations (clusters): {n_clusters}")
        print(f"  Avg turns/conversation: {n_total / n_clusters:.1f}")

        model_results = {}

        for trait in ALL_TRAITS:
            projections = np.array([r.get(trait, np.nan) for r in rows])
            valid = ~np.isnan(projections)
            t = turns_all[valid]
            p = projections[valid]
            c = conv_ids_all[valid]
            n = len(t)

            if n < 10:
                print(f"  {trait}: insufficient data (n={n})")
                model_results[trait] = {"error": "insufficient data", "n": n}
                continue

            # 1. Naive OLS
            naive = ols_naive(t, p)

            # 2. Clustered SEs
            clustered = ols_clustered(t, p, c)

            # 3. Mixed-effects
            mlm = mixed_effects_model(t, p, c)

            # Combine
            result = {**naive, **clustered, **mlm}

            # Compute SE inflation factor
            if naive["se_naive"] > 0:
                result["se_inflation"] = clustered["se_clustered"] / naive["se_naive"]
            else:
                result["se_inflation"] = np.nan

            model_results[trait] = result

            # Significance change?
            naive_sig = naive["p_value_naive"] < 0.05
            cluster_sig = clustered["p_value_clustered"] < 0.05
            sig_change = "LOST" if naive_sig and not cluster_sig else (
                "GAINED" if not naive_sig and cluster_sig else "same"
            )

            is_concerning = trait in CONCERNING_TRAITS
            tag = " [CONCERNING]" if is_concerning else ""

            print(f"\n  {trait}{tag}:")
            print(f"    slope       = {naive['slope']:+.6f}")
            print(f"    R-squared   = {naive['r_squared']:.4f}")
            print(f"    SE naive    = {naive['se_naive']:.6f}  p={naive['p_value_naive']:.6f}")
            print(f"    SE clustered= {clustered['se_clustered']:.6f}  p={clustered['p_value_clustered']:.6f}  (df={clustered['df']})")
            print(f"    SE inflation= {result.get('se_inflation', 'N/A'):.2f}x")
            if "se_mlm" in mlm:
                print(f"    SE MLM      = {mlm['se_mlm']:.6f}  p={mlm['p_value_mlm']:.6f}")
                print(f"    ICC (conv)  = {mlm.get('icc', 0):.3f}")
            print(f"    Significance: {sig_change}")

        all_results[model] = model_results

    # ================================================================
    # SUMMARY TABLE: Concerning traits with all three SE estimates
    # ================================================================
    print(f"\n\n{'='*80}")
    print("SUMMARY: CONCERNING TRAITS — Naive vs Clustered vs MLM")
    print(f"{'='*80}")
    print(f"{'Model':<10} {'Trait':<35} {'slope':>9} {'SE_naive':>9} {'p_naive':>10} "
          f"{'SE_clust':>9} {'p_clust':>10} {'SE_MLM':>9} {'p_MLM':>10} {'Infl':>6} {'Sig':>6}")
    print("-" * 130)

    for model in MODELS:
        for trait in CONCERNING_TRAITS:
            r = all_results.get(model, {}).get(trait, {})
            if "error" in r:
                print(f"{model:<10} {trait:<35} — insufficient data")
                continue

            se_n = r.get("se_naive", 0)
            p_n = r.get("p_value_naive", 1)
            se_c = r.get("se_clustered", 0)
            p_c = r.get("p_value_clustered", 1)
            se_m = r.get("se_mlm", 0)
            p_m = r.get("p_value_mlm", 1)
            infl = r.get("se_inflation", 0)
            slope = r.get("slope", 0)

            naive_sig = "*" if p_n < 0.05 else ""
            clust_sig = "*" if p_c < 0.05 else ""
            mlm_sig = "*" if p_m < 0.05 else ""

            print(f"{model:<10} {trait:<35} {slope:+.6f} "
                  f"{se_n:.6f}{naive_sig:1} {p_n:.6f}  "
                  f"{se_c:.6f}{clust_sig:1} {p_c:.6f}  "
                  f"{se_m:.6f}{mlm_sig:1} {p_m:.6f}  "
                  f"{infl:5.1f}x")

    # ================================================================
    # KEY QUESTION: Does emotional_over_involvement remain significant
    # across all 3 models with clustered SEs?
    # ================================================================
    print(f"\n{'='*80}")
    print("KEY RESULT: emotional_over_involvement significance under clustering")
    print(f"{'='*80}")
    eoi_results = {}
    for model in MODELS:
        r = all_results.get(model, {}).get("emotional_over_involvement", {})
        p_naive = r.get("p_value_naive", 1)
        p_clust = r.get("p_value_clustered", 1)
        p_mlm = r.get("p_value_mlm", 1)
        eoi_results[model] = {
            "p_naive": p_naive,
            "p_clustered": p_clust,
            "p_mlm": p_mlm,
            "significant_naive": p_naive < 0.05,
            "significant_clustered": p_clust < 0.05,
            "significant_mlm": p_mlm < 0.05,
        }
        print(f"  {model}: naive p={p_naive:.6f} | clustered p={p_clust:.6f} | MLM p={p_mlm:.6f}")
        if p_clust < 0.001:
            print(f"    -> Remains p<0.001 with clustering")
        elif p_clust < 0.05:
            print(f"    -> Still significant (p<0.05) but weaker")
        else:
            print(f"    -> LOST SIGNIFICANCE with clustering!")

    # ================================================================
    # Save results
    # ================================================================
    output = {
        "analysis": "m5_clustered_standard_errors",
        "description": "Context erosion OLS with cluster-robust SEs (by conversation) and mixed-effects models",
        "models": all_results,
        "eoi_key_result": eoi_results,
    }

    # Sanitize for JSON
    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        return obj

    output = sanitize(output)

    out_path = os.path.join(RESULTS_DIR, "m5_clustered_se_erosion.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
