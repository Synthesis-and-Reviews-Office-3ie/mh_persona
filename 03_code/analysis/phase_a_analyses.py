"""
Phase A Revision Analyses: Factor Analysis + Post-Hoc Power
============================================================
Implements:
  EXP5 — Exploratory Factor Analysis on inter-trait correlation matrices
  EXP6 — Post-hoc power analysis for Table 1 traits

Usage:
    python phase_a_analyses.py

Requires: numpy, scipy, matplotlib, factor_analyzer (optional)
"""

import json
import os
import math
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

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

TRAIT_SHORT = {
    "empathetic_responsiveness": "EmpResp",
    "non_judgmental_acceptance": "NonJudg",
    "boundary_maintenance": "BoundMt",
    "crisis_recognition": "CrisRec",
    "emotional_over_involvement": "EmotOvr",
    "abandonment_of_therapeutic_frame": "AbandFr",
    "uncritical_validation": "UncritV",
    "sycophancy_harmful_validation": "SycHarm",
}


# ============================================================================
# EXP5: EXPLORATORY FACTOR ANALYSIS
# ============================================================================

def eigenvalue_analysis(corr_matrix, model_name, n=500):
    """
    Perform eigenvalue decomposition on the correlation matrix.
    Reports eigenvalues, % variance explained, cumulative variance.
    Uses parallel analysis (Horn 1965) to determine number of factors.
    """
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_var = eigvals.sum()
    pct_var = eigvals / total_var * 100
    cum_var = np.cumsum(pct_var)

    print(f"\n{'='*60}")
    print(f"EIGENVALUE ANALYSIS: {model_name}")
    print(f"{'='*60}")
    print(f"{'Factor':<10} {'Eigenvalue':<12} {'% Variance':<12} {'Cumulative %':<12}")
    print("-" * 46)
    for i, (ev, pv, cv) in enumerate(zip(eigvals, pct_var, cum_var)):
        marker = " <-- Kaiser" if ev >= 1.0 else ""
        print(f"  F{i+1:<7} {ev:>10.4f}   {pv:>9.2f}%   {cv:>9.2f}%{marker}")

    # Parallel analysis (random data comparison)
    n_iter = 1000
    p = len(corr_matrix)
    random_eigvals = np.zeros((n_iter, p))
    for i in range(n_iter):
        random_data = np.random.randn(n, p)
        random_corr = np.corrcoef(random_data, rowvar=False)
        re = np.linalg.eigvalsh(random_corr)
        random_eigvals[i] = np.sort(re)[::-1]

    pa_95 = np.percentile(random_eigvals, 95, axis=0)
    n_factors_pa = sum(1 for ev, pa in zip(eigvals, pa_95) if ev > pa)
    n_factors_kaiser = sum(1 for ev in eigvals if ev >= 1.0)

    print(f"\nParallel analysis (95th percentile thresholds):")
    print(f"{'Factor':<10} {'Observed':<12} {'Random 95th':<12} {'Retain?':<10}")
    print("-" * 44)
    for i in range(p):
        retain = "YES" if eigvals[i] > pa_95[i] else "no"
        print(f"  F{i+1:<7} {eigvals[i]:>10.4f}   {pa_95[i]:>10.4f}   {retain}")

    print(f"\nKaiser criterion (eigenvalue > 1): {n_factors_kaiser} factors")
    print(f"Parallel analysis: {n_factors_pa} factors")

    return {
        "eigenvalues": eigvals.tolist(),
        "pct_variance": pct_var.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "n_factors_kaiser": n_factors_kaiser,
        "n_factors_parallel": n_factors_pa,
        "parallel_thresholds_95": pa_95.tolist(),
    }


def varimax_rotation(loadings, max_iter=100, tol=1e-6):
    """Simple varimax rotation implementation."""
    n, k = loadings.shape
    rotation = np.eye(k)
    d = 0

    for _ in range(max_iter):
        old_d = d
        comp = loadings @ rotation
        u, s, vt = np.linalg.svd(
            loadings.T @ (comp ** 3 - (1.0 / n) * comp @ np.diag(np.sum(comp ** 2, axis=0)))
        )
        rotation = u @ vt
        d = np.sum(s)
        if abs(d - old_d) < tol:
            break

    return loadings @ rotation


def factor_analysis(corr_matrix, n_factors, model_name):
    """
    Principal axis factoring with varimax rotation.
    Uses eigendecomposition of the correlation matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Extract loadings for n_factors
    loadings = eigvecs[:, :n_factors] * np.sqrt(eigvals[:n_factors])

    # Varimax rotation
    rotated = varimax_rotation(loadings)

    # Communalities
    communalities = np.sum(rotated ** 2, axis=1)

    print(f"\n{'='*60}")
    print(f"FACTOR LOADINGS (Varimax, {n_factors} factors): {model_name}")
    print(f"{'='*60}")

    short_names = [TRAIT_SHORT[t] for t in TRAITS]

    header = f"{'Trait':<10}" + "".join(f"{'F'+str(i+1):>10}" for i in range(n_factors)) + f"{'Commun.':>10}"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(short_names):
        row = f"{name:<10}"
        for j in range(n_factors):
            val = rotated[i, j]
            marker = "*" if abs(val) >= 0.40 else " "
            row += f"{val:>9.3f}{marker}"
        row += f"{communalities[i]:>10.3f}"
        print(row)

    # Variance explained by each rotated factor
    factor_var = np.sum(rotated ** 2, axis=0)
    total_var = corr_matrix.shape[0]
    pct = factor_var / total_var * 100

    print(f"\nVariance explained per factor:")
    for i in range(n_factors):
        print(f"  F{i+1}: {factor_var[i]:.3f} ({pct[i]:.1f}%)")
    print(f"  Total: {factor_var.sum():.3f} ({pct.sum():.1f}%)")

    # Identify factor interpretations (traits loading > 0.4)
    interpretations = {}
    for j in range(n_factors):
        high_loading = [(short_names[i], rotated[i, j])
                        for i in range(len(short_names))
                        if abs(rotated[i, j]) >= 0.40]
        high_loading.sort(key=lambda x: abs(x[1]), reverse=True)
        interpretations[f"F{j+1}"] = high_loading

    print(f"\nFactor interpretations (loadings >= 0.40):")
    for fname, traits in interpretations.items():
        trait_str = ", ".join(f"{t}({v:+.3f})" for t, v in traits)
        print(f"  {fname}: {trait_str}")

    return {
        "loadings": rotated.tolist(),
        "communalities": communalities.tolist(),
        "variance_explained": factor_var.tolist(),
        "pct_variance": pct.tolist(),
        "interpretations": {k: [(t, round(v, 3)) for t, v in vs]
                            for k, vs in interpretations.items()},
    }


def run_factor_analysis():
    """Run EFA on all model correlation matrices."""
    filepath = os.path.join(RESULTS_DIR, "m5b_cross_trait_discriminant.json")
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        return None

    with open(filepath) as f:
        data = json.load(f)

    all_results = {}

    for model_name in ["llama3", "qwen2", "mistral", "probe_qwen2", "probe_mistral"]:
        if model_name not in data:
            print(f"Skipping {model_name} (not in data)")
            continue

        corr = np.array(data[model_name]["correlation_matrix"])
        n = data[model_name]["n_scenarios"]

        # Step 1: Eigenvalue analysis
        eigen_results = eigenvalue_analysis(corr, model_name, n=n)

        # Step 2: Factor analysis with parallel-analysis-determined n_factors
        n_factors = eigen_results["n_factors_parallel"]
        if n_factors < 2:
            n_factors = 2  # Minimum 2 for interpretability
        if n_factors > 5:
            n_factors = 5  # Cap at 5

        fa_results = factor_analysis(corr, n_factors, model_name)

        all_results[model_name] = {
            "eigenvalue_analysis": eigen_results,
            "factor_analysis": fa_results,
            "n_factors_used": n_factors,
        }

    # Cross-model summary
    print(f"\n{'='*60}")
    print("CROSS-MODEL FACTOR STRUCTURE SUMMARY")
    print(f"{'='*60}")
    for model, res in all_results.items():
        n_k = res["eigenvalue_analysis"]["n_factors_kaiser"]
        n_p = res["eigenvalue_analysis"]["n_factors_parallel"]
        ev = res["eigenvalue_analysis"]["eigenvalues"][:4]
        ev_str = ", ".join(f"{v:.2f}" for v in ev)
        print(f"  {model:<15} Kaiser={n_k}, Parallel={n_p}, Top-4 eigenvalues: [{ev_str}]")

    return all_results


# ============================================================================
# EXP6: POST-HOC POWER ANALYSIS FOR TABLE 1
# ============================================================================

TABLE1_DATA = {
    "sycophancy_harmful_validation": {"r": 0.489, "n": 50, "layer": 19},
    "abandonment_of_therapeutic_frame": {"r": 0.470, "n": 50, "layer": 19},
    "emotional_over_involvement": {"r": 0.441, "n": 50, "layer": 19},
    "empathetic_responsiveness": {"r": 0.424, "n": 50, "layer": 17},
    "crisis_recognition": {"r": 0.374, "n": 50, "layer": 18},
    "uncritical_validation": {"r": 0.364, "n": 50, "layer": 18},
    "non_judgmental_acceptance": {"r": 0.346, "n": 50, "layer": 18},
    "boundary_maintenance": {"r": 0.302, "n": 50, "layer": 18},
}

ALPHA = 0.05


def r_to_fisher_z(r):
    r = max(-0.999, min(0.999, r))
    return 0.5 * math.log((1 + r) / (1 - r))


def power_for_r(r, n, alpha=ALPHA):
    """Compute statistical power for two-tailed test of Pearson r."""
    if n <= 3:
        return 0.0
    if abs(r) < 1e-10:
        return alpha

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ncp = r_to_fisher_z(r) * math.sqrt(n - 3)
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    return power


def n_for_power(r, target_power=0.80, alpha=ALPHA):
    """Find minimum N to achieve target power."""
    for n in range(10, 10000):
        if power_for_r(r, n, alpha) >= target_power:
            return n
    return None


def run_power_analysis():
    """Post-hoc power analysis for all Table 1 traits."""
    print(f"\n{'='*60}")
    print("POST-HOC POWER ANALYSIS — TABLE 1")
    print(f"{'='*60}")
    print(f"{'Trait':<42} {'r':>6} {'N':>4} {'Power':>7} {'N for 80%':>10} {'N for 90%':>10}")
    print("-" * 80)

    results = {}
    for trait, info in TABLE1_DATA.items():
        r = info["r"]
        n = info["n"]
        pwr = power_for_r(r, n)
        n80 = n_for_power(r, 0.80)
        n90 = n_for_power(r, 0.90)

        short = TRAIT_SHORT.get(trait, trait[:8])
        flag = " (!)" if pwr < 0.80 else " OK"
        print(f"  {trait:<40} {r:>6.3f} {n:>4} {pwr:>6.1%}{flag} {n80:>10} {n90:>10}")

        results[trait] = {
            "observed_r": r,
            "n": n,
            "post_hoc_power": round(pwr, 4),
            "n_for_80_power": n80,
            "n_for_90_power": n90,
            "adequately_powered": bool(pwr >= 0.80),
        }

    # Summary
    n_adequate = sum(1 for v in results.values() if v["adequately_powered"])
    print(f"\n  Adequately powered (>=80%): {n_adequate}/8 traits")
    print(f"  Underpowered (<80%): {8 - n_adequate}/8 traits")

    if 8 - n_adequate > 0:
        underpowered = [t for t, v in results.items() if not v["adequately_powered"]]
        for t in underpowered:
            print(f"    - {t}: power={results[t]['post_hoc_power']:.1%}, "
                  f"need N={results[t]['n_for_80_power']} for 80%")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # EXP5: Factor Analysis
    print("\n" + "=" * 70)
    print("EXP5: EXPLORATORY FACTOR ANALYSIS")
    print("=" * 70)
    fa_results = run_factor_analysis()

    # EXP6: Post-hoc Power
    power_results = run_power_analysis()

    # Save all results
    output = {
        "exp5_factor_analysis": fa_results,
        "exp6_posthoc_power": power_results,
    }

    outpath = os.path.join(RESULTS_DIR, "phase_a_analyses.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
