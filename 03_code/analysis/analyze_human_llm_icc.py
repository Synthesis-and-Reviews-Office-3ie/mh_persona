#!/usr/bin/env python3
"""
Human vs LLM Judge ICC Analysis
================================
Computes Intraclass Correlation Coefficients (ICC) between human rater scores
and LLM judge scores (GPT-4o, Claude, Gemini) on the same 120 steered responses.

Outputs:
  - ICC(2,1) absolute agreement and consistency
  - Pearson r (concurrent validity)
  - Bland-Altman bias and limits of agreement
  - Breakdowns by trait, model, on-target vs off-target
"""

import json, csv, os, sys
import numpy as np
from scipy import stats
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "03_code" / "results"
HUMAN_DIR = BASE / "05_docs" / "human_validation"

# ─── Trait name normalisation ──────────────────────────────────────────────
TRAIT_DISPLAY_TO_KEY = {
    "Empathetic Responsiveness": "empathetic_responsiveness",
    "Non-Judgmental Acceptance": "non_judgmental_acceptance",
    "Emotional Over-Involvement": "emotional_over_involvement",
    "Boundary Maintenance": "boundary_maintenance",
    "Crisis Recognition": "crisis_recognition",
    "Sycophancy / Harmful Validation": "sycophancy_harmful_validation",
    "Uncritical Validation": "uncritical_validation",
    "Abandonment of Therapeutic Frame": "abandonment_of_therapeutic_frame",
}


def load_human_ratings():
    """Load human rater scores from rated CSV. Returns list of dicts."""
    path = HUMAN_DIR / "rating_sheet_EN_rated.csv"
    ratings = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["response_id"]
            # Each row has two trait scores
            for i in [1, 2]:
                trait_display = row[f"trait{i}_name"].strip()
                score_str = row[f"trait{i}_score"].strip()
                if not score_str:
                    continue
                trait_key = TRAIT_DISPLAY_TO_KEY.get(trait_display, trait_display.lower().replace(" ", "_"))
                ratings.append({
                    "response_id": rid,
                    "trait_display": trait_display,
                    "trait_key": trait_key,
                    "human_score": float(score_str),
                    "trait_slot": i,
                })
    return ratings


def load_response_key():
    """Load response key mapping R001 -> corpus metadata."""
    path = HUMAN_DIR / "response_key.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["responses"]


def load_judge_scores(filename):
    """Load LLM judge scores. Returns dict: response_idx -> {trait -> score}."""
    path = RESULTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    judge_model = data["metadata"]["judge_model"]
    
    # Build lookup: response_idx -> score (each entry scored on its OWN trait)
    idx_to_score = {}
    for entry in data["scores"]:
        idx = entry["response_idx"]
        score = entry["score"]
        trait = entry["trait"]
        if score is not None:
            idx_to_score[idx] = {
                "score": score,
                "trait": trait,
                "model": entry["model"],
                "coefficient": entry["coefficient"],
            }
    
    return judge_model, idx_to_score


def compute_icc_2_1(human_scores, llm_scores):
    """
    Compute ICC(2,1) - two-way random, single measures.
    Both absolute agreement and consistency.
    
    Using the formulas from Shrout & Fleiss (1979).
    """
    n = len(human_scores)
    if n < 3:
        return {"icc_agreement": np.nan, "icc_consistency": np.nan, "n": n}
    
    # Stack as (n, 2) matrix: col0 = human, col1 = LLM
    ratings = np.column_stack([human_scores, llm_scores])
    k = 2  # number of raters
    
    # Grand mean
    grand_mean = ratings.mean()
    
    # Row means (subject means)
    row_means = ratings.mean(axis=1)
    
    # Column means (rater means)
    col_means = ratings.mean(axis=0)
    
    # Sum of squares
    SS_total = np.sum((ratings - grand_mean) ** 2)
    SS_rows = k * np.sum((row_means - grand_mean) ** 2)  # Between subjects
    SS_cols = n * np.sum((col_means - grand_mean) ** 2)  # Between raters
    SS_error = SS_total - SS_rows - SS_cols                # Residual
    
    # Mean squares
    MS_rows = SS_rows / (n - 1)
    MS_cols = SS_cols / (k - 1)
    MS_error = SS_error / ((n - 1) * (k - 1))
    
    # ICC(2,1) absolute agreement
    icc_agreement = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error + k * (MS_cols - MS_error) / n)
    
    # ICC(2,1) consistency (ignoring systematic bias)
    icc_consistency = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error)
    
    # F-test for significance
    F_value = MS_rows / MS_error
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    p_value = 1 - stats.f.cdf(F_value, df1, df2)
    
    # 95% CI for ICC(2,1) agreement using F-distribution approximation
    FL = F_value / stats.f.ppf(0.975, df1, df2)
    FU = F_value * stats.f.ppf(0.975, df2, df1)
    
    icc_lower = (FL - 1) / (FL + k - 1)
    icc_upper = (FU - 1) / (FU + k - 1)
    
    return {
        "icc_agreement": round(icc_agreement, 3),
        "icc_consistency": round(icc_consistency, 3),
        "ci_lower": round(icc_lower, 3),
        "ci_upper": round(icc_upper, 3),
        "F": round(F_value, 2),
        "p": round(p_value, 4),
        "n": n,
        "human_mean": round(np.mean(human_scores), 2),
        "llm_mean": round(np.mean(llm_scores), 2),
        "bias": round(np.mean(llm_scores) - np.mean(human_scores), 2),
    }


def compute_pearson(human_scores, llm_scores):
    """Pearson r with 95% CI via Fisher z-transform."""
    n = len(human_scores)
    if n < 3:
        return {"r": np.nan, "p": np.nan, "n": n}
    
    r, p = stats.pearsonr(human_scores, llm_scores)
    
    # Fisher z CI
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se, z + 1.96 * se
    r_lo, r_hi = np.tanh(z_lo), np.tanh(z_hi)
    
    return {
        "r": round(r, 3),
        "p": round(p, 4),
        "ci_lower": round(r_lo, 3),
        "ci_upper": round(r_hi, 3),
        "n": n,
    }


def bland_altman(human_scores, llm_scores):
    """Bland-Altman analysis: bias and limits of agreement."""
    diff = np.array(llm_scores) - np.array(human_scores)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff
    
    return {
        "bias": round(mean_diff, 3),
        "std_diff": round(std_diff, 3),
        "loa_lower": round(loa_lower, 3),
        "loa_upper": round(loa_upper, 3),
    }


def interpret_icc(icc_val):
    """Cicchetti (1994) benchmarks."""
    if np.isnan(icc_val):
        return "N/A"
    if icc_val < 0.40:
        return "Poor"
    elif icc_val < 0.60:
        return "Fair"
    elif icc_val < 0.75:
        return "Good"
    else:
        return "Excellent"


def main():
    # ─── Load data ──────────────────────────────────────────────────────
    print("=" * 70)
    print("HUMAN vs LLM JUDGE ICC ANALYSIS")
    print("=" * 70)
    
    human_ratings = load_human_ratings()
    response_key = load_response_key()
    
    print(f"\nHuman ratings loaded: {len(human_ratings)} observations")
    print(f"Response keys: {len(response_key)} responses")
    
    # Judge files
    judge_files = {
        "GPT-4o": "judge_scores_gpt4o_mini.json",
        "Claude": "judge_scores_claude.json",
        "Gemini": "judge_scores_gemini.json",
    }
    
    all_results = {}
    
    for judge_name, judge_file in judge_files.items():
        print(f"\n{'─' * 70}")
        print(f"JUDGE: {judge_name}")
        print(f"{'─' * 70}")
        
        judge_model, idx_to_score = load_judge_scores(judge_file)
        print(f"  Model: {judge_model}")
        print(f"  Scores loaded: {len(idx_to_score)}")
        
        # ─── Match human to LLM scores ────────────────────────────────
        matched_all = []         # All 240 observations
        matched_on_target = []   # On-target only (trait matches)
        matched_off_target = []  # Off-target only
        by_trait = defaultdict(list)
        by_model = defaultdict(list)
        
        for hr in human_ratings:
            rid = hr["response_id"]
            rk = response_key.get(rid)
            if not rk:
                continue
            
            resp_idx = rk["response_idx"]
            llm_entry = idx_to_score.get(resp_idx)
            if not llm_entry:
                continue
            
            human_score = hr["human_score"]
            
            # The LLM judge scored on the RESPONSE's OWN trait (the steered trait)
            llm_scored_trait = llm_entry["trait"]
            human_rated_trait = hr["trait_key"]
            
            # Is this human rating on-target (same trait as LLM scored)?
            is_on_target = (human_rated_trait == llm_scored_trait)
            
            # For on-target: both human and LLM scored the SAME trait
            if is_on_target:
                llm_score = llm_entry["score"]
                entry = {
                    "rid": rid,
                    "trait": human_rated_trait,
                    "model": rk["model"],
                    "coefficient": rk["coefficient"],
                    "human": human_score,
                    "llm": llm_score,
                    "on_target": True,
                }
                matched_on_target.append(entry)
                matched_all.append(entry)
                by_trait[human_rated_trait].append(entry)
                by_model[rk["model"]].append(entry)
            else:
                # Off-target: human rated a DIFFERENT trait than LLM scored
                # We can't directly compare these (different constructs)
                matched_off_target.append({
                    "rid": rid,
                    "human_trait": human_rated_trait,
                    "llm_trait": llm_scored_trait,
                    "human": human_score,
                    "llm": llm_entry["score"],
                    "on_target": False,
                })
        
        print(f"\n  Matched observations:")
        print(f"    On-target (same trait):  {len(matched_on_target)}")
        print(f"    Off-target (diff trait): {len(matched_off_target)}")
        
        # ─── Overall ICC (on-target only — comparing same construct) ───
        h_scores = np.array([m["human"] for m in matched_on_target])
        l_scores = np.array([m["llm"] for m in matched_on_target])
        
        print(f"\n  ── OVERALL (on-target, n={len(matched_on_target)}) ──")
        
        icc = compute_icc_2_1(h_scores, l_scores)
        pearson = compute_pearson(h_scores, l_scores)
        ba = bland_altman(h_scores, l_scores)
        
        print(f"    ICC(2,1) agreement:    {icc['icc_agreement']:.3f}  [{icc['ci_lower']:.3f}, {icc['ci_upper']:.3f}]  ({interpret_icc(icc['icc_agreement'])})")
        print(f"    ICC(2,1) consistency:  {icc['icc_consistency']:.3f}")
        print(f"    Pearson r:             {pearson['r']:.3f}  (p={pearson['p']:.4f})  [{pearson['ci_lower']:.3f}, {pearson['ci_upper']:.3f}]")
        print(f"    Human mean: {icc['human_mean']:.2f}  |  LLM mean: {icc['llm_mean']:.2f}  |  Bias: {ba['bias']:+.3f}")
        print(f"    Bland-Altman LoA: [{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}]")
        
        judge_results = {
            "judge_model": judge_model,
            "overall": {
                "icc": icc,
                "pearson": pearson,
                "bland_altman": ba,
                "interpretation": interpret_icc(icc["icc_agreement"]),
            },
        }
        
        # ─── Per-trait breakdown ──────────────────────────────────────
        print(f"\n  ── PER TRAIT ──")
        print(f"    {'Trait':<40s} {'n':>3s}  {'ICC_agr':>7s}  {'r':>6s}  {'p':>7s}  {'Bias':>5s}  {'Interp'}")
        
        trait_results = {}
        for trait in sorted(by_trait.keys()):
            entries = by_trait[trait]
            hs = np.array([e["human"] for e in entries])
            ls = np.array([e["llm"] for e in entries])
            
            t_icc = compute_icc_2_1(hs, ls)
            t_r = compute_pearson(hs, ls)
            t_ba = bland_altman(hs, ls)
            
            interp = interpret_icc(t_icc["icc_agreement"])
            r_val = t_r["r"] if not np.isnan(t_r["r"]) else 0.0
            p_val = t_r["p"] if not np.isnan(t_r["p"]) else 1.0
            
            print(f"    {trait:<40s} {t_icc['n']:>3d}  {t_icc['icc_agreement']:>7.3f}  {r_val:>6.3f}  {p_val:>7.4f}  {t_ba['bias']:>+5.2f}  {interp}")
            
            trait_results[trait] = {
                "icc": t_icc,
                "pearson": t_r,
                "bland_altman": t_ba,
                "interpretation": interp,
            }
        
        judge_results["per_trait"] = trait_results
        
        # ─── Per-model breakdown ──────────────────────────────────────
        print(f"\n  ── PER MODEL ──")
        print(f"    {'Model':<12s} {'n':>3s}  {'ICC_agr':>7s}  {'r':>6s}  {'p':>7s}  {'Bias':>5s}  {'Interp'}")
        
        model_results = {}
        for model in sorted(by_model.keys()):
            entries = by_model[model]
            hs = np.array([e["human"] for e in entries])
            ls = np.array([e["llm"] for e in entries])
            
            m_icc = compute_icc_2_1(hs, ls)
            m_r = compute_pearson(hs, ls)
            m_ba = bland_altman(hs, ls)
            
            interp = interpret_icc(m_icc["icc_agreement"])
            r_val = m_r["r"] if not np.isnan(m_r["r"]) else 0.0
            p_val = m_r["p"] if not np.isnan(m_r["p"]) else 1.0
            
            print(f"    {model:<12s} {m_icc['n']:>3d}  {m_icc['icc_agreement']:>7.3f}  {r_val:>6.3f}  {p_val:>7.4f}  {m_ba['bias']:>+5.2f}  {interp}")
            
            model_results[model] = {
                "icc": m_icc,
                "pearson": m_r,
                "bland_altman": m_ba,
                "interpretation": interp,
            }
        
        judge_results["per_model"] = model_results
        
        # ─── Score distribution comparison ────────────────────────────
        print(f"\n  ── SCORE DISTRIBUTION ──")
        print(f"    Score  Human_n  LLM_n")
        for s in range(1, 8):
            h_n = np.sum(h_scores == s)
            l_n = np.sum(l_scores == s)
            print(f"      {s}      {h_n:>3d}    {l_n:>3d}")
        
        # ─── Cohen's weighted kappa (ordinal) ─────────────────────────
        # Quadratic-weighted kappa for ordinal agreement
        from sklearn.metrics import cohen_kappa_score
        try:
            kappa = cohen_kappa_score(h_scores.astype(int), l_scores.astype(int), weights="quadratic")
            print(f"\n    Quadratic-weighted kappa: {kappa:.3f}")
            judge_results["weighted_kappa"] = round(kappa, 3)
        except Exception:
            print(f"\n    Weighted kappa: could not compute")
            judge_results["weighted_kappa"] = None
        
        # ─── Exact and adjacent agreement ─────────────────────────────
        exact = np.sum(h_scores == l_scores) / len(h_scores)
        adjacent = np.sum(np.abs(h_scores - l_scores) <= 1) / len(h_scores)
        print(f"    Exact agreement:    {exact:.1%}")
        print(f"    Adjacent (±1):      {adjacent:.1%}")
        judge_results["exact_agreement"] = round(exact, 3)
        judge_results["adjacent_agreement"] = round(adjacent, 3)
        
        all_results[judge_name] = judge_results
    
    # ─── CROSS-JUDGE COMPARISON ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CROSS-JUDGE COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Judge':<12s} {'ICC_agr':>8s} {'ICC_con':>8s} {'r':>6s} {'κ_w':>6s} {'Bias':>6s} {'Exact%':>7s} {'±1%':>7s}")
    
    for jname, jr in all_results.items():
        ov = jr["overall"]
        print(f"  {jname:<12s} {ov['icc']['icc_agreement']:>8.3f} {ov['icc']['icc_consistency']:>8.3f} {ov['pearson']['r']:>6.3f} {jr.get('weighted_kappa',0):>6.3f} {ov['bland_altman']['bias']:>+6.2f} {jr['exact_agreement']:>7.1%} {jr['adjacent_agreement']:>7.1%}")
    
    # ─── Save results ────────────────────────────────────────────────
    output_path = RESULTS_DIR / "human_llm_icc_results.json"
    
    # Clean up numpy types for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(i) for i in obj]
        return obj
    
    save_data = {
        "analysis": "Human vs LLM Judge ICC",
        "human_rater": "Rater 1 (clinical psychologist)",
        "n_responses": 120,
        "scale": "1-7 anchored",
        "results": clean_for_json(all_results),
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
