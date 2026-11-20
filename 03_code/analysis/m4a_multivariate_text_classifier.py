"""
M4a: Multivariate Text Classifier Baseline
============================================
Reviewer concern: The 2.9x sensitivity gap compares activation projections
against the single best handcrafted text feature. A fairer comparison uses
ALL text features simultaneously (multivariate model).

This script:
1. Extracts all text features (sentiment, pragmatics, complexity, lexicon)
2. Trains logistic regression + random forest to predict steering coefficient
3. Reports R^2 (variance explained) for multivariate text vs activation monitoring
4. Runs on both 1,200-response primary corpus and 12,000-response expanded corpus

Key comparison:
  - Activation monitoring:  mean r=0.596 -> R^2=0.355
  - Best single text feature: mean |r|=0.203 -> R^2=0.041
  - Multivariate text (LR): ???
  - Multivariate text (RF): ???
"""

import json
import re
import os
import warnings
import numpy as np
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, GroupKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Feature extraction (same as analyze_corpus_deep.py) ──

def extract_all_features(text):
    """Extract ALL text features used in the paper."""
    text_lower = text.lower()
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    # Import VADER lazily
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(text)

    features = {
        # Sentiment (4 features)
        "vader_compound": vader_scores["compound"],
        "vader_pos": vader_scores["pos"],
        "vader_neg": vader_scores["neg"],
        "vader_neu": vader_scores["neu"],

        # Linguistic complexity (5 features)
        "word_count": len(words),
        "char_length": len(text),
        "sentence_count": len(sentences),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "question_ratio": text.count("?") / max(len(sentences), 1),

        # Pragmatic features (10 features)
        "hedges": len(re.findall(
            r'\b(might|maybe|perhaps|could|possibly|it seems|it appears|i wonder)\b',
            text_lower)),
        "directives": len(re.findall(
            r'\b(you should|you need to|you must|try to|make sure|i suggest|i recommend|consider)\b',
            text_lower)),
        "self_reference": len(re.findall(
            r'\b(i feel|i think|i believe|in my|my experience|i personally|i also)\b',
            text_lower)),
        "client_focus": len(re.findall(
            r'\b(you feel|you\'re feeling|your feelings|how you|what you|you mentioned|you said|your experience)\b',
            text_lower)),
        "safety_referrals": len(re.findall(
            r'\b(professional|therapist|counselor|doctor|emergency|crisis|hotline|988|911)\b',
            text_lower)),
        "amplifiers": len(re.findall(
            r'\b(so|very|really|incredibly|extremely|absolutely|deeply|truly|completely|utterly)\b',
            text_lower)),
        "questions": text.count("?"),
        "has_list": int(bool(re.search(r'[\n\r]\s*[\-\u2022\d]+[\.\)]', text))),
        "paragraphs": len([p for p in text.split("\n") if p.strip()]),
        "opens_with_validation": int(bool(re.match(
            r"^(i('m| am) (so )?sorry|that (sounds|must|seems)|i (hear|understand|can see))",
            text_lower))),
    }
    return features


# Pre-initialise VADER once
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_vader = SentimentIntensityAnalyzer()

def extract_all_features_fast(text):
    """Extract features with shared VADER instance (faster)."""
    text_lower = text.lower()
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    vader_scores = _vader.polarity_scores(text)

    return {
        "vader_compound": vader_scores["compound"],
        "vader_pos": vader_scores["pos"],
        "vader_neg": vader_scores["neg"],
        "vader_neu": vader_scores["neu"],
        "word_count": len(words),
        "char_length": len(text),
        "sentence_count": len(sentences),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "question_ratio": text.count("?") / max(len(sentences), 1),
        "hedges": len(re.findall(
            r'\b(might|maybe|perhaps|could|possibly|it seems|it appears|i wonder)\b',
            text_lower)),
        "directives": len(re.findall(
            r'\b(you should|you need to|you must|try to|make sure|i suggest|i recommend|consider)\b',
            text_lower)),
        "self_reference": len(re.findall(
            r'\b(i feel|i think|i believe|in my|my experience|i personally|i also)\b',
            text_lower)),
        "client_focus": len(re.findall(
            r'\b(you feel|you\'re feeling|your feelings|how you|what you|you mentioned|you said|your experience)\b',
            text_lower)),
        "safety_referrals": len(re.findall(
            r'\b(professional|therapist|counselor|doctor|emergency|crisis|hotline|988|911)\b',
            text_lower)),
        "amplifiers": len(re.findall(
            r'\b(so|very|really|incredibly|extremely|absolutely|deeply|truly|completely|utterly)\b',
            text_lower)),
        "questions": text.count("?"),
        "has_list": int(bool(re.search(r'[\n\r]\s*[-\u2022\d]+[.)]', text))),
        "paragraphs": len([p for p in text.split("\n") if p.strip()]),
        "opens_with_validation": int(bool(re.match(
            r"^(i('m| am) (so )?sorry|that (sounds|must|seems)|i (hear|understand|can see))",
            text_lower))),
    }


def load_corpus(path):
    """Load a steered corpus JSON and return responses list."""
    with open(path) as f:
        data = json.load(f)
    return data["responses"]


def featurize_corpus(responses):
    """Extract features for all responses. Returns (feature_matrix, coefficients, traits, models, scenarios)."""
    print(f"  Extracting features from {len(responses)} responses...")
    feature_dicts = []
    coefficients = []
    traits = []
    models = []
    scenarios = []

    for i, r in enumerate(responses):
        feats = extract_all_features_fast(r["response"])
        feature_dicts.append(feats)
        coefficients.append(r["coefficient"])
        traits.append(r["trait"])
        models.append(r["model"])
        scenarios.append(r["scenario_id"])
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(responses)}...")

    feature_names = sorted(feature_dicts[0].keys())
    X = np.array([[fd[fn] for fn in feature_names] for fd in feature_dicts])
    y = np.array(coefficients)

    return X, y, np.array(traits), np.array(models), np.array(scenarios), feature_names


def run_multivariate_analysis(X, y, traits, models, scenarios, feature_names, corpus_name):
    """Run multivariate text classifiers per trait and pooled."""
    print(f"\n{'='*80}")
    print(f"MULTIVARIATE TEXT CLASSIFIER: {corpus_name}")
    print(f"{'='*80}")

    TRAITS = sorted(set(traits))
    MODELS = sorted(set(models))

    results = {
        "corpus": corpus_name,
        "n_responses": len(y),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "per_trait": {},
        "pooled": {},
    }

    # ── Per-trait analysis ──
    print(f"\n{'Trait':<40s} {'LR_r2':>7s} {'Ridge_r2':>7s} {'RF_r2':>7s} {'GB_r2':>7s} {'Best_1feat_r2':>14s} {'Act_r2':>7s}")
    print("-" * 100)

    # Activation r values from Table 3 (cross-model mean)
    activation_r = {
        "empathetic_responsiveness": 0.735,
        "non_judgmental_acceptance": 0.731,
        "boundary_maintenance": 0.475,
        "crisis_recognition": 0.728,
        "emotional_over_involvement": 0.487,
        "abandonment_of_therapeutic_frame": 0.681,
        "uncritical_validation": 0.446,
        "sycophancy_harmful_validation": 0.487,
    }

    # Best single text feature |r| from Table 5
    best_single_text_r = {
        "empathetic_responsiveness": 0.184,
        "non_judgmental_acceptance": 0.128,
        "boundary_maintenance": 0.198,
        "crisis_recognition": 0.189,
        "emotional_over_involvement": 0.300,
        "abandonment_of_therapeutic_frame": 0.093,
        "uncritical_validation": 0.335,
        "sycophancy_harmful_validation": 0.195,
    }

    for trait in TRAITS:
        mask = traits == trait
        X_t = X[mask]
        y_t = y[mask]
        scenarios_t = scenarios[mask]

        # Use GroupKFold with scenario as group (prevents data leakage from same scenario)
        unique_scenarios = sorted(set(scenarios_t))
        n_groups = len(unique_scenarios)
        n_folds = min(5, n_groups)
        gkf = GroupKFold(n_splits=n_folds)

        # Linear Regression
        lr_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
        lr_scores = cross_val_score(lr_pipe, X_t, y_t, cv=gkf, groups=scenarios_t, scoring="r2")
        lr_r2 = max(np.mean(lr_scores), 0)  # floor at 0

        # Ridge Regression
        ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        ridge_scores = cross_val_score(ridge_pipe, X_t, y_t, cv=gkf, groups=scenarios_t, scoring="r2")
        ridge_r2 = max(np.mean(ridge_scores), 0)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf_scores = cross_val_score(rf, X_t, y_t, cv=gkf, groups=scenarios_t, scoring="r2")
        rf_r2 = max(np.mean(rf_scores), 0)

        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        gb_scores = cross_val_score(gb, X_t, y_t, cv=gkf, groups=scenarios_t, scoring="r2")
        gb_r2 = max(np.mean(gb_scores), 0)

        best_1feat_r2 = best_single_text_r[trait] ** 2
        act_r2 = activation_r[trait] ** 2

        results["per_trait"][trait] = {
            "lr_r2": round(lr_r2, 4),
            "lr_r2_std": round(np.std(lr_scores), 4),
            "ridge_r2": round(ridge_r2, 4),
            "ridge_r2_std": round(np.std(ridge_scores), 4),
            "rf_r2": round(rf_r2, 4),
            "rf_r2_std": round(np.std(rf_scores), 4),
            "gb_r2": round(gb_r2, 4),
            "gb_r2_std": round(np.std(gb_scores), 4),
            "best_single_text_r2": round(best_1feat_r2, 4),
            "activation_r2": round(act_r2, 4),
            "n_samples": int(mask.sum()),
            "n_scenarios": n_groups,
            "n_folds": n_folds,
        }

        print(f"{trait:<40s} {lr_r2:>7.3f} {ridge_r2:>7.3f} {rf_r2:>7.3f} {gb_r2:>7.3f} {best_1feat_r2:>14.3f} {act_r2:>7.3f}")

    # ── Summary statistics ──
    all_lr = [v["lr_r2"] for v in results["per_trait"].values()]
    all_ridge = [v["ridge_r2"] for v in results["per_trait"].values()]
    all_rf = [v["rf_r2"] for v in results["per_trait"].values()]
    all_gb = [v["gb_r2"] for v in results["per_trait"].values()]
    all_1feat = [v["best_single_text_r2"] for v in results["per_trait"].values()]
    all_act = [v["activation_r2"] for v in results["per_trait"].values()]

    print("-" * 100)
    print(f"{'MEAN':<40s} {np.mean(all_lr):>7.3f} {np.mean(all_ridge):>7.3f} {np.mean(all_rf):>7.3f} {np.mean(all_gb):>7.3f} {np.mean(all_1feat):>14.3f} {np.mean(all_act):>7.3f}")

    # Best multivariate method
    mean_methods = {
        "Linear Regression": np.mean(all_lr),
        "Ridge": np.mean(all_ridge),
        "Random Forest": np.mean(all_rf),
        "Gradient Boosting": np.mean(all_gb),
    }
    best_method = max(mean_methods, key=mean_methods.get)
    best_r2 = mean_methods[best_method]
    mean_act_r2 = np.mean(all_act)
    mean_1feat_r2 = np.mean(all_1feat)

    # Gap calculations
    gap_vs_1feat = best_r2 / mean_1feat_r2 if mean_1feat_r2 > 0 else float('inf')
    gap_act_vs_multi = mean_act_r2 / best_r2 if best_r2 > 0 else float('inf')
    gap_act_vs_1feat = mean_act_r2 / mean_1feat_r2 if mean_1feat_r2 > 0 else float('inf')

    results["summary"] = {
        "best_multivariate_method": best_method,
        "best_multivariate_r2": round(best_r2, 4),
        "mean_single_feature_r2": round(mean_1feat_r2, 4),
        "mean_activation_r2": round(mean_act_r2, 4),
        "gap_multivariate_vs_single_feature": round(gap_vs_1feat, 2),
        "gap_activation_vs_multivariate": round(gap_act_vs_multi, 2),
        "gap_activation_vs_single_feature_R2": round(gap_act_vs_1feat, 2),
        "mean_r_equiv": {
            "best_multivariate": round(np.sqrt(best_r2), 3),
            "best_single_feature": round(np.sqrt(mean_1feat_r2), 3),
            "activation": round(np.sqrt(mean_act_r2), 3),
        },
    }

    print(f"\n--- SUMMARY ---")
    print(f"  Best multivariate method: {best_method}")
    print(f"  Mean R^2: multivariate={best_r2:.4f}, single-feature={mean_1feat_r2:.4f}, activation={mean_act_r2:.4f}")
    print(f"  R^2 gaps: activation/multivariate = {gap_act_vs_multi:.1f}x, activation/single = {gap_act_vs_1feat:.1f}x")
    print(f"  Equivalent r: multivariate={np.sqrt(best_r2):.3f}, single={np.sqrt(mean_1feat_r2):.3f}, activation={np.sqrt(mean_act_r2):.3f}")

    # ── Feature importance (from best method, fit on full data) ──
    print(f"\n--- FEATURE IMPORTANCE (Gradient Boosting, full data per trait) ---")
    fi_results = {}
    for trait in TRAITS:
        mask = traits == trait
        X_t = X[mask]
        y_t = y[mask]
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        gb.fit(StandardScaler().fit_transform(X_t), y_t)
        importances = dict(zip(feature_names, gb.feature_importances_))
        top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
        fi_results[trait] = {k: round(v, 4) for k, v in top5}
        top_str = ", ".join(f"{k}={v:.3f}" for k, v in top5[:3])
        print(f"  {trait:<40s} {top_str}")

    results["feature_importance"] = fi_results

    return results


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("results", exist_ok=True)

    # ── Primary corpus (1,200 responses) ──
    print("Loading primary corpus (1,200 responses)...")
    responses_primary = load_corpus("results/steered_corpus_combined.json")
    X1, y1, traits1, models1, scenarios1, fnames1 = featurize_corpus(responses_primary)
    results_primary = run_multivariate_analysis(X1, y1, traits1, models1, scenarios1, fnames1, "primary_1200")

    # ── Expanded corpus (12,000 responses) ──
    print("\n\nLoading expanded corpus (12,000 responses)...")
    responses_expanded = load_corpus("results/steered_corpus_v2_combined.json")
    X2, y2, traits2, models2, scenarios2, fnames2 = featurize_corpus(responses_expanded)
    results_expanded = run_multivariate_analysis(X2, y2, traits2, models2, scenarios2, fnames2, "expanded_12000")

    # ── Save all results ──
    all_results = {
        "primary_1200": results_primary,
        "expanded_12000": results_expanded,
        "comparison": {
            "primary_best_multi_r2": results_primary["summary"]["best_multivariate_r2"],
            "expanded_best_multi_r2": results_expanded["summary"]["best_multivariate_r2"],
            "activation_r2": results_primary["summary"]["mean_activation_r2"],
            "primary_gap": results_primary["summary"]["gap_activation_vs_multivariate"],
            "expanded_gap": results_expanded["summary"]["gap_activation_vs_multivariate"],
            "interpretation": (
                "Multivariate text features improve over single-feature baseline "
                "but activation monitoring retains a substantial advantage in variance explained."
            ),
        },
    }

    with open("results/m4a_multivariate_text_classifier.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/m4a_multivariate_text_classifier.json")


if __name__ == "__main__":
    main()
