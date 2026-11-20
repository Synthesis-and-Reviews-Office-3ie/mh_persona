"""Compute 95% CIs for Cohen's d values in Table 6."""
import json
import math
import numpy as np

# Table 6 values from the manuscript
# Each row: (trait, feature, d_plus3, d_minus3, n_per_group)
# n = 30 per group (10 scenarios x 3 models at each coefficient level)
TABLE6 = [
    ("Uncritical validation", "Questions", -0.61, +0.21),
    ("Uncritical validation", "Hedging", -0.49, +0.46),
    ("Emotional over-involvement", "Hedging", -0.54, +0.11),
    ("Emotional over-involvement", "Client focus", +0.16, -0.57),
    ("Boundary maintenance", "Questions", -0.51, +0.10),
    ("Boundary maintenance", "Safety referrals", +0.37, -0.13),
    ("Empathetic responsiveness", "Amplifiers", -0.15, -0.44),
]

N = 30  # per group (10 scenarios x 3 models)

def cohens_d_ci(d, n1, n2, alpha=0.05):
    """Compute CI for Cohen's d using the variance formula (Hedges & Olkin, 1985).
    
    SE(d) = sqrt(n1+n2)/(n1*n2) + d^2/(2*(n1+n2))
    """
    se = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    z = 1.96  # for 95% CI
    return (round(d - z * se, 2), round(d + z * se, 2))


print("Table 6 with 95% CIs (N=30 per group)")
print(f"{'Trait':<35s} {'Feature':<18s} {'d(+3)':<8s} {'95% CI':<18s} {'d(-3)':<8s} {'95% CI':<18s}")
print("-" * 105)

results = []
for trait, feature, d_plus, d_minus in TABLE6:
    ci_plus = cohens_d_ci(d_plus, N, N)
    ci_minus = cohens_d_ci(d_minus, N, N)
    print(f"{trait:<35s} {feature:<18s} {d_plus:>+6.2f}   [{ci_plus[0]:>+.2f}, {ci_plus[1]:>+.2f}]    {d_minus:>+6.2f}   [{ci_minus[0]:>+.2f}, {ci_minus[1]:>+.2f}]")
    results.append({
        "trait": trait,
        "feature": feature,
        "d_plus3": d_plus,
        "ci_plus3": list(ci_plus),
        "d_minus3": d_minus,
        "ci_minus3": list(ci_minus),
        "n_per_group": N,
    })

# Save
with open("results/cohens_d_table6_ci.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to results/cohens_d_table6_ci.json")
print(f"\nNote: All CIs are wide due to N={N}. Most cross zero, indicating")
print("that individual effects are not statistically significant at this sample size.")
print("The pattern across multiple features is more informative than any single d.")
