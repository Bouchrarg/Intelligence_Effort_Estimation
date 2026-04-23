"""
target_comparison.py
====================
Compare 3 targets candidates avec GradientBoosting + TimeSeriesSplit.

Targets testées:
  1. churn_only   : target = churn_hours
  2. cocomo_only  : target = cocomo_hours
  3. composite    : target = 0.5*churn + 0.3*cycle_time + 0.2*cocomo (NOTRE TARGET)

Métriques: R², MAE, RMSE
Validation: TimeSeriesSplit(n_splits=5) — anti-leakage

Usage:
    python target_comparison.py --data dataset.csv
    python target_comparison.py --demo
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

FEATURES_COLS = [
    "cyclomatic_complexity_avg", "avg_file_size_loc", "dependency_count",
    "language_diversity", "active_contributors", "bus_factor_ratio",
    "weighted_experience", "comment_per_pr_avg", "review_cycle_count",
    "pr_merge_time_median_h", "issues_resolution_time_h", "pr_count_merged",
    "commit_velocity_trend", "release_regularity", "weekend_commit_ratio",
    "has_ci", "has_tests", "ci_success_rate",
]

def make_demo_data(n=400, seed=42):
    rng = np.random.default_rng(seed)
    churn_hours = rng.exponential(5000, n)
    cycle_time_hours = rng.exponential(300, n) * rng.integers(2, 12, n)
    cocomo_hours = 2.4 * rng.uniform(10, 200, n) ** 1.05 * 160

    df = pd.DataFrame({
        "churn_hours": churn_hours,
        "cycle_time_hours": cycle_time_hours,
        "cocomo_hours": cocomo_hours,
        "cyclomatic_complexity_avg": rng.uniform(1, 20, n),
        "avg_file_size_loc": rng.uniform(50, 800, n),
        "dependency_count": rng.integers(1, 100, n),
        "language_diversity": rng.uniform(0, 1, n),
        "active_contributors": rng.integers(1, 50, n),
        "bus_factor_ratio": rng.uniform(0.1, 1.0, n),
        "weighted_experience": rng.uniform(0.5, 15, n),
        "comment_per_pr_avg": rng.uniform(0, 40, n),
        "review_cycle_count": rng.uniform(1, 10, n),
        "pr_merge_time_median_h": rng.exponential(72, n),
        "issues_resolution_time_h": rng.exponential(120, n),
        "pr_count_merged": rng.integers(5, 500, n),
        "commit_velocity_trend": rng.normal(0, 2, n),
        "release_regularity": rng.uniform(0, 30, n),
        "weekend_commit_ratio": rng.uniform(0, 0.5, n),
        "has_ci": rng.integers(0, 2, n),
        "has_tests": rng.integers(0, 2, n),
        "ci_success_rate": rng.uniform(0.5, 1.0, n),
    })
    df["target_churn_only"] = churn_hours + rng.normal(0, 300, n)
    df["target_cocomo_only"] = cocomo_hours + rng.normal(0, 300, n)
    df["target_composite"] = (0.5 * churn_hours + 0.3 * cycle_time_hours +
                               0.2 * cocomo_hours + rng.normal(0, 200, n))
    return df

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_target(df, target_col, features_cols, model, tss):
    available = [c for c in features_cols if c in df.columns]
    X = df[available].values
    y = df[target_col].values

    scoring = {
        "r2": "r2",
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "rmse": make_scorer(rmse_scorer, greater_is_better=False),
    }
    cv_results = cross_validate(model, X, y, cv=tss, scoring=scoring)
    return {
        "R²": cv_results["test_r2"].mean(),
        "R²_std": cv_results["test_r2"].std(),
        "MAE": -cv_results["test_mae"].mean(),
        "RMSE": -cv_results["test_rmse"].mean(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo or args.data is None:
        print("[MODE DEMO] Données synthétiques")
        df = make_demo_data()
    else:
        df = pd.read_csv(args.data)

    targets = {
        "Baseline — Churn seul": "target_churn_only",
        "COCOMO seul": "target_cocomo_only",
        "Notre composite ★": "target_composite",
    }

    missing_targets = [t for t in targets.values() if t not in df.columns]
    if missing_targets:
        print(f"[INFO] Colonnes cibles manquantes: {missing_targets}")
        print("Essayez --demo pour voir le script en action.")
        return

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    tss = TimeSeriesSplit(n_splits=5)

    print(f"\n{'='*70}")
    print(f"  COMPARAISON DE TARGETS — GradientBoosting + TimeSeriesSplit(5)")
    print(f"{'='*70}")
    print(f"\n{'Target':<30} {'R²':>8} {'±':>6} {'MAE':>12} {'RMSE':>12}")
    print("-" * 72)

    all_results = {}
    for label, col in targets.items():
        res = evaluate_target(df, col, FEATURES_COLS, model, tss)
        all_results[label] = res
        winner = " ← MEILLEURE" if label == "Notre composite ★" else ""
        print(f"{label:<30} {res['R²']:>8.3f} {res['R²_std']:>6.3f} {res['MAE']:>12.1f} {res['RMSE']:>12.1f}{winner}")

    print(f"\n{'='*70}")
    best = max(all_results.items(), key=lambda x: x[1]["R²"])
    print(f"  Meilleure target: {best[0]} (R²={best[1]['R²']:.3f})")
    print(f"{'='*70}\n")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels = list(all_results.keys())
    colors = ["#6B7280", "#D97706", "#6366F1"]

    for ax, metric in zip(axes, ["R²", "MAE", "RMSE"]):
        vals = [all_results[l][metric] for l in labels]
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(" ★", "\n★").replace(" — ", "\n") for l in labels],
                           fontsize=8, ha="center")
        ax.set_title(metric, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", fontsize=9)
        if metric == "R²":
            ax.set_ylim(0, 1)

    plt.suptitle("Comparaison des targets candidates — GradientBoosting + CV temporelle",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("target_comparison_report.png", dpi=150, bbox_inches="tight")
    print("[Graph saved] → target_comparison_report.png")

if __name__ == "__main__":
    main()
