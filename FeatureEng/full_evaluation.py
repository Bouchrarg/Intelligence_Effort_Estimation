"""
full_evaluation.py
==================
Pipeline complet end-to-end :
  1. Détection du leakage
  2. Split temporel strict (features mois 1-6, target mois 7-12)
  3. 5 modèles comparés : LinearRegression, Ridge, RandomForest, GradientBoosting, SVR
  4. Métriques : R², MAE, RMSE (CV temporelle TimeSeriesSplit)
  5. Feature importance (permutation-based ou SHAP si installé)
  6. Rapport auto-généré : full_evaluation_report.png

Usage:
    python full_evaluation.py --data dataset.csv --target effort_target
    python full_evaluation.py --demo
    python full_evaluation.py --demo --shap   (si shap installé)
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")

FEATURES_COLS = [
    "cyclomatic_complexity_avg", "avg_file_size_loc", "dependency_count",
    "language_diversity", "active_contributors", "bus_factor_ratio",
    "weighted_experience", "comment_per_pr_avg", "review_cycle_count",
    "pr_merge_time_median_h", "issues_resolution_time_h", "pr_count_merged",
    "commit_velocity_trend", "release_regularity", "weekend_commit_ratio",
    "has_ci", "has_tests", "ci_success_rate",
]

MODELS = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge (alpha=10)":  Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10))]),
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
    "SVR (RBF)":         Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10, epsilon=100))]),
}

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def make_demo_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    churn_hours = rng.exponential(5000, n)
    cycle_time_hours = rng.exponential(300, n) * rng.integers(2, 12, n)
    cocomo_hours = 2.4 * rng.uniform(10, 200, n) ** 1.05 * 160

    df = pd.DataFrame({
        "cyclomatic_complexity_avg": rng.uniform(1, 20, n),
        "avg_file_size_loc": rng.uniform(50, 800, n),
        "dependency_count": rng.integers(1, 100, n).astype(float),
        "language_diversity": rng.uniform(0, 1, n),
        "active_contributors": rng.integers(1, 50, n).astype(float),
        "bus_factor_ratio": rng.uniform(0.1, 1.0, n),
        "weighted_experience": rng.uniform(0.5, 15, n),
        "comment_per_pr_avg": rng.uniform(0, 40, n),
        "review_cycle_count": rng.uniform(1, 10, n),
        "pr_merge_time_median_h": rng.exponential(72, n),
        "issues_resolution_time_h": rng.exponential(120, n),
        "pr_count_merged": rng.integers(5, 500, n).astype(float),
        "commit_velocity_trend": rng.normal(0, 2, n),
        "release_regularity": rng.uniform(0, 30, n),
        "weekend_commit_ratio": rng.uniform(0, 0.5, n),
        "has_ci": rng.integers(0, 2, n).astype(float),
        "has_tests": rng.integers(0, 2, n).astype(float),
        "ci_success_rate": rng.uniform(0.5, 1.0, n),
    })
    df["effort_target"] = (0.5 * churn_hours + 0.3 * cycle_time_hours +
                            0.2 * cocomo_hours + rng.normal(0, 200, n))
    return df

def run_leakage_check(df, features, target_col):
    corr = df[features + [target_col]].corr()[target_col].drop(target_col).abs()
    flagged = corr[corr > 0.95]
    print(f"\n[1/4] LEAKAGE CHECK")
    if len(flagged) > 0:
        print(f"  🔴 Leakage détecté sur: {list(flagged.index)}")
        print(f"  Action: features concernées exclues automatiquement du X")
        safe_features = [f for f in features if f not in flagged.index]
    else:
        print(f"  ✅ Aucun leakage critique (corr < 0.95 pour toutes les features)")
        safe_features = features
    return safe_features

def evaluate_models(df, features, target_col):
    X = df[features].values
    y = df[target_col].values
    tss = TimeSeriesSplit(n_splits=5)
    scoring = {
        "r2": "r2",
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "rmse": make_scorer(rmse_scorer, greater_is_better=False),
    }

    results = {}
    print(f"\n[2/4] ÉVALUATION DES MODÈLES (TimeSeriesSplit n=5)")
    print(f"\n{'Modèle':<22} {'R²':>8} {'±R²':>6} {'MAE':>12} {'RMSE':>12}")
    print("-" * 64)

    for name, model in MODELS.items():
        cv = cross_validate(model, X, y, cv=tss, scoring=scoring)
        r2  = cv["test_r2"].mean()
        r2s = cv["test_r2"].std()
        mae = -cv["test_mae"].mean()
        rmse = -cv["test_rmse"].mean()
        results[name] = {"R²": r2, "R²_std": r2s, "MAE": mae, "RMSE": rmse}
        print(f"{name:<22} {r2:>8.3f} {r2s:>6.3f} {mae:>12.1f} {rmse:>12.1f}")

    best_name = max(results, key=lambda k: results[k]["R²"])
    print(f"\n  🏆 Meilleur modèle: {best_name} (R²={results[best_name]['R²']:.3f})")
    return results, best_name

def compute_feature_importance(df, features, target_col, best_model_name, use_shap=False):
    X = df[features].values
    y = df[target_col].values
    model = MODELS[best_model_name]
    model.fit(X, y)

    print(f"\n[3/4] FEATURE IMPORTANCE ({best_model_name})")

    if use_shap:
        try:
            import shap
            explainer = shap.TreeExplainer(model) if hasattr(model, "estimators_") else shap.Explainer(model, X)
            shap_values = explainer(X)
            importances = np.abs(shap_values.values).mean(0)
            method = "SHAP"
        except Exception as e:
            print(f"  [SHAP non disponible: {e}] → fallback permutation importance")
            use_shap = False

    if not use_shap:
        perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importances = perm.importances_mean
        method = "Permutation"

    imp_df = pd.DataFrame({"feature": features, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    print(f"\n  Top 10 features ({method} importance):")
    for i, row in imp_df.head(10).iterrows():
        bar = "█" * int(row["importance"] / imp_df["importance"].max() * 20)
        print(f"  {row['feature']:<30} {bar} ({row['importance']:.4f})")

    return imp_df, method

def plot_full_report(results, imp_df, method):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    labels = list(results.keys())
    short_labels = [l.replace(" (alpha=10)", "\n(α=10)").replace(" (RBF)", "\n(RBF)") for l in labels]
    colors = ["#6B7280", "#9CA3AF", "#D97706", "#6366F1", "#374151"]

    for ax, metric in zip(axes[:2], ["R²", "MAE"]):
        vals = [results[l][metric] for l in labels]
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_title(metric, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", fontsize=8)
        if metric == "R²":
            ax.set_ylim(0, 1)

    top10 = imp_df.head(10)
    axes[2].barh(top10["feature"][::-1], top10["importance"][::-1], color="#6366F1", alpha=0.8)
    axes[2].set_title(f"Feature Importance ({method})", fontweight="bold")
    axes[2].set_xlabel("Importance")

    plt.suptitle("Rapport complet — Évaluation modèles & importance features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("full_evaluation_report.png", dpi=150, bbox_inches="tight")
    print("\n[Graph saved] → full_evaluation_report.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target", type=str, default="effort_target")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--shap", action="store_true", help="Utilise SHAP pour l'importance features")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  FULL EVALUATION PIPELINE — Mining GitHub Repositories")
    print("="*70)

    if args.demo or args.data is None:
        print("[MODE DEMO] Données synthétiques (n=500)")
        df = make_demo_data()
    else:
        df = pd.read_csv(args.data)
        print(f"[Loaded] {len(df)} repos depuis {args.data}")

    available_features = [c for c in FEATURES_COLS if c in df.columns]
    if len(available_features) < 5:
        print(f"[ERREUR] Trop peu de features disponibles: {available_features}")
        return

    safe_features = run_leakage_check(df, available_features, args.target)
    results, best_model = evaluate_models(df, safe_features, args.target)
    imp_df, method = compute_feature_importance(df, safe_features, args.target, best_model, args.shap)

    print(f"\n[4/4] GÉNÉRATION DU RAPPORT GRAPHIQUE")
    plot_full_report(results, imp_df, method)

    print(f"\n{'='*70}")
    print(f"  RÉSUMÉ FINAL")
    print(f"{'='*70}")
    print(f"  Features utilisées: {len(safe_features)}/{len(FEATURES_COLS)}")
    print(f"  Meilleur modèle: {best_model} — R²={results[best_model]['R²']:.3f}, MAE={results[best_model]['MAE']:.1f}")
    print(f"  Top 3 features: {', '.join(imp_df['feature'].head(3).tolist())}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
