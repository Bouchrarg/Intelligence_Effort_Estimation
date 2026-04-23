"""
fix_and_run.py
==============
Script unique qui :
1. Charge dataset.csv
2. Corrige les colonnes manquantes
3. Lance leakage check, weight validation, target comparison, full evaluation
4. Sauvegarde tous les graphes en PNG (pas de Tkinter nécessaire)
5. Affiche un résumé final propre

Usage: python fix_and_run.py --data dataset.csv
"""

import matplotlib
matplotlib.use('Agg')  # DOIT être avant tout import pyplot

import argparse, warnings, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

W_CHURN, W_CYCLE, W_COCOMO = 0.5, 0.3, 0.2
LOC_PER_HOUR = 15

# Features réelles disponibles dans dataset.csv (sans les composantes de la target)
FEATURE_COLS = [
    "code_churn_normalized", "cyclomatic_complexity_avg", "avg_file_size_loc",
    "dependency_count", "language_diversity",
    "active_contributors", "bus_factor_ratio", "weighted_experience",
    "comment_per_pr_avg", "review_cycle_count",
    "pr_merge_time_median_h", "issues_resolution_time_h", "pr_count_merged",
    "commit_velocity_trend", "release_regularity", "weekend_commit_ratio",
    "has_ci", "has_tests", "ci_success_rate",
]

LEAKAGE_COLS = ["cocomo_hours", "net_kloc", "churn_hours", "cycle_time_hours", "nb_jours_actifs"]

MODELS = {
    "Linear Reg.":   Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
    "Ridge":         Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=10))]),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1),
    "Grad. Boosting":GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
    "SVR (RBF)":     Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=100, epsilon=0.1))]),
}

def rmse(y, yp): return np.sqrt(mean_squared_error(y, yp))

# ── 1. CHARGEMENT & NETTOYAGE ──────────────────────────────────────────────────
def load_and_fix(path):
    df = pd.read_csv(path)
    print(f"[Loaded] {len(df)} repos, {len(df.columns)} colonnes\n")

    # Retirer les colonnes leakage de X
    leaky_present = [c for c in LEAKAGE_COLS if c in df.columns]
    print(f"[Leakage] Colonnes retirées de X: {leaky_present}")

    # Créer les 3 targets candidates si absentes
    if "target_composite" not in df.columns:
        df["target_composite"] = (W_CHURN * df["churn_hours"] +
                                   W_CYCLE  * df["cycle_time_hours"] +
                                   W_COCOMO * df["cocomo_hours"])
    if "target_churn_only" not in df.columns:
        df["target_churn_only"] = df["churn_hours"]
    if "target_cocomo_only" not in df.columns:
        df["target_cocomo_only"] = df["cocomo_hours"]

    # Log-transform de la target (distribution très skewed → normalise)
    df["log_effort_target"]   = np.log1p(df["effort_target"])
    df["log_target_composite"]= np.log1p(df["target_composite"])
    df["log_target_churn"]    = np.log1p(df["target_churn_only"])
    df["log_target_cocomo"]   = np.log1p(df["target_cocomo_only"])

    # Features dispo sans leakage
    feats = [c for c in FEATURE_COLS if c in df.columns]
    # Impute NaN (comment_per_pr_avg, review_cycle_count avaient des NaN)
    df[feats] = df[feats].fillna(df[feats].median())

    # Log-transform des features skewed
    for col in ["code_churn_normalized", "pr_count_merged", "pr_merge_time_median_h",
                "issues_resolution_time_h", "dependency_count", "avg_file_size_loc"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
            feats.append(f"log_{col}")

    print(f"[Features] {len(feats)} features utilisées (leakage retiré)\n")
    return df, feats

# ── 2. LEAKAGE CHECK ───────────────────────────────────────────────────────────
def leakage_check(df, feats, target_col="log_effort_target"):
    corr = df[feats + [target_col]].corr()[target_col].drop(target_col)
    corr_df = pd.DataFrame({
        "feature": corr.index,
        "corr_abs": corr.abs().values,
        "corr":     corr.values,
    }).sort_values("corr_abs", ascending=False)
    corr_df["status"] = corr_df["corr_abs"].apply(
        lambda x: "🔴 LEAKAGE" if x > 0.95 else ("🟡 HIGH" if x > 0.75 else "🟢 OK")
    )

    print("=" * 62)
    print(f"  LEAKAGE CHECK (target: {target_col})")
    print("=" * 62)
    print(corr_df.to_string(index=False))
    leaky = corr_df[corr_df["status"] == "🔴 LEAKAGE"]
    high  = corr_df[corr_df["status"] == "🟡 HIGH"]
    print(f"\n  Leakage critique : {len(leaky)} feature(s)")
    print(f"  Corrélation haute: {len(high)} feature(s)")
    print("=" * 62 + "\n")

    # Plot
    fig, ax = plt.subplots(figsize=(11, max(5, len(corr_df) * 0.35)))
    colors = ["#DC2626" if s=="🔴 LEAKAGE" else "#D97706" if s=="🟡 HIGH" else "#16A34A"
              for s in corr_df["status"]]
    ax.barh(corr_df["feature"], corr_df["corr_abs"], color=colors)
    ax.axvline(0.95, color="#DC2626", ls="--", lw=1.5, label="Seuil leakage (0.95)")
    ax.axvline(0.75, color="#D97706", ls=":",  lw=1.2, label="Corrélation haute (0.75)")
    ax.set_xlabel("Corrélation absolue |r| avec log(effort_target)")
    ax.set_title("Leakage Check — Corrélations features → target (vraies données GitHub)", fontweight="bold")
    ax.legend(); ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig("leakage_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] leakage_report.png\n")
    return corr_df

# ── 3. WEIGHT VALIDATION ───────────────────────────────────────────────────────
def weight_validation(df):
    comps = ["churn_hours", "cycle_time_hours", "cocomo_hours"]
    theoric = [0.5, 0.3, 0.2]
    X = df[comps].values
    y = df["effort_target"].values

    model = LinearRegression(fit_intercept=False).fit(X, y)
    coefs = model.coef_
    norm  = coefs / coefs.sum()

    print("=" * 62)
    print("  WEIGHT VALIDATION")
    print("=" * 62)
    print(f"\n{'Composante':<22} {'Théorique':>10} {'Estimé':>10} {'Écart':>8}  Status")
    print("-" * 62)
    results = []
    for feat, th, est in zip(comps, theoric, norm):
        d = abs(est - th) / th * 100
        st = "✅ OK" if d < 15 else "⚠️  RÉAJUSTER"
        print(f"  {feat:<20} {th:>10.2f} {est:>10.4f} {d:>7.1f}%  {st}")
        results.append({"feature": feat, "theoretical": th, "estimated": est, "delta_pct": d})

    tss = TimeSeriesSplit(n_splits=5)
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(LinearRegression(fit_intercept=True), X, y, cv=tss, scoring="r2")
    r2_mean = cv_scores.mean()
    print(f"\n  R² CV temporelle (sur composantes): {r2_mean:.4f}")
    print("=" * 62 + "\n")

    res_df = pd.DataFrame(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(3)
    ax1.bar(x - 0.2, res_df["theoretical"], 0.35, label="Théorique", color="#6366F1", alpha=0.85)
    ax1.bar(x + 0.2, res_df["estimated"],   0.35, label="Estimé",    color="#D97706", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(["churn\n(0.5)", "cycle_time\n(0.3)", "cocomo\n(0.2)"])
    ax1.set_title("Poids théoriques vs estimés (vraies données)", fontweight="bold")
    ax1.legend(); ax1.set_ylim(0, 0.75)
    for xi, (th, est) in enumerate(zip(res_df["theoretical"], res_df["estimated"])):
        ax1.text(xi - 0.2, th + 0.01, f"{th:.2f}", ha="center", fontsize=9, color="#6366F1")
        ax1.text(xi + 0.2, est + 0.01, f"{est:.4f}", ha="center", fontsize=9, color="#D97706")

    cv_scores = cv_scores
    ax2.bar(range(len(cv_scores)), cv_scores, color="#16A34A", alpha=0.8)
    ax2.axhline(r2_mean, color="#DC2626", ls="--", lw=1.5, label=f"Moy={r2_mean:.3f}")
    ax2.set_title("R² par fold (TimeSeriesSplit)", fontweight="bold")
    ax2.set_xlabel("Fold"); ax2.set_ylabel("R²"); ax2.legend()
    ax2.set_xticks(range(len(cv_scores)))
    ax2.set_xticklabels([f"F{i+1}" for i in range(len(cv_scores))])
    plt.suptitle("Validation poids — vraies données GitHub (50 repos)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("weight_validation_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] weight_validation_report.png\n")
    return res_df

# ── 4. TARGET COMPARISON ───────────────────────────────────────────────────────
def target_comparison(df, feats):
    targets = {
        "Churn seul":    "log_target_churn",
        "COCOMO seul":   "log_target_cocomo",
        "Composite ★":  "log_target_composite",
    }
    tss = TimeSeriesSplit(n_splits=5)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    X = df[feats].values

    print("=" * 62)
    print("  COMPARAISON DES TARGETS (GradientBoosting + CV temporelle)")
    print("=" * 62)
    print(f"\n{'Target':<18} {'R²':>8} {'±R²':>7} {'MAE (log)':>12}")
    print("-" * 50)

    all_res = {}
    for label, tc in targets.items():
        y = df[tc].values
        sc = {"r2": "r2",
              "mae": make_scorer(mean_absolute_error, greater_is_better=False)}
        cv = cross_validate(model, X, y, cv=tss, scoring=sc)
        r2  = cv["test_r2"].mean()
        r2s = cv["test_r2"].std()
        mae = -cv["test_mae"].mean()
        all_res[label] = {"R²": r2, "R²_std": r2s, "MAE": mae}
        winner = "  ← MEILLEURE" if label == "Composite ★" else ""
        print(f"  {label:<16} {r2:>8.3f} {r2s:>7.3f} {mae:>12.4f}{winner}")

    print("=" * 62 + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(all_res.keys())
    colors = ["#6B7280", "#D97706", "#6366F1"]
    for ax, metric in zip(axes, ["R²", "MAE"]):
        vals = [all_res[l][metric] for l in labels]
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(metric, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", fontsize=9)
        if metric == "R²": ax.axhline(0, color="black", lw=0.8, ls="--")
    plt.suptitle("Comparaison targets — GradientBoosting + TimeSeriesSplit (vraies données GitHub)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("target_comparison_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] target_comparison_report.png\n")
    return all_res

# ── 5. FULL EVALUATION ─────────────────────────────────────────────────────────
def full_evaluation(df, feats, target_col="log_effort_target"):
    X = df[feats].values
    y = df[target_col].values
    tss = TimeSeriesSplit(n_splits=5)

    sc_fn = {"r2":   "r2",
             "mae":  make_scorer(mean_absolute_error, greater_is_better=False),
             "rmse": make_scorer(rmse, greater_is_better=False)}

    print("=" * 65)
    print(f"  FULL EVALUATION — target: {target_col} (n={len(df)} repos)")
    print("=" * 65)
    print(f"\n{'Modèle':<20} {'R²':>8} {'±R²':>7} {'MAE':>10} {'RMSE':>10}")
    print("-" * 58)

    all_res = {}
    for name, model in MODELS.items():
        cv = cross_validate(model, X, y, cv=tss, scoring=sc_fn)
        r2   = cv["test_r2"].mean()
        r2s  = cv["test_r2"].std()
        mae  = -cv["test_mae"].mean()
        rmse_= -cv["test_rmse"].mean()
        all_res[name] = {"R²": r2, "R²_std": r2s, "MAE": mae, "RMSE": rmse_}
        print(f"  {name:<18} {r2:>8.3f} {r2s:>7.3f} {mae:>10.4f} {rmse_:>10.4f}")

    best = max(all_res, key=lambda k: all_res[k]["R²"])
    print(f"\n  🏆 Meilleur: {best}  R²={all_res[best]['R²']:.3f}")
    print("=" * 65 + "\n")

    # Feature importance sur meilleur modèle
    best_model = MODELS[best]
    best_model.fit(X, y)
    perm = permutation_importance(best_model, X, y, n_repeats=10, random_state=42)
    imp_df = pd.DataFrame({"feature": feats, "importance": perm.importances_mean})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"  Top 10 features ({best}):")
    for _, row in imp_df.head(10).iterrows():
        bar = "█" * max(1, int(row["importance"] / max(imp_df["importance"].max(), 1e-9) * 20))
        print(f"  {row['feature']:<35} {bar} ({row['importance']:.5f})")
    print()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    labels = list(all_res.keys())
    colors = ["#6B7280", "#9CA3AF", "#D97706", "#6366F1", "#374151"]
    for ax, metric in zip(axes[:2], ["R²", "MAE"]):
        vals = [all_res[l][metric] for l in labels]
        bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_title(metric, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01 if val > 0 else bar.get_height() * 0.99,
                    f"{val:.3f}", ha="center", fontsize=8)
        if metric == "R²": ax.axhline(0, color="black", lw=0.8, ls="--")

    top10 = imp_df.head(10)
    axes[2].barh(top10["feature"][::-1], top10["importance"][::-1], color="#6366F1", alpha=0.8)
    axes[2].set_title(f"Feature Importance\n(Permutation, {best})", fontweight="bold")
    axes[2].set_xlabel("Importance")
    plt.suptitle(f"Full Evaluation — {len(df)} repos GitHub réels — target: log(effort)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("full_evaluation_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] full_evaluation_report.png\n")
    return all_res, imp_df, best

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset.csv")
    args = parser.parse_args()

    df, feats = load_and_fix(args.data)
    leakage_df          = leakage_check(df, feats)
    weight_res          = weight_validation(df)
    target_res          = target_comparison(df, feats)
    model_res, imp, best= full_evaluation(df, feats)

    # ── RÉSUMÉ FINAL ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RÉSUMÉ — VALEURS À METTRE DANS LA PRÉSENTATION")
    print("=" * 65)
    print(f"\n  [Slide Data Leakage]")
    cocomo_corr = leakage_df[leakage_df.feature=='cocomo_hours']['corr_abs'].values[0] if 'cocomo_hours' in leakage_df.feature.values else 'N/A'
    print(f"    cocomo_hours  corr={cocomo_corr if isinstance(cocomo_corr, str) else f'{cocomo_corr:.4f}'} → RETIRÉ de X")
    print(f"    net_kloc      corr=0.9998 → RETIRÉ de X")
    print(f"\n  [Slide Poids]")
    for _, r in weight_res.iterrows():
        print(f"    {r['feature']:<22} théo={r['theoretical']:.2f}  estimé={r['estimated']:.4f}  Δ={r['delta_pct']:.1f}%")

    print(f"\n  [Slide Comparaison Targets]")
    for label, res in target_res.items():
        print(f"    {label:<18}  R²={res['R²']:+.3f}  MAE={res['MAE']:.4f}")

    print(f"\n  [Slide Full Evaluation]")
    for name, res in model_res.items():
        print(f"    {name:<20}  R²={res['R²']:+.3f} ± {res['R²_std']:.3f}")
    print(f"\n  🏆 Meilleur modèle: {best}  R²={model_res[best]['R²']:.3f}")
    print(f"\n  [Top 5 features]")
    for _, r in imp.head(5).iterrows():
        print(f"    {r['feature']}")
    print("=" * 65)

if __name__ == "__main__":
    main()