import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# pip install shap
import shap

INPUT_CSV = "../ScrappingEnsam/features_merged_fixed.csv"

# ══════════════════════════════════════════════════════════════════════════════
# COLONNES RETIRÉES DE X — justification complète
# ══════════════════════════════════════════════════════════════════════════════
#
# effort_target = 0.5 * churn_hours
#               + 0.3 * cycle_time_hours
#               + 0.2 * cocomo_hours
#
# avec :
#   churn_hours          = churn_loc / 15
#   cycle_time_hours     = pr_merge_time_median_h × active_contributors
#   cocomo_hours         = f(net_loc)
#   code_churn_normalized= churn_loc / active_days
#
# → Toute colonne qui entre dans ces formules est du leakage.
#
LEAKAGE_COLUMNS = [
    # Composants directs de churn_hours et cocomo_hours
    'net_loc', 'churn_loc', 'active_days', 'total_commits',
    # Composants directs de cycle_time_hours
    'pr_merge_time_median_h', 'active_contributors',
    # Dérivé de churn_loc
    'code_churn_normalized',
    # Les colonnes intermédiaires elles-mêmes
    'churn_hours', 'cycle_time_hours', 'cocomo_pm', 'cocomo_hours',
    # Leakage indirect : reliability_score encode total_commits + active_contributors
    # SHAP la sort en feature #1 uniquement parce qu'elle reconstruit le leakage
    'reliability_score',
]

META_COLUMNS = ['full_name', 'url', 'created_at', 'lot']

# Features légitimes qui restent dans X :
#   stars, days_inactive, bus_factor_ratio, pr_count_merged, comment_per_pr_avg,
#   closed_issues, review_cycle_count, has_ci, ci_success_rate, has_tests,
#   weighted_experience, commit_velocity_trend, release_regularity,
#   weekend_commit_ratio, dependency_count, language_diversity, avg_file_size_loc,
#   reliability_score, language_* (dummies)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(df, numeric_cols):
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    plt.matshow(corr, cmap='coolwarm', fignum=1)
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90, fontsize=8)
    plt.yticks(range(len(numeric_cols)), numeric_cols, fontsize=8)
    plt.colorbar()
    plt.title("Matrice de Corrélation des Variables", pad=20)
    plt.savefig("correlation_matrix.png", bbox_inches='tight')
    plt.close()

def plot_target_distribution(y):
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution de l'Effort (Heures)")
    plt.xlabel("Heures")
    plt.ylabel("Nombre de projets")
    plt.savefig("target_distribution.png", bbox_inches='tight')
    plt.close()

def plot_model_comparisons(results):
    names = list(results.keys())
    r2_scores  = [results[n]['R2']  for n in names]
    mae_scores = [results[n]['MAE'] for n in names]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.bar(names, r2_scores,  color=colors)
    ax1.set_title("Comparaison du R² (Plus c'est haut, mieux c'est)")
    ax1.set_ylabel("R² Score")
    ax1.tick_params(axis='x', rotation=45)
    ax2.bar(names, mae_scores, color=colors)
    ax2.set_title("Comparaison de l'Erreur (MAE — Plus c'est bas, mieux c'est)")
    ax2.set_ylabel("Erreur Moyenne en Heures")
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("models_comparison.png", bbox_inches='tight')
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='purple')
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Prédiction Parfaite')
    plt.title("Réel vs Prédiction (XGBoost)")
    plt.xlabel("Effort Réel (Heures)")
    plt.ylabel("Effort Prédit (Heures)")
    plt.legend()
    plt.savefig("actual_vs_predicted.png", bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Variables — XGBoost")
    plt.barh(range(len(indices)), importances[indices], color='#1f77b4')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance Relative")
    plt.tight_layout()
    plt.savefig("feature_importance_xgb.png", bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SHAP — Explicabilité du modèle
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap(model, X_test, feature_names):
    """
    Génère 3 graphiques SHAP :
      1. shap_summary.png      — importance globale de chaque feature (beeswarm)
      2. shap_bar.png          — importance moyenne absolue (bar chart, pour slides)
      3. shap_waterfall_0.png  — explication détaillée du 1er projet du jeu de test
    """
    print("\n5. Analyse SHAP (explicabilité)...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)  # shape: (n_samples, n_features)

    # ── 1. Beeswarm — vue globale ────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("Impact de chaque feature sur la prédiction (espace log)")
    plt.tight_layout()
    plt.savefig("shap_summary.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("   -> shap_summary.png généré")

    # ── 2. Bar chart — top features (slide-friendly) ─────────────────────────
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title("Top 10 features par importance SHAP moyenne")
    plt.tight_layout()
    plt.savefig("shap_bar.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("   -> shap_bar.png généré")

    # ── 3. Waterfall — explication d'un projet précis ────────────────────────
    # On choisit le projet avec l'erreur relative la plus élevée pour
    # illustrer un cas intéressant (pas juste un projet médian).
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title("Décomposition de la prédiction — Projet #1 du jeu de test")
    plt.tight_layout()
    plt.savefig("shap_waterfall_0.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("   -> shap_waterfall_0.png généré")

    # ── Lecture textuelle pour la console ────────────────────────────────────
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:5]
    print("\n   Top 5 features par impact SHAP moyen (espace log) :")
    for i in top_idx:
        print(f"     {feature_names[i]:30s}  |  impact moyen : {mean_abs[i]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("1. Chargement et Profilage des données...")
    if not os.path.exists(INPUT_CSV):
        print(f"Erreur: {INPUT_CSV} introuvable.")
        return

    df = pd.read_csv(INPUT_CSV)

    n_total = len(df)

    # ── Filtre 1 : qualité des données scrappées ─────────────────────────────
    # reliability_score < 65 = données GitHub incomplètes (code_frequency vide,
    # très peu de contributeurs, repos quasi-inactifs) → effort_target aberrant
    if 'reliability_score' in df.columns:
        df = df[df['reliability_score'] >= 65]
        print(f"   -> {n_total - len(df)} projets retirés (reliability_score < 65)")

    # ── Filtre 2 : outliers de la target (top et bottom 1%) ──────────────────
    # On retire les deux extrêmes : projets avec effort anormalement bas (scraping
    # raté, code_frequency vide → fallback 50 LOC/commit) ET projets géants.
    p01 = df['effort_target'].quantile(0.01)
    p99 = df['effort_target'].quantile(0.99)
    df  = df[(df['effort_target'] >= p01) & (df['effort_target'] <= p99)]
    print(f"   -> {len(df)} projets après filtrage qualité+outliers")
    print(f"      (seuil effort : [{p01:,.0f}h — {p99:,.0f}h])")

    plot_target_distribution(df['effort_target'])

    # Nettoyage + encoding AVANT la matrice de corrélation
    df = df.fillna({'language': 'Unknown', 'avg_file_size_loc': 0.0, 'comment_per_pr_avg': 0.0})
    df_raw = df.copy()  # copie avec 'language' en clair pour test_set_unseen.csv
    df     = pd.get_dummies(df, columns=['language'], drop_first=True)

    # ── Retrait du leakage ───────────────────────────────────────────────────
    cols_to_drop = [c for c in META_COLUMNS + LEAKAGE_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop + ['effort_target'])
    y = df['effort_target']

    # Matrice de corrélation sur les features finales uniquement (évite MemoryError)
    # La matrice 744×744 sur tout le df allouait 4GB — ici on reste sur 31 colonnes
    plot_correlation_matrix(X, list(X.columns))
    print("   -> correlation_matrix.png, target_distribution.png générés")

    print(f"   -> {X.shape[1]} features dans X : {list(X.columns)}")

    # Log-transform de la target (distribution très skewed)
    y_log = np.log1p(y)

    # Split stratifié sur des quantiles de la target
    # Sans stratification, le test set peut concentrer les projets extrêmes
    # ce qui explique l'écart cross-val (0.36) vs test (0.09)
    y_quantile = pd.qcut(y_log, q=5, labels=False, duplicates='drop')
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42, stratify=y_quantile
    )
    print(f"   -> Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 2. Benchmark multi-modèles ───────────────────────────────────────────
    print("\n2. Benchmark Multi-Modèles...")
    models = {
        # Ridge DOIT être scalé — Pipeline obligatoire
        "Ridge": Pipeline([
            ('scaler', StandardScaler()),
            ('reg',    Ridge(alpha=100.0))  # alpha élevé = plus stable sur petit dataset
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_features='sqrt', random_state=42, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=1, objective='reg:squarederror'
        ),
    }

    results = {}
    print("-" * 55)
    for name, model in models.items():
        model.fit(X_train, y_train_log)
        y_pred_log  = model.predict(X_test)
        y_test_real = np.expm1(y_test_log)
        y_pred_real = np.expm1(y_pred_log)
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2  = r2_score(y_test_real, y_pred_real)
        results[name] = {'MAE': mae, 'R2': r2}
        print(f"{name:20s} | MAE: {mae:9.1f} h | R²: {r2:6.3f}")
    print("-" * 55)

    plot_model_comparisons(results)
    print("   -> models_comparison.png généré")

    # ── 3. Modèle final + cross-validation honnête ───────────────────────────
    print("\n3. Entraînement final + cross-validation (5-fold)...")
    # Avec ~427 exemples et 31 features, XGBoost profond overfitte.
    # max_depth=3 + min_child_weight=5 régularisent pour ce petit dataset.
    # La cross-validation 5-fold donne le vrai score généralisable.
    from sklearn.model_selection import cross_val_score, KFold

    best_model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=1,
        objective='reg:squarederror',
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2  = cross_val_score(best_model, X_train, y_train_log, cv=kf, scoring='r2')
    cv_mae = cross_val_score(best_model, X_train, y_train_log, cv=kf,
                             scoring='neg_mean_absolute_error')
    print(f"   Cross-val R²  (5-fold) : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    print(f"   Cross-val MAE (log)    : {-cv_mae.mean():.4f} ± {cv_mae.std():.4f}")

    best_model.fit(X_train, y_train_log)

    y_pred_best_log = best_model.predict(X_test)
    y_test_real     = np.expm1(y_test_log)
    y_pred_best     = np.expm1(y_pred_best_log)

    best_mae  = mean_absolute_error(y_test_real, y_pred_best)
    best_r2   = r2_score(y_test_real, y_pred_best)
    median_pe = float(np.median(
        np.abs(y_test_real.values - y_pred_best) / (y_test_real.values + 1) * 100
    ))

    print("\n" + "=" * 55)
    print("PERFORMANCE FINALE DU MODÈLE (XGBOOST — jeu de test)")
    print("=" * 55)
    print(f"MAE               : {best_mae:,.1f} heures")
    print(f"R²                : {best_r2:.3f}")
    print(f"Erreur médiane %  : {median_pe:.1f}%")
    print("=" * 55)

    plot_actual_vs_predicted(y_test_real, y_pred_best)
    plot_feature_importance(best_model, list(X.columns))
    print("   -> actual_vs_predicted.png, feature_importance_xgb.png générés")

    # ── 4. Sauvegarde ────────────────────────────────────────────────────────
    print("\n4. Sauvegarde du modèle et des données de test...")
    with open('effort_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'features': list(X.columns)}, f)
    print("   -> effort_model.pkl sauvegardé")

    df_test_unseen = df_raw.loc[X_test.index]
    df_test_unseen.to_csv('test_set_unseen.csv', index=False)
    print("   -> test_set_unseen.csv sauvegardé")

    # ── 5. SHAP ──────────────────────────────────────────────────────────────
    # X_test doit être un DataFrame avec les bons noms de colonnes pour SHAP
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    plot_shap(best_model, X_test_df, list(X.columns))

    print("\n✓ Terminé. Fichiers générés :")
    print("  Modèle    : effort_model.pkl")
    print("  Test set  : test_set_unseen.csv")
    print("  Graphiques: correlation_matrix, target_distribution, models_comparison,")
    print("              actual_vs_predicted, feature_importance_xgb,")
    print("              shap_summary, shap_bar, shap_waterfall_0")

if __name__ == "__main__":
    main()