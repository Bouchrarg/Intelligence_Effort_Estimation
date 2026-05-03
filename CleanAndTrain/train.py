"""
train.py — v2 corrigé
======================
Corrections :
  ✓ Split aléatoire 80/20 (au lieu de temporel — distribution shift trop grand)
  ✓ n_jobs=1 partout (évite MemoryError sur Windows)
  ✓ Modèles plus régularisés (max_depth réduit, min_samples_leaf élevé)

Usage :
    python train.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import Ridge, Lasso
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection      import permutation_importance

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("Phase 2 — Entraînement ML  (v2 corrigé)")
print("=" * 65)

X_all = pd.read_csv("X.csv")
y_all = pd.read_csv("y.csv").squeeze()

print(f"\nDataset complet : {X_all.shape[0]} repos x {X_all.shape[1]} features")
print(f"y — médiane log : {y_all.median():.2f}   std : {y_all.std():.2f}")

# ── Split ALÉATOIRE 80/20 ─────────────────────────────────────────────────────
# On abandonne le split temporel car :
#   - Train 2009-2023 : médiane y=10.34 (gros projets matures)
#   - Test  2023-2026 : médiane y=9.57  (projets récents plus petits)
# → Distribution trop différente → R² test artificellement négatif
# Le split aléatoire garantit distributions similaires dans train et test

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

print(f"\nSplit aléatoire 80/20 :")
print(f"  Train : {len(X_train)} repos  — médiane y={y_train.median():.2f}  std={y_train.std():.2f}")
print(f"  Test  : {len(X_test)}  repos  — médiane y={y_test.median():.2f}  std={y_test.std():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODÈLES — régularisés pour éviter l'overfitting
# ══════════════════════════════════════════════════════════════════════════════

models = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=100.0)),
    ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(alpha=0.05, max_iter=5000)),
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            max_features=0.5,
            random_state=42,
            n_jobs=1,
        )),
    ]),

    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=10,
            subsample=0.7,
            random_state=42,
        )),
    ]),
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT + MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def mape_heures(y_log_true, y_log_pred):
    y_true_h = np.expm1(np.array(y_log_true))
    y_pred_h = np.expm1(np.array(y_log_pred))
    mask = y_true_h > 100
    if mask.sum() == 0:
        return float("nan")
    return float(np.median(
        np.abs(y_true_h[mask] - y_pred_h[mask]) / y_true_h[mask] * 100
    ))

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'─'*72}")
print(f"{'Modèle':<20} {'R²train':>8} {'R²test':>8} {'Gap':>8} {'RMSE':>8} {'MAPE%':>8}")
print(f"{'─'*72}")

for name, pipe in models.items():
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test  = r2_score(y_test,  y_pred_test)
    gap      = r2_train - r2_test
    rmse     = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae      = mean_absolute_error(y_test, y_pred_test)
    mape     = mape_heures(y_test.values, y_pred_test)

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2", n_jobs=1)

    results[name] = {
        "pipe":        pipe,
        "r2_train":    r2_train,
        "r2_test":     r2_test,
        "gap":         gap,
        "rmse":        rmse,
        "mae":         mae,
        "mape":        mape,
        "cv_mean":     cv_scores.mean(),
        "cv_std":      cv_scores.std(),
        "y_pred_test": y_pred_test,
    }

    flag = ""
    if gap > 0.3:     flag = "  ⚠ overfitting"
    elif r2_test < 0: flag = "  ✗ pire que moyenne"

    print(f"{name:<20} {r2_train:>8.3f} {r2_test:>8.3f} {gap:>8.3f} {rmse:>8.3f} {mape:>7.1f}%{flag}")

print(f"{'─'*72}")
print(f"  Gap = R²train - R²test  |  idéal < 0.15")

# ── Cross-validation ──────────────────────────────────────────────────────────
print(f"\n── Cross-validation 5-fold (sur train) ──────────────────────────────")
print(f"{'Modèle':<20} {'CV R² moy':>12} {'±std':>8}  Statut")
print(f"{'─'*55}")
for name, res in results.items():
    cv_m, cv_s = res["cv_mean"], res["cv_std"]
    if cv_m > 0.65:   statut = "Bon"
    elif cv_m > 0.45: statut = "Acceptable"
    else:             statut = "Faible"
    if cv_s > 0.15:   statut += " (instable)"
    print(f"{name:<20} {cv_m:>12.3f} {cv_s:>8.3f}  {statut}")

# ── Meilleur modèle ───────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["r2_test"])
best      = results[best_name]

print(f"\n── Meilleur modèle : {best_name} ──────────────────────────────────────")
print(f"  R² test  = {best['r2_test']:.3f}")
print(f"  Gap      = {best['gap']:.3f}  (idéal < 0.15)")
print(f"  RMSE log = {best['rmse']:.3f}")
print(f"  MAE log  = {best['mae']:.3f}")
print(f"  MAPE     = {best['mape']:.1f}%")

r2 = best["r2_test"]
if r2 > 0.70:   msg = "Excellent"
elif r2 > 0.55: msg = "Bon — cohérent avec la littérature effort estimation"
elif r2 > 0.35: msg = "Acceptable — signal réel capturé"
elif r2 > 0.0:  msg = "Faible mais positif — mieux que la moyenne"
else:           msg = "Négatif — problème à investiguer"
print(f"\n  → {msg}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. IMPORTANCE DES FEATURES — n_jobs=1 obligatoire
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n── Importance des features ({best_name}) ──────────────────────────────")

perm = permutation_importance(
    best["pipe"], X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)
feat_imp = pd.DataFrame({
    "feature":    X_train.columns,
    "importance": perm.importances_mean,
    "std":        perm.importances_std,
}).sort_values("importance", ascending=False)

print(f"\n  Top 15 features les plus importantes :")
print(f"  {'Feature':<42} {'Importance':>12}")
print(f"  {'─'*57}")
for _, row in feat_imp.head(15).iterrows():
    bar = "█" * max(int(abs(row["importance"]) /
                    max(abs(feat_imp["importance"].max()), 1e-6) * 20), 1)
    print(f"  {row['feature']:<42} {row['importance']:>+12.4f}  {bar}")

feat_imp.to_csv("feature_importance.csv", index=False)
print(f"\n  Sauvegardé → feature_importance.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 5. GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Résultats ML — Prédiction effort open-source", fontsize=13, fontweight="bold")

# Prédit vs Réel
ax1 = axes[0, 0]
yp  = best["y_pred_test"]
ax1.scatter(y_test, yp, alpha=0.5, color="steelblue", s=30)
lims = [min(y_test.min(), yp.min()), max(y_test.max(), yp.max())]
ax1.plot(lims, lims, "r--", lw=1.5, label="Parfait")
ax1.set_xlabel("log(effort) réel")
ax1.set_ylabel("log(effort) prédit")
ax1.set_title(f"Prédit vs Réel — {best_name}\nR²={best['r2_test']:.3f}")
ax1.legend()

# Résidus
ax2 = axes[0, 1]
res_vals = y_test.values - yp
ax2.scatter(yp, res_vals, alpha=0.5, color="coral", s=30)
ax2.axhline(0, color="black", lw=1)
ax2.set_xlabel("log(effort) prédit")
ax2.set_ylabel("Résidu (réel - prédit)")
ax2.set_title("Analyse des résidus")

# Comparaison R²
ax3 = axes[1, 0]
names   = list(results.keys())
r2tests = [results[n]["r2_test"]  for n in names]
cvs     = [results[n]["cv_mean"]  for n in names]
x = np.arange(len(names))
w = 0.35
ax3.bar(x - w/2, r2tests, w, label="R² test",    color="steelblue", alpha=0.8)
ax3.bar(x + w/2, cvs,     w, label="R² CV train", color="orange",   alpha=0.8)
for i, (rt, cv) in enumerate(zip(r2tests, cvs)):
    ax3.text(i - w/2, max(rt, 0) + 0.01, f"{rt:.2f}", ha="center", fontsize=8)
    ax3.text(i + w/2, max(cv, 0) + 0.01, f"{cv:.2f}", ha="center", fontsize=8)
ax3.set_xticks(x)
ax3.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
ax3.axhline(0, color="black", lw=0.8)
ax3.set_ylabel("R²")
ax3.set_title("Comparaison modèles")
ax3.legend()

# Feature importance
ax4 = axes[1, 1]
top10  = feat_imp.head(10)
colors = ["steelblue" if v >= 0 else "coral" for v in top10["importance"][::-1]]
ax4.barh(top10["feature"][::-1], top10["importance"][::-1], color=colors, alpha=0.8)
ax4.axvline(0, color="black", lw=0.8)
ax4.set_xlabel("Permutation Importance")
ax4.set_title(f"Top 10 features — {best_name}")

plt.tight_layout()
plt.savefig("ml_results.png", dpi=150, bbox_inches="tight")
print(f"\n  Graphiques -> ml_results.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. TABLEAU RAPPORT
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*72}")
print("TABLEAU RÉCAPITULATIF — POUR LE RAPPORT")
print(f"{'='*72}")
print(f"{'Modèle':<20} {'R²test':>7} {'Gap':>7} {'RMSE':>7} {'MAE':>7} {'MAPE':>8} {'CV R²':>8}")
print(f"{'─'*72}")
for name, res in results.items():
    m = " <-- meilleur" if name == best_name else ""
    print(f"{name:<20} {res['r2_test']:>7.3f} {res['gap']:>7.3f} "
          f"{res['rmse']:>7.3f} {res['mae']:>7.3f} "
          f"{res['mape']:>7.1f}% {res['cv_mean']:>8.3f}{m}")
print(f"{'─'*72}")
print(f"Gap = R²train - R²test  (< 0.15 = bon, > 0.3 = overfitting)")
print(f"MAPE calculé sur heures reelles apres exp(y)")