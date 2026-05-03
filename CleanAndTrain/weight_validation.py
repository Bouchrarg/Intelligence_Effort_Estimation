"""
weight_validation.py — Calcul des poids réels de la target hybride
===================================================================
Compare les poids théoriques (0.5 / 0.3 / 0.2) aux poids estimés
empiriquement sur nos données via régression linéaire contrainte.

Usage :
    python weight_validation.py --input features_clean.csv

Sorties :
  - Poids réels estimés vs poids théoriques
  - Décision : garder ou ajuster les poids
  - effort_target recalculée avec les nouveaux poids si nécessaire
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import normalize
from scipy.optimize import minimize

INPUT_CSV  = "features_clean.csv"
OUTPUT_CSV = "features_clean_reweighted.csv"

# Poids théoriques actuels
THEORETICAL_WEIGHTS = {"churn": 0.5, "cycle": 0.3, "cocomo": 0.2}
ADJUSTMENT_THRESHOLD = 0.15   # si écart > 15% → on ajuste


def estimate_weights(df: pd.DataFrame) -> dict:
    """
    Estime les poids réels par régression linéaire sans intercept,
    contrainte à des poids positifs qui somment à 1.
    """
    required = ["churn_hours", "cycle_time_hours", "cocomo_hours", "effort_target"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠ Colonnes manquantes : {missing}")
        return THEORETICAL_WEIGHTS

    X = df[["churn_hours", "cycle_time_hours", "cocomo_hours"]].values
    y = df["effort_target"].values

    # ── Méthode 1 : régression OLS sans intercept ───────────────────────────
    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(X, y)
    raw_weights = reg.coef_

    # Normaliser pour sommer à 1
    total = raw_weights.sum()
    if total > 0:
        normalized = raw_weights / total
    else:
        normalized = np.array([0.5, 0.3, 0.2])

    ols_weights = {
        "churn": round(float(normalized[0]), 3),
        "cycle": round(float(normalized[1]), 3),
        "cocomo": round(float(normalized[2]), 3),
    }

    # ── Méthode 2 : optimisation contrainte (somme = 1, tous >= 0) ──────────
    def objective(w):
        pred = X @ w
        return np.mean((pred - y) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0, 1)] * 3
    x0          = np.array([0.5, 0.3, 0.2])
    result      = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    opt_weights = {
        "churn": round(float(result.x[0]), 3),
        "cycle": round(float(result.x[1]), 3),
        "cocomo": round(float(result.x[2]), 3),
    }

    print("\n── Méthode 1 : Régression OLS sans intercept ────────────────────")
    print(f"  churn  : {ols_weights['churn']:.3f}  (théorique : 0.500)")
    print(f"  cycle  : {ols_weights['cycle']:.3f}  (théorique : 0.300)")
    print(f"  cocomo : {ols_weights['cocomo']:.3f}  (théorique : 0.200)")

    print("\n── Méthode 2 : Optimisation contrainte (SLSQP) ──────────────────")
    print(f"  churn  : {opt_weights['churn']:.3f}  (théorique : 0.500)")
    print(f"  cycle  : {opt_weights['cycle']:.3f}  (théorique : 0.300)")
    print(f"  cocomo : {opt_weights['cocomo']:.3f}  (théorique : 0.200)")

    # Moyenne des deux méthodes comme estimation finale
    final_weights = {
        "churn":  round((ols_weights["churn"]  + opt_weights["churn"])  / 2, 3),
        "cycle":  round((ols_weights["cycle"]  + opt_weights["cycle"])  / 2, 3),
        "cocomo": round((ols_weights["cocomo"] + opt_weights["cocomo"]) / 2, 3),
    }
    # Re-normaliser
    total = sum(final_weights.values())
    final_weights = {k: round(v / total, 3) for k, v in final_weights.items()}

    return ols_weights, opt_weights, final_weights


def evaluate_adjustment_needed(final_weights: dict) -> bool:
    """Décide si on ajuste les poids ou si les théoriques sont OK."""
    max_ecart = max(
        abs(final_weights["churn"]  - THEORETICAL_WEIGHTS["churn"]),
        abs(final_weights["cycle"]  - THEORETICAL_WEIGHTS["cycle"]),
        abs(final_weights["cocomo"] - THEORETICAL_WEIGHTS["cocomo"]),
    )
    print(f"\n── Décision d'ajustement ────────────────────────────────────────")
    print(f"  Écart maximal : {max_ecart:.3f}  (seuil : {ADJUSTMENT_THRESHOLD})")
    if max_ecart > ADJUSTMENT_THRESHOLD:
        print(f"  → AJUSTEMENT RECOMMANDÉ — écart > {ADJUSTMENT_THRESHOLD*100:.0f}%")
        return True
    else:
        print(f"  → POIDS THÉORIQUES VALIDÉS — écart < {ADJUSTMENT_THRESHOLD*100:.0f}%")
        return False


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return round(1 - ss_res / (ss_tot + 1e-9), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    print("=" * 65)
    print("Weight Validation — Poids réels vs poids théoriques")
    print("=" * 65)

    df = pd.read_csv(args.input)
    print(f"Dataset : {len(df)} repos")

    # Vérifier qu'on a les colonnes composantes
    required = ["churn_hours", "cycle_time_hours", "cocomo_hours", "effort_target"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n⚠ Colonnes manquantes : {missing}")
        print("  Assurez-vous d'avoir lancé clean.py avant.")
        return

    # Stats des composantes
    print("\n── Statistiques des composantes ─────────────────────────────────")
    for col in ["churn_hours", "cycle_time_hours", "cocomo_hours"]:
        print(f"  {col:25s} médiane={df[col].median():.0f}  "
              f"std={df[col].std():.0f}  max={df[col].max():.0f}")

    # Corrélations avec effort_target
    print("\n── Corrélations avec effort_target ──────────────────────────────")
    for col in ["churn_hours", "cycle_time_hours", "cocomo_hours"]:
        corr = df[col].corr(df["effort_target"])
        bar  = "█" * int(abs(corr) * 20)
        print(f"  {col:25s} r={corr:+.3f}  {bar}")

    # Estimation des poids
    ols_w, opt_w, final_w = estimate_weights(df)

    print(f"\n── Poids finaux (moyenne OLS + optimisation) ────────────────────")
    print(f"  churn  : {final_w['churn']:.3f}  (théorique : 0.500)  "
          f"écart : {abs(final_w['churn']-0.5)*100:.1f}%")
    print(f"  cycle  : {final_w['cycle']:.3f}  (théorique : 0.300)  "
          f"écart : {abs(final_w['cycle']-0.3)*100:.1f}%")
    print(f"  cocomo : {final_w['cocomo']:.3f}  (théorique : 0.200)  "
          f"écart : {abs(final_w['cocomo']-0.2)*100:.1f}%")

    # Décision
    adjust = evaluate_adjustment_needed(final_w)

    # Comparer R² des deux formulations
    X = df[["churn_hours", "cycle_time_hours", "cocomo_hours"]].values
    y = df["effort_target"].values

    pred_theoretical = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
    pred_empirical   = final_w["churn"] * X[:, 0] + final_w["cycle"] * X[:, 1] + final_w["cocomo"] * X[:, 2]

    r2_theoretical = compute_r2(y, pred_theoretical)
    r2_empirical   = compute_r2(y, pred_empirical)

    print(f"\n── Comparaison R² ───────────────────────────────────────────────")
    print(f"  Poids théoriques (0.5/0.3/0.2) : R² = {r2_theoretical}")
    print(f"  Poids empiriques               : R² = {r2_empirical}")
    print(f"  Gain : {r2_empirical - r2_theoretical:+.4f}")

    # Si ajustement, recalculer effort_target et sauvegarder
    if adjust:
        print(f"\n── Recalcul effort_target avec poids empiriques ─────────────────")
        df["effort_target_empirical"] = (
            final_w["churn"]  * df["churn_hours"] +
            final_w["cycle"]  * df["cycle_time_hours"] +
            final_w["cocomo"] * df["cocomo_hours"]
        ).round(1)

        # Comparer les deux targets
        diff = (df["effort_target_empirical"] - df["effort_target"]).abs()
        print(f"  Différence médiane entre les deux targets : {diff.median():.0f} h")
        print(f"  Différence max : {diff.max():.0f} h")

        # Sauvegarder avec les deux colonnes pour pouvoir comparer
        df["weights_used"] = f"churn={final_w['churn']}_cycle={final_w['cycle']}_cocomo={final_w['cocomo']}"
        df.to_csv(args.output, index=False)
        print(f"\n✓ Dataset avec poids empiriques → {args.output}")
        print(f"  Colonne effort_target_empirical ajoutée")
    else:
        print(f"\n✓ Poids théoriques (0.5/0.3/0.2) validés — pas de modification")
        print(f"  Dataset original conservé : {args.input}")

    # Résumé pour le rapport
    print(f"\n── À mentionner dans le rapport ─────────────────────────────────")
    print(f"  Poids théoriques justifiés par R² littérature : 0.5 / 0.3 / 0.2")
    print(f"  Poids estimés sur {len(df)} repos réels        : "
          f"{final_w['churn']} / {final_w['cycle']} / {final_w['cocomo']}")
    print(f"  Écart maximal                                 : "
          f"{max(abs(final_w['churn']-0.5), abs(final_w['cycle']-0.3), abs(final_w['cocomo']-0.2))*100:.1f}%")
    print(f"  Décision                                      : "
          f"{'Ajustement appliqué' if adjust else 'Poids théoriques conservés'}")


if __name__ == "__main__":
    main()
