"""
weight_validation.py
====================
Valide empiriquement les poids 0.5 / 0.3 / 0.2 de la target composite
en faisant une régression linéaire forcée sur les 3 composantes.

Si les coefficients estimés sont proches des poids théoriques (écart < 15%),
les poids sont confirmés empiriquement.

Usage:
    python weight_validation.py --data dataset.csv
    python weight_validation.py --demo
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import sys

THEORETICAL_WEIGHTS = {
    "churn_hours": 0.5,
    "cycle_time_hours": 0.3,
    "cocomo_hours": 0.2,
}

def make_demo_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    churn_hours = rng.exponential(5000, n)
    cycle_time_hours = rng.exponential(300, n) * rng.integers(2, 12, n)
    cocomo_hours = 2.4 * rng.uniform(10, 200, n) ** 1.05 * 160
    noise = rng.normal(0, 500, n)
    effort_target = 0.5 * churn_hours + 0.3 * cycle_time_hours + 0.2 * cocomo_hours + noise
    df = pd.DataFrame({
        "churn_hours": churn_hours,
        "cycle_time_hours": cycle_time_hours,
        "cocomo_hours": cocomo_hours,
        "effort_target": effort_target,
    })
    return df

def validate_weights(df: pd.DataFrame, target_col="effort_target"):
    components = list(THEORETICAL_WEIGHTS.keys())
    X = df[components].values
    y = df[target_col].values

    # Régression avec intercept=False pour avoir des coefficients comparables aux poids
    model_raw = LinearRegression(fit_intercept=False)
    model_raw.fit(X, y)
    raw_coefs = model_raw.coef_

    # Normalisation des coefficients pour sommer à 1 (comme les poids théoriques)
    total = raw_coefs.sum()
    normalized_coefs = raw_coefs / total if total > 0 else raw_coefs

    # Cross-validation temporelle
    tss = TimeSeriesSplit(n_splits=5)
    model_cv = LinearRegression(fit_intercept=True)
    cv_scores = cross_val_score(model_cv, X, y, cv=tss, scoring="r2")

    results = []
    print(f"\n{'='*65}")
    print(f"  VALIDATION DES POIDS — Régression sur les 3 composantes")
    print(f"{'='*65}")
    print(f"\n{'Composante':<25} {'Théorique':>10} {'Estimé':>10} {'Écart':>10} {'Status':>12}")
    print("-" * 70)

    for feat, theoric, estimated in zip(components, THEORETICAL_WEIGHTS.values(), normalized_coefs):
        delta = abs(estimated - theoric)
        delta_pct = delta / theoric * 100
        status = "✅ OK" if delta_pct < 15 else ("⚠️  RÉAJUSTER" if delta_pct < 30 else "❌ ÉCART FORT")
        print(f"{feat:<25} {theoric:>10.2f} {estimated:>10.3f} {delta_pct:>9.1f}% {status:>12}")
        results.append({
            "feature": feat,
            "theoretical": theoric,
            "estimated": estimated,
            "delta_pct": delta_pct,
            "status": status,
        })

    print(f"\n  R² moyen (TimeSeriesSplit 5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  R² par fold: {[f'{s:.3f}' for s in cv_scores]}")

    all_ok = all(r["delta_pct"] < 15 for r in results)
    print(f"\n{'='*65}")
    if all_ok:
        print("  ✅ POIDS CONFIRMÉS empiriquement — écarts < 15% sur tous")
    else:
        bad = [r for r in results if r["delta_pct"] >= 15]
        print(f"  ⚠️  {len(bad)} poids à reconsidérer:")
        for r in bad:
            print(f"     → {r['feature']}: théorique={r['theoretical']}, estimé={r['estimated']:.3f}")
    print(f"{'='*65}\n")

    return pd.DataFrame(results), cv_scores

def plot_weights(results_df, cv_scores):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(results_df))
    width = 0.35
    bars1 = ax1.bar(x - width/2, results_df["theoretical"], width, label="Poids théoriques", color="#6366F1", alpha=0.85)
    bars2 = ax1.bar(x + width/2, results_df["estimated"], width, label="Coefficients estimés", color="#D97706", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df["feature"], rotation=15, ha="right")
    ax1.set_ylabel("Poids")
    ax1.set_title("Poids théoriques vs estimés", fontweight="bold")
    ax1.legend()
    ax1.set_ylim(0, 0.75)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", fontsize=9, color="#D97706")

    ax2.bar(range(len(cv_scores)), cv_scores, color="#16A34A", alpha=0.8)
    ax2.axhline(cv_scores.mean(), color="#DC2626", linestyle="--", linewidth=1.5,
                label=f"Moyenne: {cv_scores.mean():.3f}")
    ax2.set_xlabel("Fold (TimeSeriesSplit)")
    ax2.set_ylabel("R²")
    ax2.set_title("R² par fold (CV temporelle)", fontweight="bold")
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.set_xticks(range(len(cv_scores)))
    ax2.set_xticklabels([f"Fold {i+1}" for i in range(len(cv_scores))])

    plt.suptitle("Validation empirique des poids de la target composite", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("weight_validation_report.png", dpi=150, bbox_inches="tight")
    print("[Graph saved] → weight_validation_report.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target", type=str, default="effort_target")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo or args.data is None:
        print("[MODE DEMO] Données synthétiques avec bruit gaussien")
        df = make_demo_data()
    else:
        df = pd.read_csv(args.data)

    missing = [c for c in list(THEORETICAL_WEIGHTS.keys()) + [args.target] if c not in df.columns]
    if missing:
        print(f"[ERREUR] Colonnes manquantes: {missing}")
        print(f"  Colonnes attendues: {list(THEORETICAL_WEIGHTS.keys()) + [args.target]}")
        sys.exit(1)

    results_df, cv_scores = validate_weights(df, args.target)
    plot_weights(results_df, cv_scores)

if __name__ == "__main__":
    main()
