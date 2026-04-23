"""
leakage_check.py
================
Vérifie si une feature est trop corrélée avec la target (data leakage).
Seuil : corrélation de Pearson > 0.95 => alerte LEAKAGE DETECTED.

Usage:
    python leakage_check.py --data chemin/vers/dataset.csv --target effort_target

Si pas de dataset encore:
    python leakage_check.py --demo
    (génère des données synthétiques pour tester le script)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

LEAKAGE_THRESHOLD = 0.95

def compute_correlations(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    result = pd.DataFrame({
        "feature": corr.index,
        "correlation": corr.values,
        "abs_corr": corr.abs().values,
    }).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return result

def flag_leakage(corr_df: pd.DataFrame) -> pd.DataFrame:
    corr_df["leakage_risk"] = corr_df["abs_corr"].apply(
        lambda x: "🔴 LEAKAGE" if x > LEAKAGE_THRESHOLD
                  else ("🟡 HIGH" if x > 0.80 else "🟢 OK")
    )
    return corr_df

def plot_correlations(corr_df: pd.DataFrame, target_col: str, output_path="leakage_report.png"):
    fig, ax = plt.subplots(figsize=(10, max(5, len(corr_df) * 0.4)))
    colors = []
    for _, row in corr_df.iterrows():
        if row["abs_corr"] > LEAKAGE_THRESHOLD:
            colors.append("#DC2626")
        elif row["abs_corr"] > 0.80:
            colors.append("#D97706")
        else:
            colors.append("#16A34A")

    bars = ax.barh(corr_df["feature"], corr_df["abs_corr"], color=colors)
    ax.axvline(x=LEAKAGE_THRESHOLD, color="#DC2626", linestyle="--", linewidth=1.5,
               label=f"Seuil leakage ({LEAKAGE_THRESHOLD})")
    ax.axvline(x=0.80, color="#D97706", linestyle=":", linewidth=1.2,
               label="Corrélation haute (0.80)")
    ax.set_xlabel("Corrélation absolue avec la target", fontsize=11)
    ax.set_title(f"Corrélations features → {target_col}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, corr_df["abs_corr"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[Graph saved] → {output_path}")
    return fig

def make_demo_data(n=200, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    churn_hours = rng.exponential(scale=5000, size=n)
    cycle_time_hours = rng.exponential(scale=300, size=n) * rng.integers(2, 12, size=n)
    cocomo_hours = 2.4 * (rng.uniform(10, 200, size=n)) ** 1.05 * 160

    effort_target = (0.5 * churn_hours +
                     0.3 * cycle_time_hours +
                     0.2 * cocomo_hours +
                     rng.normal(0, 200, size=n))

    # Churn feature = churn_hours / nb_jours (corrélé ~0.99 avec la target → leakage!)
    nb_jours = rng.integers(30, 365, size=n)
    code_churn_normalized = churn_hours / nb_jours

    df = pd.DataFrame({
        "code_churn_normalized": code_churn_normalized,  # LEAKAGE FEATURE
        "cyclomatic_complexity_avg": rng.uniform(1, 20, size=n),
        "avg_file_size_loc": rng.uniform(50, 800, size=n),
        "dependency_count": rng.integers(1, 100, size=n),
        "language_diversity": rng.uniform(0, 1, size=n),
        "active_contributors": rng.integers(1, 50, size=n),
        "bus_factor_ratio": rng.uniform(0.1, 1.0, size=n),
        "weighted_experience": rng.uniform(0.5, 15, size=n),
        "comment_per_pr_avg": rng.uniform(0, 40, size=n),
        "review_cycle_count": rng.uniform(1, 10, size=n),
        "pr_merge_time_median_h": rng.exponential(72, size=n),
        "issues_resolution_time_h": rng.exponential(120, size=n),
        "pr_count_merged": rng.integers(5, 500, size=n),
        "commit_velocity_trend": rng.normal(0, 2, size=n),
        "release_regularity": rng.uniform(0, 30, size=n),
        "weekend_commit_ratio": rng.uniform(0, 0.5, size=n),
        "has_ci": rng.integers(0, 2, size=n),
        "has_tests": rng.integers(0, 2, size=n),
        "ci_success_rate": rng.uniform(0.5, 1.0, size=n),
        "effort_target": effort_target,
    })
    return df

def main():
    global LEAKAGE_THRESHOLD
    parser = argparse.ArgumentParser(description="Leakage check — corrélations features/target")
    parser.add_argument("--data", type=str, default=None, help="Chemin vers le CSV dataset")
    parser.add_argument("--target", type=str, default="effort_target", help="Nom de la colonne target")
    parser.add_argument("--demo", action="store_true", help="Génère données synthétiques pour démo")
    parser.add_argument("--threshold", type=float, default=LEAKAGE_THRESHOLD,
                        help=f"Seuil corrélation leakage (défaut={LEAKAGE_THRESHOLD})")
    args = parser.parse_args()

    LEAKAGE_THRESHOLD = args.threshold

    if args.demo or args.data is None:
        print("[MODE DEMO] Données synthétiques — churn_normalized sera détecté comme leakage")
        df = make_demo_data()
    else:
        df = pd.read_csv(args.data)
        print(f"[Loaded] {len(df)} lignes, {len(df.columns)} colonnes depuis {args.data}")

    if args.target not in df.columns:
        print(f"[ERREUR] Colonne target '{args.target}' introuvable. Colonnes: {list(df.columns)}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  LEAKAGE CHECK — target: '{args.target}'")
    print(f"  Seuil alerte: corr absolue > {LEAKAGE_THRESHOLD}")
    print(f"{'='*60}\n")

    corr_df = compute_correlations(df, args.target)
    corr_df = flag_leakage(corr_df)

    print(corr_df.to_string(index=False))

    leaky = corr_df[corr_df["leakage_risk"] == "🔴 LEAKAGE"]
    high = corr_df[corr_df["leakage_risk"] == "🟡 HIGH"]

    print(f"\n{'='*60}")
    if len(leaky) > 0:
        print(f"  🔴 LEAKAGE DETECTED sur {len(leaky)} feature(s):")
        for _, row in leaky.iterrows():
            print(f"     → {row['feature']} (corr={row['correlation']:.4f})")
        print(f"\n  RECOMMANDATION: Retirer ces features de X OU appliquer")
        print(f"  une séparation temporelle stricte (features sur mois 1-6,")
        print(f"  target calculée sur mois 7-12).")
    else:
        print("  ✅ Aucun leakage critique détecté (corr < {LEAKAGE_THRESHOLD})")

    if len(high) > 0:
        print(f"\n  🟡 Corrélations élevées (> 0.80) à surveiller:")
        for _, row in high.iterrows():
            print(f"     → {row['feature']} (corr={row['correlation']:.4f})")
    print(f"{'='*60}\n")

    plot_correlations(corr_df, args.target)

if __name__ == "__main__":
    main()