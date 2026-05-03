"""
feature_engineering.py — v3 FINAL
===================================
Corrections appliquées :
  ✓ log_total_commits retiré (leakage indirect via COCOMO)
  ✓ avg_file_size_loc retiré (toujours 0 — SonarQube pas fait)
  ✓ closed_issues retiré (toujours 1 — filtre scraper MIN_CLOSED_ISSUES=1)
  ✓ log_closed_issues retiré (dérivé de closed_issues=constante)
  ✓ comment_per_pr_avg retiré (toujours 0 — bug scraper)
  ✓ reliability_score retiré (score composite → leakage indirect possible)

Usage :
    python feature_engineering.py --input features_clean.csv
"""

import pandas as pd
import numpy as np
import argparse

INPUT_CSV  = "features_clean.csv"
TARGET_COL = "effort_target"

# ══════════════════════════════════════════════════════════════════════════════
# COLONNES EXCLUES DE X
# ══════════════════════════════════════════════════════════════════════════════

# 1. Leakage direct — composantes de la formule effort_target
LEAKAGE_DIRECT = [
    "churn_hours",
    "cycle_time_hours",
    "cocomo_hours",
    "cocomo_pm",
    "net_loc",
    "churn_loc",
    "active_days",
]

# 2. Leakage indirect — corrélation > 0.95 avec effort_target
LEAKAGE_INDIRECT = [
    "total_commits",      # → net_loc → cocomo → effort_target  (r=0.999)
    "log_total_commits",  # idem en log
]

# 3. Colonnes constantes ou inutiles
USELESS_COLS = [
    "avg_file_size_loc",   # toujours 0.0 (SonarQube pas fait)
    "closed_issues",       # toujours 1.0 (filtre scraper MIN_CLOSED_ISSUES=1)
    "log_closed_issues",   # dérivé de closed_issues=constante → nan
    "comment_per_pr_avg",  # toujours 0.0 (bug scraper)
    "reliability_score",   # score composite maison → leakage indirect possible
]

# 4. Meta-colonnes (identifiants, pas de valeur ML)
META_COLS = [
    "full_name", "url", "language", "created_at",
    "lot", "weights_used", "project_age_days",
]

ALL_EXCLUDED = set(
    LEAKAGE_DIRECT + LEAKAGE_INDIRECT + USELESS_COLS + META_COLS + [TARGET_COL]
)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features dérivées depuis les métriques GitHub observables."""
    print(f"\nDataset entrant : {len(df)} repos, {len(df.columns)} colonnes")

    # ── Log-transforms ────────────────────────────────────────────────────────
    log_candidates = [
        "active_contributors", "pr_count_merged", "stars",
        "dependency_count", "code_churn_normalized",
        "pr_merge_time_median_h", "issues_resolution_time_h",
    ]
    for col in log_candidates:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # ── Features dérivées ─────────────────────────────────────────────────────
    if all(c in df.columns for c in ["has_ci", "has_tests"]):
        df["process_maturity"] = (
            df["has_ci"].astype(float) + df["has_tests"].astype(float)
        ) / 2.0

    if "bus_factor_ratio" in df.columns:
        df["effort_distribution"] = 1.0 - df["bus_factor_ratio"]

    if "ci_success_rate" in df.columns:
        df["ci_failure_rate"] = 1.0 - df["ci_success_rate"]

    if all(c in df.columns for c in ["pr_count_merged", "active_contributors"]):
        df["pr_per_contributor"] = (
            df["pr_count_merged"] / (df["active_contributors"] + 1)
        ).clip(0, 500)

    if all(c in df.columns for c in ["code_churn_normalized", "weighted_experience"]):
        df["exp_weighted_churn"] = (
            df["code_churn_normalized"] * df["weighted_experience"]
        ).clip(lower=0)

    if all(c in df.columns for c in ["code_churn_normalized", "active_contributors"]):
        df["contributor_productivity"] = (
            df["code_churn_normalized"] / (df["active_contributors"] + 1)
        )

    if all(c in df.columns for c in ["review_cycle_count", "pr_merge_time_median_h"]):
        df["review_effort_proxy"] = (
            df["review_cycle_count"] * df["pr_merge_time_median_h"]
        ).clip(lower=0)

    # ── Nettoyage inf / nan ───────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    print(f"Dataset après feature engineering : {len(df.columns)} colonnes")
    return df


def build_X_y(df: pd.DataFrame):
    """Construit X propre et y log-transformé."""

    feature_cols = [
        c for c in df.columns
        if c not in ALL_EXCLUDED
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X     = df[feature_cols].copy()
    y_raw = df[TARGET_COL].copy()
    y_log = np.log1p(y_raw)
    y_log.name = "log_effort_target"

    # ── Supprimer colonnes constantes résiduelles ─────────────────────────────
    print("\n── Vérification colonnes constantes ─────────────────────────────")
    dropped = [col for col in feature_cols if X[col].std() < 1e-6]
    if dropped:
        for col in dropped:
            print(f"  ✗ {col} — constante (std≈0), retirée")
        X = X.drop(columns=dropped)
        feature_cols = [c for c in feature_cols if c not in dropped]
    else:
        print("  ✓ Aucune colonne constante résiduelle")

    # ── Corrélations avec y_log ───────────────────────────────────────────────
    print("\n── Features retenues dans X ─────────────────────────────────────")
    leakage_residuel = []
    for col in sorted(X.columns):
        try:
            corr = float(X[col].corr(y_log))
        except Exception:
            corr = float("nan")
        marker = "★" if abs(corr) > 0.4 else " "
        print(f"  {marker} {col:45s}  r={corr:+.3f}")
        if abs(corr) > 0.95:
            leakage_residuel.append((col, corr))

    # ── Alerte leakage résiduel ───────────────────────────────────────────────
    print("\n── Leakage résiduel ─────────────────────────────────────────────")
    if leakage_residuel:
        for col, corr in leakage_residuel:
            print(f"  ⚠ LEAKAGE : {col}  r={corr:.4f}  → retirer manuellement")
    else:
        print("  ✓ Aucun leakage résiduel (r < 0.95 pour toutes les features)")

    # ── Résumé ────────────────────────────────────────────────────────────────
    print(f"\n── Résumé X / y ─────────────────────────────────────────────────")
    print(f"  X  : {X.shape[0]} repos × {X.shape[1]} features")
    print(f"  y  : log(effort_target)")
    print(f"       médiane brute = {y_raw.median():.0f} h")
    print(f"       médiane log   = {y_log.median():.2f}")
    print(f"       std brute     = {y_raw.std():.0f} h")
    print(f"       std log       = {y_log.std():.2f}")

    return X, y_log, y_raw, feature_cols


def temporal_split(df, X, y_log, train_ratio=0.8):
    """Split temporel : anciens → train, récents → test."""
    if "created_at" not in df.columns:
        print("  ⚠ created_at absent — split aléatoire")
        idx = X.index.tolist()
        np.random.shuffle(idx)
        n = int(len(idx) * train_ratio)
        train_idx, test_idx = idx[:n], idx[n:]
    else:
        df_sorted = df.sort_values("created_at", ascending=True)
        n_train   = int(len(df_sorted) * train_ratio)
        train_idx = df_sorted.index[:n_train].tolist()
        test_idx  = df_sorted.index[n_train:].tolist()

    X_train = X.loc[train_idx]
    X_test  = X.loc[test_idx]
    y_train = y_log.loc[train_idx]
    y_test  = y_log.loc[test_idx]

    print(f"\n── Split temporel {int(train_ratio*100)}/{int((1-train_ratio)*100)} ──────────────────────────────────────")
    print(f"  Train : {len(X_train)} repos (projets les plus anciens)")
    print(f"  Test  : {len(X_test)}  repos (projets les plus récents)")
    if "created_at" in df.columns:
        print(f"  Train période : {df.loc[train_idx,'created_at'].min()[:10]} → {df.loc[train_idx,'created_at'].max()[:10]}")
        print(f"  Test  période : {df.loc[test_idx, 'created_at'].min()[:10]} → {df.loc[test_idx, 'created_at'].max()[:10]}")

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default=INPUT_CSV)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    args = parser.parse_args()

    print("=" * 65)
    print("Feature Engineering Pipeline — v3 FINAL")
    print("=" * 65)

    df = pd.read_csv(args.input)

    if TARGET_COL not in df.columns:
        print(f"✗ Colonne '{TARGET_COL}' introuvable. Arrêt.")
        return

    df = engineer(df)
    X, y_log, y_raw, feature_cols = build_X_y(df)
    X_train, X_test, y_train, y_test = temporal_split(df, X, y_log, args.train_ratio)

    # Sauvegardes
    df.to_csv("features_final.csv", index=False)
    X.to_csv("X.csv",               index=False)
    y_log.to_csv("y.csv",           index=False, header=True)
    y_raw.to_csv("y_raw.csv",       index=False, header=True)
    X_train.to_csv("X_train.csv",   index=False)
    X_test.to_csv("X_test.csv",     index=False)
    y_train.to_csv("y_train.csv",   index=False, header=True)
    y_test.to_csv("y_test.csv",     index=False, header=True)

    print(f"\n── Fichiers générés ─────────────────────────────────────────────")
    print(f"  X.csv           — {X.shape[1]} features propres, sans leakage")
    print(f"  y.csv           — log(effort_target)")
    print(f"  y_raw.csv       — effort_target en heures brutes")
    print(f"  X_train/y_train — {len(X_train)} repos")
    print(f"  X_test/y_test   — {len(X_test)}  repos")
    print(f"\n✓ Prêt pour Phase 2 — entraînement ML")
    print(f"\n  Rappel interprétation des prédictions ML :")
    print(f"  y_pred en heures = exp(y_pred_log) - 1")


if __name__ == "__main__":
    main()