"""
clean.py — Nettoyage et correction du dataset brut
===================================================
Corrige le bug churn_normalized, détecte les outliers,
vérifie la cohérence de toutes les colonnes.

Usage :
    python clean.py --input ../ScrappingEnsam/features_merged.csv --output features_clean.csv
    python clean.py  (utilise les fichiers par défaut)
"""

import pandas as pd
import numpy as np
import argparse

INPUT_CSV  = "../ScrappingEnsam/features_merged.csv"
OUTPUT_CSV = "features_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)
    print(f"Dataset initial : {n_start} repos, {len(df.columns)} colonnes")
    print()

    # ══ 1. CORRECTION DU BUG churn_normalized ════════════════════════════════
    # Bug : churn_normalized = churn_loc / active_days
    #       active_days = active_weeks * 5
    #       churn_loc = sum_weekly_churn
    # Or sum_weekly_churn / (active_weeks * 5) ≈ constante car les deux
    # varient proportionnellement. Résultat : ~160 pour tous les repos.
    #
    # Correction : normaliser par le NOMBRE DE JOURS CALENDAIRES du projet
    # (date dernière push - date création) — une mesure indépendante du churn.

    print("── Correction bug churn_normalized ──────────────────────────────")

    if "created_at" in df.columns and "days_inactive" in df.columns:
        # Reconstruire durée projet en jours depuis created_at
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        today = pd.Timestamp.now(tz="UTC")
        df["project_age_days"] = (today - df["created_at"]).dt.days - df["days_inactive"]
        df["project_age_days"] = df["project_age_days"].clip(lower=30)  # minimum 30 jours

        # Recalculer churn_normalized = churn_loc / project_age_days
        if "churn_loc" in df.columns:
            df["code_churn_normalized"] = (df["churn_loc"] / df["project_age_days"]).round(2)
            print(f"  Avant : médiane={160:.1f} (constante bugguée)")
            print(f"  Après : médiane={df['code_churn_normalized'].median():.1f}  "
                  f"std={df['code_churn_normalized'].std():.1f}")

            # Recalculer churn_hours avec la vraie normalisation
            df["churn_hours"] = (df["churn_loc"] / 15.0).round(1)
    else:
        print("  ⚠ Colonnes created_at ou days_inactive manquantes — correction impossible")

    # ══ 2. RECALCUL DE LA TARGET avec les valeurs corrigées ══════════════════
    print("\n── Recalcul effort_target ───────────────────────────────────────")

    required = ["churn_hours", "cycle_time_hours", "cocomo_hours"]
    if all(c in df.columns for c in required):
        df["effort_target"] = (
            0.5 * df["churn_hours"] +
            0.3 * df["cycle_time_hours"] +
            0.2 * df["cocomo_hours"]
        ).round(1)
        print(f"  Médiane effort_target : {df['effort_target'].median():.0f} h")
        print(f"  Min : {df['effort_target'].min():.0f} h  |  Max : {df['effort_target'].max():.0f} h")
    else:
        missing = [c for c in required if c not in df.columns]
        print(f"  ⚠ Colonnes manquantes : {missing}")

    # ══ 3. SUPPRESSION DES OUTLIERS EXTRÊMES ═════════════════════════════════
    print("\n── Suppression outliers ─────────────────────────────────────────")

    before = len(df)

    # effort_target : retirer < p1 et > p99
    if "effort_target" in df.columns:
        p1  = df["effort_target"].quantile(0.01)
        p99 = df["effort_target"].quantile(0.99)
        df  = df[(df["effort_target"] >= p1) & (df["effort_target"] <= p99)]
        print(f"  effort_target hors [{p1:.0f}h, {p99:.0f}h] → {before - len(df)} repos retirés")
        before = len(df)

    # Repos avec net_loc absurde (< 100 = site statique, > 50M = impossible)
    if "net_loc" in df.columns:
        df = df[(df["net_loc"] >= 100) & (df["net_loc"] <= 50_000_000)]
        print(f"  net_loc hors [100, 50M] → {before - len(df)} repos retirés")
        before = len(df)

    # ══ 4. VALEURS MANQUANTES ════════════════════════════════════════════════
    print("\n── Valeurs manquantes ───────────────────────────────────────────")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_before = df[num_cols].isnull().sum().sum()

    # Colonnes à 0 qui sont normales (SonarQube pas encore fait)
    sonar_cols = [c for c in df.columns if "sonar" in c or c == "avg_file_size_loc"]
    fill_zero  = sonar_cols + ["review_cycle_count"]

    for col in fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Autres colonnes numériques : médiane
    other_num = [c for c in num_cols if c not in fill_zero]
    df[other_num] = df[other_num].fillna(df[other_num].median())

    missing_after = df[num_cols].isnull().sum().sum()
    print(f"  Valeurs manquantes : {missing_before} → {missing_after}")

    # ══ 5. VÉRIFICATION COHÉRENCE ════════════════════════════════════════════
    print("\n── Vérification cohérence ───────────────────────────────────────")

    checks = {
        "bus_factor_ratio entre 0 et 1":
            ((df["bus_factor_ratio"] >= 0) & (df["bus_factor_ratio"] <= 1)).all()
            if "bus_factor_ratio" in df.columns else None,
        "active_contributors > 0":
            (df["active_contributors"] > 0).all()
            if "active_contributors" in df.columns else None,
        "effort_target > 0":
            (df["effort_target"] > 0).all()
            if "effort_target" in df.columns else None,
        "churn_normalized constant (bug)":
            not (df["code_churn_normalized"].std() < 1)
            if "code_churn_normalized" in df.columns else None,
    }

    for check, result in checks.items():
        if result is None:
            print(f"  ⚠ {check} — colonne absente")
        elif result:
            print(f"  ✓ {check}")
        else:
            print(f"  ✗ {check} — PROBLÈME DÉTECTÉ")

    # ══ 6. RÉSUMÉ FINAL ═══════════════════════════════════════════════════════
    print(f"\n── Résumé ───────────────────────────────────────────────────────")
    print(f"  Repos initial  : {n_start}")
    print(f"  Repos final    : {len(df)}")
    print(f"  Retirés        : {n_start - len(df)}")
    print(f"  Colonnes       : {len(df.columns)}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_clean = clean(df)
    df_clean.to_csv(args.output, index=False)
    print(f"\n✓ Sauvegardé → {args.output}")


if __name__ == "__main__":
    main()
