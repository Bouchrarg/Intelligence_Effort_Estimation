"""
merge_all.py — Fusion des 4 CSVs en un seul dataset
====================================================
À lancer sur UN SEUL PC après que tous les membres ont fini leur lot.

Usage :
    python merge_all.py

Input  : features_raw_LOT1.csv, features_raw_LOT2.csv, ...
Output : features_merged.csv
"""

import pandas as pd
import os
import glob

INPUT_PATTERN = "Scrapped_Data/features_raw_LOT*.csv"
OUTPUT_CSV    = "features_merged.csv"

def main():
    files = sorted(glob.glob(INPUT_PATTERN))
    if not files:
        print(f"Aucun fichier trouvé matching '{INPUT_PATTERN}'")
        print("Assurez-vous que tous les CSV sont dans le même dossier.")
        return

    print(f"Fichiers trouvés : {files}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"  {f} → {len(df)} repos")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nAvant déduplication : {len(merged)} repos")

    # Supprimer les doublons sur full_name (garder la première occurrence)
    merged = merged.drop_duplicates(subset="full_name", keep="first")
    print(f"Après déduplication : {len(merged)} repos")

    # Stats rapides
    print(f"\nDistribution par lot :")
    print(merged["lot"].value_counts().sort_index().to_string())
    print(f"\nDistribution par langage :")
    print(merged["language"].value_counts().head(10).to_string())
    print(f"\nTarget (effort_target) — statistiques :")
    print(merged["effort_target"].describe().round(1).to_string())

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Dataset fusionné → {OUTPUT_CSV} ({len(merged)} repos)")


if __name__ == "__main__":
    main()
