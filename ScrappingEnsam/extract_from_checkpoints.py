"""
extract_from_checkpoints.py — Extrait les CSV depuis les fichiers checkpoint
=============================================================================
À utiliser quand les checkpoints existent mais pas les CSV.

Usage :
    python extract_from_checkpoints.py

Input  : checkpoint_LOT1.json, checkpoint_LOT2.json, ...
Output : features_raw_LOT1.csv, features_raw_LOT2.csv, ... + features_merged.csv
"""

import json
import csv
import os
import glob
import pandas as pd

CHECKPOINT_PATTERN = "checkpoint_LOT*.json"

def extract_csv_from_checkpoint(checkpoint_file: str) -> str:
    """Lit un checkpoint et génère le CSV correspondant."""
    with open(checkpoint_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    done    = data.get("done", [])

    # Déduire le numéro de lot depuis le nom du fichier
    lot_num = checkpoint_file.replace("checkpoint_LOT", "").replace(".json", "")
    output_csv = f"features_raw_LOT{lot_num}.csv"

    print(f"\n{checkpoint_file}")
    print(f"  Repos analysés   : {len(done)}")
    print(f"  Repos retenus    : {len(results)}")
    print(f"  Taux de rétention: {len(results)/max(len(done),1)*100:.1f}%")

    if not results:
        print(f"  ⚠ Aucune donnée dans ce checkpoint")
        return None

    # Écrire le CSV
    keys = list(results[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    print(f"  ✓ CSV créé → {output_csv}")

    # Afficher quelques stats
    efforts = [r.get("effort_target", 0) for r in results if r.get("effort_target", 0) > 0]
    if efforts:
        import numpy as np
        print(f"  effort_target : min={min(efforts):.0f}h  médiane={np.median(efforts):.0f}h  max={max(efforts):.0f}h")

    return output_csv


def main():
    print("=" * 60)
    print("Extraction CSV depuis les fichiers checkpoint")
    print("=" * 60)

    checkpoints = sorted(glob.glob(CHECKPOINT_PATTERN))

    if not checkpoints:
        print(f"\n⚠ Aucun fichier checkpoint trouvé (pattern: {CHECKPOINT_PATTERN})")
        print("Assurez-vous d'être dans le bon dossier.")
        return

    print(f"\nCheckpoints trouvés : {checkpoints}")

    csv_files = []
    for ckpt in checkpoints:
        csv_file = extract_csv_from_checkpoint(ckpt)
        if csv_file:
            csv_files.append(csv_file)

    # Merge automatique si plusieurs CSVs
    if len(csv_files) > 1:
        print(f"\n{'='*60}")
        print("Fusion des CSV en features_merged.csv")
        dfs = [pd.read_csv(f) for f in csv_files]
        merged = pd.concat(dfs, ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset="full_name", keep="first")
        print(f"Total : {before} → {len(merged)} après déduplication")
        merged.to_csv("features_merged.csv", index=False)
        print(f"✓ features_merged.csv créé ({len(merged)} repos)")
    elif len(csv_files) == 1:
        print(f"\n✓ Un seul lot trouvé — CSV disponible : {csv_files[0]}")
        print("  Attends les autres membres pour faire le merge complet.")

    print("\nDone.")


if __name__ == "__main__":
    main()
