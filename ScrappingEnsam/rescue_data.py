import os
import csv
import time
import requests
from rich.console import Console

console = Console()

# ==========================================
# CONFIGURATION
# ==========================================
GH_TOKEN = os.getenv("GH_TOKEN") # Remplacez par votre token si la variable d'env n'est pas définie
# GH_TOKEN = "votre_token_ici" 

INPUT_CSV = "features_merged.csv"
OUTPUT_CSV = "features_merged_fixed.csv"

# Constantes du modèle d'effort (identiques à scraper.py)
HOURS_PER_PM = 160
PRODUCTIVITY_LOC_PER_HOUR = 15.0

def get_repo_size(full_name, headers):
    """
    Récupère la taille du dépôt en KB via l'API principale de GitHub.
    C'est un appel très rapide (pas de timeout 202 comme code_frequency).
    """
    url = f"https://api.github.com/repos/{full_name}"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                return data.get("size", 0)
            elif r.status_code == 403 or r.status_code == 429:
                console.print(f"[yellow]Rate limit atteint, attente 60s...[/]")
                time.sleep(60)
            elif r.status_code == 404:
                return 0
        except Exception as e:
            time.sleep(2)
    return 0

def main():
    if not GH_TOKEN or GH_TOKEN == "votre_token_ici":
        console.print("[red bold]ATTENTION : GH_TOKEN n'est pas défini.[/]")
        console.print("Veuillez éditer le script pour y mettre votre token GitHub.")
        return

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GH_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Charger les repos déjà traités si le fichier de sortie existe
    processed_repos = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                processed_repos.add(r["full_name"])

    # Lire les données existantes
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    to_process = [r for r in rows if r["full_name"] not in processed_repos]

    if not to_process:
        console.print("[green]Aucun nouveau dépôt à traiter. Tout est à jour ![/]")
        return

    console.print(f"[cyan]Démarrage du sauvetage pour {len(to_process)} NOUVEAUX dépôts ({len(processed_repos)} déjà faits)...[/]")
    
    fixed_rows = []
    
    for i, row in enumerate(to_process):
        full_name = row["full_name"]
        console.print(f"[{i+1}/{len(rows)}] Traitement de [bold]{full_name}[/]...", end=" ")
        
        # 1. Obtenir la taille réelle (en KB)
        size_kb = get_repo_size(full_name, headers)
        
        # 2. Estimer les LOC réelles (Net LOC)
        # 1 KB = 1024 octets. On estime qu'une ligne de code fait environ 35 octets.
        if size_kb > 0:
            net_loc = (size_kb * 1024) / 35
        else:
            # Fallback raisonnable si repo introuvable
            commits = float(row.get("total_commits", 0))
            net_loc = commits * 15  # Estimation prudente
            
        # 3. Estimer le Code Churn
        # Un commit modifie en moyenne 30 lignes
        commits = float(row.get("total_commits", 0))
        churn_loc = commits * 30
        
        # 4. Recalcul de COCOMO
        kloc = max(net_loc / 1000.0, 0.1)
        cocomo_pm = 2.4 * (kloc ** 1.05)
        bus_factor_ratio = float(row.get("bus_factor_ratio", 0))
        if bus_factor_ratio > 0.7:
            cocomo_pm *= 1.2
        cocomo_hours = round(cocomo_pm * HOURS_PER_PM, 1)
        
        # 5. Recalcul des composantes de l'effort
        churn_hours = round(churn_loc / PRODUCTIVITY_LOC_PER_HOUR, 1)
        cycle_time_hours = float(row.get("cycle_time_hours", 0)) 
        
        # L'ancienne valeur était fausse si le scraper.py initial a échoué.
        # On recalcule proprement cycle_time_hours
        pr_median_h = float(row.get("pr_merge_time_median_h", 0))
        contribs = float(row.get("active_contributors", 1))
        cycle_time_hours = round(pr_median_h * max(contribs, 1), 1)
        
        # 6. Recalcul de l'Effort Target
        effort_target = round(0.5 * churn_hours + 0.3 * cycle_time_hours + 0.2 * cocomo_hours, 1)
        
        # Mettre à jour la ligne
        row["net_loc"] = round(net_loc, 1)
        row["churn_loc"] = round(churn_loc, 1)
        row["churn_hours"] = churn_hours
        row["cycle_time_hours"] = cycle_time_hours
        row["cocomo_pm"] = round(cocomo_pm, 2)
        row["cocomo_hours"] = cocomo_hours
        row["effort_target"] = effort_target
        
        # On corrige aussi le churn normalisé aberrant de 160.0
        active_days = max(float(row.get("active_days", 1)), 1)
        row["code_churn_normalized"] = round(churn_loc / active_days, 2)
        
        console.print(f"[green]OK[/] (Effort corrigé : {effort_target}h)")
        fixed_rows.append(row)
        
        # Petite pause pour respecter les limites de l'API
        time.sleep(0.5)

    # Sauvegarder dans le nouveau fichier (en mode ajout si existant)
    mode = "a" if os.path.exists(OUTPUT_CSV) else "w"
    with open(OUTPUT_CSV, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(fixed_rows)
        
    console.print(f"\n[bold green]Terminé ![/] Les données corrigées sont sauvegardées dans [cyan]{OUTPUT_CSV}[/].")

if __name__ == "__main__":
    main()
