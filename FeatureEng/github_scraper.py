"""
github_scraper.py
=================
Scrape des vrais repos GitHub et calcule les 17 features + target composite.

Prérequis:
    pip install requests pandas numpy tqdm

Token GitHub (OBLIGATOIRE pour éviter rate limit 60 req/h → 5000 req/h):
    1. https://github.com/settings/tokens → Generate new token (classic)
    2. Cocher : repo (public_repo suffit), read:user
    3. export GITHUB_TOKEN=ghp_xxxxx   (ou passer --token ghp_xxxxx)

Usage:
    python github_scraper.py --token ghp_xxx --repos 50 --output dataset.csv
    python github_scraper.py --token ghp_xxx --repos 20 --lang python
    python github_scraper.py --token ghp_xxx --file mes_repos.txt   # liste de "owner/repo" 1 par ligne

Durée estimée : ~3 min pour 50 repos (rate limit respecté automatiquement)
"""

import argparse
import os
import sys
import time
import math
import json
import logging
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─── CONSTANTES ────────────────────────────────────────────────────────────────
BASE_URL = "https://api.github.com"
HEADERS_BASE = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

# Productivité moyenne industrie (LOC/heure) — [Harding, GitClear 2021]
LOC_PER_HOUR = 15

# Poids target composite — [notre formule]
W_CHURN, W_CYCLE, W_COCOMO = 0.5, 0.3, 0.2


# ─── CLIENT API ────────────────────────────────────────────────────────────────
class GitHubClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({**HEADERS_BASE, "Authorization": f"Bearer {token}"})
        self._remaining = 5000
        self._reset_at = 0

    def _throttle(self):
        if self._remaining < 50:
            wait = max(0, self._reset_at - time.time()) + 2
            log.warning(f"Rate limit bas ({self._remaining} restants) — pause {wait:.0f}s")
            time.sleep(wait)

    def get(self, path: str, params: dict = None, retries=3) -> Optional[dict]:
        self._throttle()
        url = path if path.startswith("http") else f"{BASE_URL}{path}"
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=15)
                self._remaining = int(r.headers.get("X-RateLimit-Remaining", self._remaining))
                self._reset_at = int(r.headers.get("X-RateLimit-Reset", self._reset_at))
                if r.status_code == 404:
                    return None
                if r.status_code == 403 and "rate limit" in r.text.lower():
                    wait = max(0, self._reset_at - time.time()) + 5
                    log.warning(f"Rate limit atteint — pause {wait:.0f}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt == retries - 1:
                    log.debug(f"Échec {url}: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def get_paginated(self, path: str, params: dict = None, max_pages=5) -> list:
        params = params or {}
        params.setdefault("per_page", 100)
        results = []
        url = path if path.startswith("http") else f"{BASE_URL}{path}"
        for _ in range(max_pages):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=15)
                self._remaining = int(r.headers.get("X-RateLimit-Remaining", self._remaining))
                self._reset_at = int(r.headers.get("X-RateLimit-Reset", self._reset_at))
                if r.status_code != 200:
                    break
                data = r.json()
                if isinstance(data, dict):
                    data = data.get("items", data.get("values", []))
                if not data:
                    break
                results.extend(data)
                link = r.headers.get("Link", "")
                if 'rel="next"' not in link:
                    break
                import re
                nxt = re.search(r'<([^>]+)>;\s*rel="next"', link)
                if not nxt:
                    break
                url = nxt.group(1)
                params = {}  # URL already has params
            except Exception:
                break
        return results


# ─── DÉCOUVERTE DE REPOS ───────────────────────────────────────────────────────
def discover_repos(client: GitHubClient, n: int, lang: str = None) -> list:
    """Cherche n repos actifs, variés, avec CI et tests."""
    log.info(f"Recherche de {n} repos GitHub...")
    queries = [
        f"stars:200..10000 forks:50..5000 pushed:>2023-01-01{' language:' + lang if lang else ''}",
        f"stars:50..500 forks:10..500 pushed:>2023-06-01 has:issues{' language:' + lang if lang else ''}",
        f"stars:500..5000 size:>500 pushed:>2023-01-01{' language:' + lang if lang else ''}",
    ]
    seen = set()
    repos = []
    for q in queries:
        if len(repos) >= n * 2:
            break
        data = client.get("/search/repositories", params={
            "q": q, "sort": "updated", "order": "desc", "per_page": min(50, n * 2)
        })
        if not data:
            continue
        for item in data.get("items", []):
            full = item["full_name"]
            if full not in seen and not item.get("fork", False) and not item.get("archived", False):
                seen.add(full)
                repos.append(full)
    log.info(f"  {len(repos)} repos candidats trouvés")
    return repos[:n * 2]  # on prend le double pour compenser les échecs


# ─── CALCUL DES FEATURES ──────────────────────────────────────────────────────
def compute_features(client: GitHubClient, full_name: str) -> Optional[dict]:
    """Calcule toutes les 17 features + target pour un repo donné."""
    owner, repo = full_name.split("/", 1)

    # ① Info de base
    info = client.get(f"/repos/{owner}/{repo}")
    if not info:
        return None

    # Fenêtre temporelle : 6 derniers mois actifs
    pushed = info.get("pushed_at", "")
    if not pushed:
        return None
    try:
        last_push = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
        days_since = (datetime.now(timezone.utc) - last_push).days
        if days_since > 730:  # inactif depuis 2 ans → skip
            return None
    except Exception:
        return None

    row = {"repo": full_name}

    # ─ ② Commits (200 max, 2 pages) pour churn et vélocité ─
    commits = client.get_paginated(f"/repos/{owner}/{repo}/commits",
                                   params={"per_page": 100}, max_pages=2)
    nb_commits = len(commits)
    if nb_commits < 10:  # trop petit
        return None

    # Date du commit le plus vieux de notre fenêtre
    dates = []
    for c in commits:
        try:
            d = c["commit"]["author"]["date"]
            dates.append(datetime.fromisoformat(d.replace("Z", "+00:00")))
        except Exception:
            pass

    if len(dates) < 2:
        return None
    dates.sort()
    nb_jours_actifs = max(1, (dates[-1] - dates[0]).days)

    # ─ ③ Stats de code via /stats/contributors ─
    stats = client.get(f"/repos/{owner}/{repo}/stats/contributors")
    total_add, total_del = 0, 0
    contrib_commits = {}
    if isinstance(stats, list):
        for contrib in stats:
            login = contrib.get("author", {}).get("login", "bot")
            if "bot" in login.lower():
                continue
            c_total = contrib.get("total", 0)
            contrib_commits[login] = c_total
            for week in contrib.get("weeks", []):
                total_add += week.get("a", 0)
                total_del += week.get("d", 0)
    else:
        # stats pas dispo (repo trop récent) → estimation via LOC
        total_add = info.get("size", 1000) * 5  # approximation
        total_del = total_add // 3

    total_churn = total_add + total_del
    code_churn_normalized = total_churn / nb_jours_actifs if nb_jours_actifs > 0 else 0
    row["code_churn_normalized"] = round(code_churn_normalized, 4)

    # ─ ④ Complexité via langages ─
    langs = client.get(f"/repos/{owner}/{repo}/languages") or {}
    total_loc = sum(langs.values()) if langs else 1
    main_lang_loc = max(langs.values()) if langs else total_loc
    language_diversity = 1 - (main_lang_loc / total_loc) if total_loc > 0 else 0
    row["language_diversity"] = round(language_diversity, 4)

    # Complexité cyclomatique approx via taille moyenne des fichiers
    # (radon n'est pas dispo via API — on utilise size/nb_files comme proxy)
    nb_files_est = max(1, total_loc // 200)  # estimation ~200 LOC/fichier
    avg_file_size = total_loc / nb_files_est
    row["avg_file_size_loc"] = round(min(avg_file_size, 5000), 2)
    # Complexité cyclomatique : proxy basé sur LOC et diversité langages
    row["cyclomatic_complexity_avg"] = round(2.0 + language_diversity * 8 + avg_file_size / 500, 2)

    # ─ ⑤ Dépendances ─
    dep_count = 0
    for dep_file in ["requirements.txt", "package.json", "Pipfile", "pyproject.toml", "pom.xml"]:
        dep_content = client.get(f"/repos/{owner}/{repo}/contents/{dep_file}")
        if dep_content and isinstance(dep_content, dict):
            import base64
            try:
                content = base64.b64decode(dep_content.get("content", "")).decode("utf-8", errors="ignore")
                if dep_file == "package.json":
                    data = json.loads(content)
                    dep_count += len(data.get("dependencies", {})) + len(data.get("devDependencies", {}))
                else:
                    dep_count += sum(1 for line in content.splitlines()
                                     if line.strip() and not line.startswith("#") and not line.startswith("-"))
            except Exception:
                dep_count += 5  # fallback
            break
    row["dependency_count"] = dep_count

    # ─ ⑥ Collaboration ─
    all_logins = list(contrib_commits.keys())
    active_contributors = sum(1 for v in contrib_commits.values() if v >= 5)
    row["active_contributors"] = max(1, active_contributors)

    total_contrib_commits = sum(contrib_commits.values()) or 1
    top_commit = max(contrib_commits.values()) if contrib_commits else total_contrib_commits
    bus_factor_ratio = top_commit / total_contrib_commits
    row["bus_factor_ratio"] = round(bus_factor_ratio, 4)

    # Expérience pondérée : proxy via ancienneté des contributeurs (nb de repos publics)
    # On ne fait pas N appels par contributeur (trop lent) — on estime via commits/actif
    row["weighted_experience"] = round(np.log1p(active_contributors) * 2 + bus_factor_ratio, 3)

    # ─ ⑦ Pull Requests ─
    prs_closed = client.get_paginated(f"/repos/{owner}/{repo}/pulls",
                                      params={"state": "closed", "sort": "updated", "per_page": 100},
                                      max_pages=1)

    pr_merge_times = []
    pr_cycle_counts = []
    pr_comment_counts = []
    pr_count_merged = 0

    for pr in prs_closed[:50]:
        if not pr.get("merged_at"):
            continue
        pr_count_merged += 1
        try:
            created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
            merged  = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
            merge_h = (merged - created).total_seconds() / 3600
            if merge_h > 0:
                pr_merge_times.append(merge_h)
        except Exception:
            pass
        pr_comment_counts.append(pr.get("comments", 0) + pr.get("review_comments", 0))

    row["pr_count_merged"] = pr_count_merged
    row["pr_merge_time_median_h"] = round(np.median(pr_merge_times), 2) if pr_merge_times else 48.0
    row["comment_per_pr_avg"] = round(np.mean(pr_comment_counts), 2) if pr_comment_counts else 0.0
    row["review_cycle_count"] = round(np.mean(pr_cycle_counts), 2) if pr_cycle_counts else 1.0

    # ─ ⑧ Issues ─
    issues_closed = client.get_paginated(f"/repos/{owner}/{repo}/issues",
                                         params={"state": "closed", "per_page": 100},
                                         max_pages=1)
    issue_times = []
    for iss in issues_closed[:50]:
        if iss.get("pull_request"):
            continue
        try:
            opened = datetime.fromisoformat(iss["created_at"].replace("Z", "+00:00"))
            closed = datetime.fromisoformat(iss["closed_at"].replace("Z", "+00:00"))
            h = (closed - opened).total_seconds() / 3600
            if h > 0:
                issue_times.append(h)
        except Exception:
            pass
    row["issues_resolution_time_h"] = round(np.median(issue_times), 2) if issue_times else 72.0

    # ─ ⑨ Vélocité temporelle (commit par semaine sur 12 semaines) ─
    weekly = {}
    for d in dates[-84:]:  # 12 semaines = 84 jours
        week_num = d.isocalendar()[1]
        weekly[week_num] = weekly.get(week_num, 0) + 1
    counts = list(weekly.values())
    if len(counts) >= 3:
        x = np.arange(len(counts))
        slope = np.polyfit(x, counts, 1)[0]
    else:
        slope = 0.0
    row["commit_velocity_trend"] = round(float(slope), 4)

    # Weekend commits
    weekend = sum(1 for d in dates if d.weekday() >= 5)
    row["weekend_commit_ratio"] = round(weekend / max(1, len(dates)), 4)

    # ─ ⑩ Releases ─
    releases = client.get_paginated(f"/repos/{owner}/{repo}/releases",
                                    params={"per_page": 30}, max_pages=1)
    if len(releases) >= 2:
        rel_dates = []
        for r in releases:
            try:
                rel_dates.append(datetime.fromisoformat(r["published_at"].replace("Z", "+00:00")))
            except Exception:
                pass
        if len(rel_dates) >= 2:
            rel_dates.sort()
            intervals = [(rel_dates[i+1] - rel_dates[i]).days for i in range(len(rel_dates)-1)]
            row["release_regularity"] = round(float(np.std(intervals)), 2)
        else:
            row["release_regularity"] = 30.0
    else:
        row["release_regularity"] = 60.0

    # ─ ⑪ CI / Tests ─
    workflows = client.get(f"/repos/{owner}/{repo}/contents/.github/workflows")
    has_ci = 1 if isinstance(workflows, list) and len(workflows) > 0 else 0
    row["has_ci"] = has_ci

    tests_dir = None
    for test_path in ["tests", "__tests__", "test", "spec"]:
        t = client.get(f"/repos/{owner}/{repo}/contents/{test_path}")
        if isinstance(t, list):
            tests_dir = True
            break
    row["has_tests"] = 1 if tests_dir else 0

    # CI success rate via check runs (sur les 20 derniers commits)
    if has_ci and commits:
        success_count, total_runs = 0, 0
        for c in commits[:20]:
            sha = c.get("sha", "")
            if not sha:
                continue
            runs = client.get(f"/repos/{owner}/{repo}/commits/{sha}/check-runs")
            if isinstance(runs, dict):
                for run in runs.get("check_runs", []):
                    total_runs += 1
                    if run.get("conclusion") == "success":
                        success_count += 1
        row["ci_success_rate"] = round(success_count / total_runs, 4) if total_runs > 0 else 0.85
    else:
        row["ci_success_rate"] = 0.5 if not has_ci else 0.85

    # ─ ⑫ TARGET COMPOSITE ─
    net_loc = total_loc / 1000  # en KLOC pour COCOMO
    churn_hours = total_churn / LOC_PER_HOUR
    cycle_time_hours = row["pr_merge_time_median_h"] * row["active_contributors"]
    cocomo_pm = 2.4 * (max(net_loc, 0.1) ** 1.05)
    cocomo_hours = cocomo_pm * 160

    effort_target = W_CHURN * churn_hours + W_CYCLE * cycle_time_hours + W_COCOMO * cocomo_hours

    row["churn_hours"] = round(churn_hours, 2)
    row["cycle_time_hours"] = round(cycle_time_hours, 2)
    row["cocomo_hours"] = round(cocomo_hours, 2)
    row["effort_target"] = round(effort_target, 2)
    row["net_kloc"] = round(net_loc, 3)
    row["nb_jours_actifs"] = nb_jours_actifs

    return row


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GitHub scraper — 17 features + target composite")
    parser.add_argument("--token",  type=str, default=os.environ.get("GITHUB_TOKEN"),
                        help="Token GitHub (ou export GITHUB_TOKEN=...)")
    parser.add_argument("--repos",  type=int, default=50,  help="Nombre de repos cibles (défaut: 50)")
    parser.add_argument("--lang",   type=str, default=None, help="Langage filtre (python, javascript...)")
    parser.add_argument("--file",   type=str, default=None, help="Fichier texte avec 'owner/repo' (1 par ligne)")
    parser.add_argument("--output", type=str, default="dataset_github.csv", help="Fichier CSV de sortie")
    args = parser.parse_args()

    if not args.token:
        print("[ERREUR] Token GitHub requis.")
        print("  → https://github.com/settings/tokens → Generate new token (classic)")
        print("  → export GITHUB_TOKEN=ghp_xxxx")
        print("  → python github_scraper.py --token ghp_xxxx")
        sys.exit(1)

    client = GitHubClient(args.token)

    # Vérification du token
    me = client.get("/user")
    if not me:
        print("[ERREUR] Token invalide ou expiré.")
        sys.exit(1)
    log.info(f"Connecté en tant que: @{me.get('login')} ({client._remaining} req restantes)")

    # Liste de repos
    if args.file:
        with open(args.file) as f:
            repo_list = [line.strip() for line in f if "/" in line.strip()]
        log.info(f"Chargé {len(repo_list)} repos depuis {args.file}")
    else:
        repo_list = discover_repos(client, args.repos, args.lang)

    # Scraping
    results = []
    errors = 0
    n_target = args.repos

    try:
        from tqdm import tqdm
        iterator = tqdm(repo_list[:n_target * 2], desc="Scraping repos", unit="repo")
    except ImportError:
        iterator = repo_list[:n_target * 2]
        log.info("(pip install tqdm pour une barre de progression)")

    for full_name in iterator:
        if len(results) >= n_target:
            break
        try:
            row = compute_features(client, full_name)
            if row:
                results.append(row)
                log.info(f"  ✓ {full_name} → effort={row['effort_target']:.0f}h | "
                         f"R²_proxy=churn={row['code_churn_normalized']:.1f} | "
                         f"{len(results)}/{n_target}")
            else:
                errors += 1
        except Exception as e:
            log.debug(f"  ✗ {full_name}: {e}")
            errors += 1

    if not results:
        print("[ERREUR] Aucun repo scraé avec succès. Vérifiez le token et la connexion.")
        sys.exit(1)

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"  SCRAPING TERMINÉ")
    print(f"{'='*60}")
    print(f"  Repos scrappés : {len(results)} / {len(results) + errors} tentatives")
    print(f"  Fichier CSV    : {args.output}")
    print(f"  Shape          : {df.shape}")
    print(f"\n  Aperçu des métriques :")
    print(f"  effort_target  : {df['effort_target'].describe()[['mean','min','max']].to_string()}")
    print(f"  churn_norm     : {df['code_churn_normalized'].describe()[['mean','min','max']].to_string()}")
    print(f"  active_contrib : {df['active_contributors'].describe()[['mean','min','max']].to_string()}")
    print(f"\n  Prochaines étapes :")
    print(f"  python leakage_check.py --data {args.output}")
    print(f"  python weight_validation.py --data {args.output}")
    print(f"  python target_comparison.py --data {args.output}")
    print(f"  python full_evaluation.py --data {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
