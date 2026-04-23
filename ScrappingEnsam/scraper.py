"""
scraper.py — GitHub Scraper Distribué (Multi-PC, Multi-Token)
=============================================================

USAGE :
    1. Chaque membre du groupe lance ce script sur son PC
    2. Chaque membre met son propre token GitHub dans GH_TOKEN
    3. Chaque membre choisit son LOT (1, 2, 3, ou 4) dans MEMBER_LOT
    4. Le script génère features_raw_LOT{N}.csv + checkpoint_LOT{N}.json
    5. À la fin : merge_all.py pour fusionner tous les CSV

SETUP (à faire sur chaque PC) :
    pip install requests python-dateutil rich numpy pandas

VARIABLES À MODIFIER AVANT DE LANCER :
    - GH_TOKEN  : ton token GitHub personnel
    - MEMBER_LOT: ton numéro de lot (1, 2, 3 ou 4)
"""

import os, json, time, csv
from datetime import datetime, timezone
from typing import Optional
import requests
import numpy as np
from dateutil import parser as dparser
from rich.console import Console

console = Console()

# ══════════════════════════════════════════════════════════════════════════════
# ⚙️  CONFIGURATION — MODIFIER ICI AVANT DE LANCER
# ══════════════════════════════════════════════════════════════════════════════

GH_TOKEN   = os.getenv("GH_TOKEN", "")   # Ton token GitHub personnel
MEMBER_LOT = 1                            # ← CHANGER : 1, 2, 3 ou 4

OUTPUT_CSV       = f"features_raw_LOT{MEMBER_LOT}.csv"
CHECKPOINT_FILE  = f"checkpoint_LOT{MEMBER_LOT}.json"
CHECKPOINT_EVERY = 30

# ── Seuils qualité ────────────────────────────────────────────────────────────
MIN_COMMITS       = 200
MIN_CONTRIBUTORS  = 3
MAX_DAYS_INACTIVE = 180
MIN_STARS         = 30
MIN_CLOSED_ISSUES = 10

# ── Constantes ────────────────────────────────────────────────────────────────
PRODUCTIVITY_LOC_PER_HOUR = 15.0
HOURS_PER_PM              = 160

# ══════════════════════════════════════════════════════════════════════════════
# 📋  REQUÊTES PAR LOT — chaque membre scrappe un segment différent
# ══════════════════════════════════════════════════════════════════════════════

LOT_QUERIES = {
    1: [
        "stars:>1000 pushed:>2024-01-01 language:Python is:public forks:>200",
        "stars:>500  pushed:>2024-01-01 language:Python is:public forks:>100",
        "stars:>1000 pushed:>2024-01-01 language:JavaScript is:public forks:>200",
        "stars:>500  pushed:>2024-01-01 language:JavaScript is:public forks:>100",
    ],
    2: [
        "stars:>1000 pushed:>2024-01-01 language:TypeScript is:public forks:>200",
        "stars:>500  pushed:>2024-01-01 language:TypeScript is:public forks:>100",
        "stars:>1000 pushed:>2024-01-01 language:Java is:public forks:>200",
        "stars:>500  pushed:>2024-01-01 language:Java is:public forks:>100",
    ],
    3: [
        "stars:>500  pushed:>2024-01-01 language:Go is:public forks:>100",
        "stars:>200  pushed:>2024-03-01 language:Go is:public forks:>50",
        "stars:>500  pushed:>2024-01-01 language:Rust is:public forks:>100",
        "stars:>300  pushed:>2024-01-01 language:C++ is:public forks:>100",
    ],
    4: [
        "stars:>500  pushed:>2024-01-01 topic:machine-learning is:public",
        "stars:>300  pushed:>2024-01-01 topic:data-science is:public",
        "stars:>500  pushed:>2024-01-01 topic:devtools is:public",
        "stars:>200  pushed:>2024-06-01 topic:web-framework is:public",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 🌐  CLIENT GITHUB
# ══════════════════════════════════════════════════════════════════════════════

class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self._remaining = 5000
        self._reset_at  = 0

    def _get(self, path: str, params: dict = None, retries: int = 4):
        url = path if path.startswith("http") else f"{self.BASE}{path}"
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=20)
                self._remaining = int(r.headers.get("X-RateLimit-Remaining", 99))
                self._reset_at  = int(r.headers.get("X-RateLimit-Reset", 0))

                if self._remaining < 15:
                    wait = max(self._reset_at - time.time() + 3, 1)
                    console.print(f"[yellow]⏳ Rate limit → attente {int(wait)}s[/]")
                    time.sleep(wait)

                if r.status_code == 204: return {}
                if r.status_code == 404: return None
                if r.status_code == 202:
                    time.sleep(5)
                    continue
                if r.status_code in (403, 429):
                    wait = 60 * (attempt + 1)
                    console.print(f"[red]🚫 Rate limit (403/429) → attente {wait}s[/]")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()

            except requests.exceptions.Timeout:
                time.sleep(5)
            except Exception as e:
                if attempt == retries - 1:
                    console.print(f"[red]✗ {url}: {e}[/]")
                time.sleep(3)
        return None

    def get_rate_limit(self):
        r = self._get("/rate_limit")
        if r:
            core = r.get("resources", {}).get("core", {})
            console.print(f"[cyan]Rate limit → {core.get('remaining')}/{core.get('limit')} requêtes restantes[/]")

    def search_repos(self, query: str, per_page=50, max_pages=4) -> list:
        results = []
        for page in range(1, max_pages + 1):
            data = self._get("/search/repositories", {
                "q": query, "sort": "stars", "order": "desc",
                "per_page": per_page, "page": page,
            })
            if not data or "items" not in data:
                break
            results.extend(data["items"])
            if len(data["items"]) < per_page:
                break
            time.sleep(2)
        return results

    def get_code_frequency(self, full_name: str) -> list:
        data = self._get(f"/repos/{full_name}/stats/code_frequency")
        if not isinstance(data, list):
            time.sleep(5)
            data = self._get(f"/repos/{full_name}/stats/code_frequency")
        return data if isinstance(data, list) else []

    def get_commit_stats(self, full_name: str) -> dict:
        data = self._get(f"/repos/{full_name}/stats/contributors")
        if not isinstance(data, list):
            time.sleep(5)
            data = self._get(f"/repos/{full_name}/stats/contributors")
        if not isinstance(data, list):
            return {}
        filtered = [c for c in data if c.get("total", 0) >= 5]
        if not filtered:
            return {}
        total = sum(c.get("total", 0) for c in filtered)
        top   = max((c.get("total", 0) for c in filtered), default=0)
        return {"total_commits": total, "top_commits": top, "contributors": len(filtered)}

    def get_pr_stats(self, full_name: str) -> dict:
        prs = self._get(f"/repos/{full_name}/pulls", {
            "state": "closed", "per_page": 100, "sort": "updated"
        })
        if not isinstance(prs, list):
            return {"median_h": 0.0, "count": 0, "comment_avg": 0.0}
        merged_times, comment_counts = [], []
        for pr in prs:
            if pr.get("merged_at") and pr.get("created_at"):
                try:
                    t1 = dparser.parse(pr["created_at"])
                    t2 = dparser.parse(pr["merged_at"])
                    h  = (t2 - t1).total_seconds() / 3600
                    if 0.1 < h < 8760:
                        merged_times.append(h)
                    comment_counts.append(pr.get("comments", 0) + pr.get("review_comments", 0))
                except Exception:
                    pass
        return {
            "median_h":    float(np.median(merged_times)) if merged_times else 0.0,
            "count":       len(merged_times),
            "comment_avg": float(np.mean(comment_counts)) if comment_counts else 0.0,
        }

    def get_issues_resolution_time(self, full_name: str) -> float:
        issues = self._get(f"/repos/{full_name}/issues", {"state": "closed", "per_page": 100})
        if not isinstance(issues, list):
            return 0.0
        times = []
        for iss in issues:
            if iss.get("pull_request"):
                continue
            if iss.get("closed_at") and iss.get("created_at"):
                try:
                    t1 = dparser.parse(iss["created_at"])
                    t2 = dparser.parse(iss["closed_at"])
                    h  = (t2 - t1).total_seconds() / 3600
                    if 0.1 < h < 8760:
                        times.append(h)
                except Exception:
                    pass
        return float(np.median(times)) if times else 0.0

    def get_closed_issues_count(self, full_name: str) -> int:
        r = self._get("/search/issues", {
            "q": f"repo:{full_name} is:issue is:closed", "per_page": 1
        })
        return r.get("total_count", 0) if isinstance(r, dict) else 0

    def get_commit_velocity_trend(self, full_name: str) -> float:
        data = self._get(f"/repos/{full_name}/stats/participation")
        if not isinstance(data, dict):
            return 0.0
        all_commits = data.get("all", [])
        if len(all_commits) < 12:
            return 0.0
        recent = all_commits[-12:]
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)
        n = len(x)
        slope = (n * np.dot(x, y) - x.sum() * y.sum()) / (n * np.dot(x, x) - x.sum()**2 + 1e-9)
        return round(float(slope), 4)

    def get_release_regularity(self, full_name: str) -> float:
        releases = self._get(f"/repos/{full_name}/releases", {"per_page": 30})
        if not isinstance(releases, list) or len(releases) < 3:
            return 0.0
        dates = []
        for rel in releases:
            if rel.get("published_at"):
                try:
                    dates.append(dparser.parse(rel["published_at"]).timestamp())
                except Exception:
                    pass
        if len(dates) < 3:
            return 0.0
        dates.sort()
        intervals = [(dates[i+1] - dates[i]) / 86400 for i in range(len(dates)-1)]
        return round(float(np.std(intervals)), 2) if intervals else 0.0

    def get_weekend_commit_ratio(self, full_name: str) -> float:
        commits = self._get(f"/repos/{full_name}/commits", {"per_page": 100})
        if not isinstance(commits, list) or not commits:
            return 0.0
        weekend = sum(
            1 for c in commits
            if c.get("commit", {}).get("committer", {}) and
            dparser.parse(c["commit"]["committer"]["date"]).weekday() >= 5
        )
        return round(weekend / len(commits), 4)

    def get_ci_info(self, full_name: str) -> dict:
        workflows = self._get(f"/repos/{full_name}/actions/workflows", {"per_page": 10})
        has_ci = isinstance(workflows, dict) and len(workflows.get("workflows", [])) > 0
        ci_success_rate = 0.0
        if has_ci:
            runs = self._get(f"/repos/{full_name}/actions/runs", {"per_page": 50, "status": "completed"})
            if isinstance(runs, dict) and runs.get("workflow_runs"):
                all_runs = runs["workflow_runs"]
                if all_runs:
                    success = sum(1 for r in all_runs if r.get("conclusion") == "success")
                    ci_success_rate = round(success / len(all_runs), 4)
        return {"has_ci": int(has_ci), "ci_success_rate": ci_success_rate}

    def get_has_tests(self, full_name: str) -> int:
        for path in ("tests", "test", "__tests__", "spec"):
            if self._get(f"/repos/{full_name}/contents/{path}") is not None:
                return 1
        return 0

    def get_dependency_count(self, full_name: str) -> int:
        for dep_file in ["requirements.txt", "package.json", "Pipfile", "pyproject.toml"]:
            content = self._get(f"/repos/{full_name}/contents/{dep_file}")
            if isinstance(content, dict) and content.get("size", 0) > 0:
                return int(content.get("size", 0) / 1024 * 20)
        return 0

    def get_language_diversity(self, full_name: str) -> float:
        langs = self._get(f"/repos/{full_name}/languages")
        if not isinstance(langs, dict) or not langs:
            return 0.0
        total = sum(langs.values())
        main  = max(langs.values())
        return round(1 - (main / max(total, 1)), 4)

    def get_weighted_experience(self, full_name: str, limit=8) -> float:
        contribs = self._get(f"/repos/{full_name}/contributors", {"per_page": limit})
        if not isinstance(contribs, list):
            return 1.0
        w_num, w_den = 0.0, 0.0
        for c in contribs[:limit]:
            login = c.get("login")
            if not login:
                continue
            user = self._get(f"/users/{login}")
            if not isinstance(user, dict):
                time.sleep(0.2)
                continue
            try:
                yrs = (datetime.now(timezone.utc) - dparser.parse(user["created_at"])).days / 365.0
                n   = c.get("contributions", 1)
                w_num += yrs * n
                w_den += n
            except Exception:
                pass
            time.sleep(0.15)
        return round(w_num / w_den, 3) if w_den > 0 else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 🔧  EXTRACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(client: GitHubClient, repo: dict) -> Optional[dict]:
    name = repo["full_name"]

    # Métadonnées de base
    last_push = repo.get("pushed_at", "")
    days_inactive = 9999
    if last_push:
        pushed = dparser.parse(last_push)
        days_inactive = (datetime.now(timezone.utc) - pushed).days
    stars = repo.get("stargazers_count", 0)

    # Commits & contributeurs
    stats = client.get_commit_stats(name)
    total_commits = stats.get("total_commits", 0)
    contributors  = stats.get("contributors", 0)
    top_commits   = stats.get("top_commits", 0)

    # Filtre qualité 1
    if (total_commits < MIN_COMMITS or contributors < MIN_CONTRIBUTORS
            or days_inactive > MAX_DAYS_INACTIVE or stars < MIN_STARS):
        return None

    closed_issues = client.get_closed_issues_count(name)
    time.sleep(0.4)
    if closed_issues < MIN_CLOSED_ISSUES:
        return None

    # Bus factor
    bus_factor_ratio = round(top_commits / total_commits, 4) if total_commits > 0 else 1.0

    # LOC & churn
    code_freq = client.get_code_frequency(name)
    time.sleep(0.5)
    total_additions = sum(w[1] for w in code_freq if len(w) > 1 and w[1] > 0)
    total_deletions = sum(abs(w[2]) for w in code_freq if len(w) > 2)
    net_loc   = max(total_additions - total_deletions, 0)
    churn_loc = total_additions + total_deletions
    active_weeks = sum(1 for w in code_freq if len(w) > 1 and (abs(w[1]) + (abs(w[2]) if len(w) > 2 else 0)) > 0)
    active_days  = max(active_weeks * 5, 1)
    churn_normalized = round(churn_loc / active_days, 2)

    # Filtre qualité 2 — repos trop petits (sites statiques, projets vides)
    if net_loc < 100 or churn_loc < 500:
        return None

    # PR stats
    time.sleep(0.3)
    pr_stats = client.get_pr_stats(name)

    # Issues
    time.sleep(0.3)
    issues_resolution_time_h = client.get_issues_resolution_time(name)

    # Temporelles
    time.sleep(0.3)
    commit_velocity_trend = client.get_commit_velocity_trend(name)
    time.sleep(0.2)
    release_regularity = client.get_release_regularity(name)
    time.sleep(0.3)
    weekend_commit_ratio = client.get_weekend_commit_ratio(name)

    # Processus
    time.sleep(0.3)
    ci_info = client.get_ci_info(name)
    time.sleep(0.2)
    has_tests = client.get_has_tests(name)

    # Structure code
    time.sleep(0.2)
    dep_count = client.get_dependency_count(name)
    time.sleep(0.2)
    language_diversity = client.get_language_diversity(name)

    # Expérience équipe
    time.sleep(0.3)
    weighted_experience = client.get_weighted_experience(name)

    # COCOMO II
    kloc       = max(net_loc / 1000, 0.1)
    cocomo_pm  = 2.4 * (kloc ** 1.05)
    if bus_factor_ratio > 0.7:
        cocomo_pm *= 1.2
    cocomo_hours = round(cocomo_pm * HOURS_PER_PM, 1)

    # Composantes target (seront RETIRÉES de X avant ML — leakage)
    churn_hours      = round(churn_loc / PRODUCTIVITY_LOC_PER_HOUR, 1)
    cycle_time_hours = round(pr_stats["median_h"] * max(contributors, 1), 1)
    effort_target    = round(0.5 * churn_hours + 0.3 * cycle_time_hours + 0.2 * cocomo_hours, 1)

    # Filtrer outliers extrêmes
    if effort_target > 50_000_000 or effort_target <= 0:
        return None

    # Reliability score
    score  = min(stars / 1000 * 20, 20)
    score += min(contributors / 20 * 15, 15)
    score += min(total_commits / 1000 * 15, 15)
    score += max(0.0, 15 - days_inactive / 12)
    score += 10 if ci_info["has_ci"] else 0
    score += 5  if has_tests           else 0
    score += max(0.0, 10 - bus_factor_ratio * 10)

    return {
        # Identifiant
        "full_name":               name,
        "url":                     repo.get("html_url", ""),
        "language":                repo.get("language", ""),
        "stars":                   stars,
        "created_at":              repo.get("created_at", ""),
        "days_inactive":           days_inactive,
        "lot":                     MEMBER_LOT,

        # ── Features Tier 1 (dans X) ──────────────────────────────────────────
        "code_churn_normalized":    churn_normalized,
        "pr_merge_time_median_h":   pr_stats["median_h"],
        "issues_resolution_time_h": issues_resolution_time_h,
        "active_contributors":      contributors,
        "bus_factor_ratio":         bus_factor_ratio,

        # ── Features Tier 2 (dans X) ──────────────────────────────────────────
        "pr_count_merged":          pr_stats["count"],
        "comment_per_pr_avg":       pr_stats["comment_avg"],
        "closed_issues":            closed_issues,
        "has_ci":                   ci_info["has_ci"],
        "ci_success_rate":          ci_info["ci_success_rate"],
        "has_tests":                has_tests,
        "weighted_experience":      weighted_experience,
        "commit_velocity_trend":    commit_velocity_trend,
        "release_regularity":       release_regularity,
        "weekend_commit_ratio":     weekend_commit_ratio,
        "dependency_count":         dep_count,
        "language_diversity":       language_diversity,
        "avg_file_size_loc":        0.0,   # complété par SonarQube après scraping
        "total_commits":            total_commits,

        # ── Colonnes intermédiaires (⚠ RETIRER DE X avant ML) ────────────────
        "net_loc":           net_loc,
        "churn_loc":         churn_loc,
        "active_days":       active_days,
        "churn_hours":       churn_hours,        # composante target
        "cycle_time_hours":  cycle_time_hours,   # composante target
        "cocomo_pm":         round(cocomo_pm, 2),
        "cocomo_hours":      cocomo_hours,       # composante target

        # ── TARGET ────────────────────────────────────────────────────────────
        "effort_target":     effort_target,

        # ── Qualité ───────────────────────────────────────────────────────────
        "reliability_score": round(min(score, 100), 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 💾  CHECKPOINT & SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        console.print(f"[green]✓ Checkpoint — {len(data['done'])} repos déjà traités[/]")
        return data
    return {"done": [], "results": []}

def save_checkpoint(done, results):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"done": done, "results": results}, f)

def save_csv(results):
    if not results:
        return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    console.print(f"[green]✓ {OUTPUT_CSV} — {len(results)} repos[/]")


# ══════════════════════════════════════════════════════════════════════════════
# 🚀  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    console.rule(f"[bold blue]GitHub Scraper — LOT {MEMBER_LOT}[/]")

    if not GH_TOKEN:
        console.print("[red bold]⚠ GH_TOKEN non défini ![/]")
        console.print("  1. Va sur github.com → Settings → Developer Settings → Personal Access Tokens")
        console.print("  2. Generate new token (classic) → cocher : repo, read:user")
        console.print("  3. Lance : set GH_TOKEN=ton_token  (Windows) ou export GH_TOKEN=ton_token  (Linux/Mac)")
        return

    client = GitHubClient(GH_TOKEN)
    client.get_rate_limit()

    # Collecte des repos selon le lot
    queries = LOT_QUERIES.get(MEMBER_LOT, LOT_QUERIES[1])
    console.print(f"[cyan]Lot {MEMBER_LOT} — {len(queries)} requêtes[/]")

    all_repos = {}
    for q in queries:
        console.print(f"[dim]🔍 {q}[/]")
        repos = client.search_repos(q, per_page=50, max_pages=4)
        for r in repos:
            all_repos[r["full_name"]] = r
        time.sleep(2)

    console.print(f"[bold]{len(all_repos)} repos trouvés[/]")

    # Reprise depuis checkpoint si interrompu
    ckpt    = load_checkpoint()
    done    = set(ckpt["done"])
    results = ckpt["results"]

    to_process = [r for name, r in all_repos.items() if name not in done]
    console.print(f"[cyan]{len(to_process)} repos à traiter ({len(done)} déjà faits)[/]\n")

    passed = filtered = errors = 0

    for i, repo in enumerate(to_process):
        name = repo["full_name"]
        console.print(f"[dim]({i+1}/{len(to_process)})[/] [cyan]{name}[/]...", end=" ")

        try:
            feat = extract_features(client, repo)
            if feat:
                results.append(feat)
                passed += 1
                console.print(f"[green]✓[/] effort={feat['effort_target']:.0f}h")
            else:
                filtered += 1
                console.print("[dim]filtré[/]")
        except Exception as e:
            errors += 1
            console.print(f"[red]✗ {e}[/]")

        done.add(name)
        time.sleep(1.2)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(list(done), results)
            save_csv(results)
            console.print(f"\n[yellow]💾 Checkpoint — {passed} retenus / {filtered} filtrés / {errors} erreurs[/]\n")
            client.get_rate_limit()

    save_checkpoint(list(done), results)
    save_csv(results)

    console.rule("Résumé Final")
    console.print(f"  [green]Retenus  : {passed}[/]")
    console.print(f"  [dim]Filtrés  : {filtered}[/]")
    console.print(f"  [red]Erreurs  : {errors}[/]")
    console.print(f"\n[bold]✓ LOT {MEMBER_LOT} terminé — transmettre [cyan]{OUTPUT_CSV}[/] pour le merge[/]")


if __name__ == "__main__":
    main()
