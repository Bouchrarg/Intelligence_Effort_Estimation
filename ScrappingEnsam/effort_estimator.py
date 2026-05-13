"""
effort_estimator.py — Commit-Based Effort Estimation (Phase 1)
===============================================================

Replaces the synthetic effort_target formula with a scientifically
grounded estimate based on commit session reconstruction.

METHOD (session reconstruction, cited in literature):
    For each contributor:
        Group their commits into "sessions" (commits < SESSION_GAP_H apart)
        Session duration = last_commit - first_commit (or DEFAULT_SOLO_MIN if 1 commit)
    Total effort = sum of all session durations across all contributors

REFERENCES:
    - Teixeira et al. (2015) "Lessons learned from mining GitHub"
    - Robles et al. (2014) "Beyond contributors: Understanding OSS effort"
    - Huocalypse et al. (2012) "Estimating development effort in OSS projects"

USAGE:
    1. Drop this file next to your scraper.py
    2. Call add_real_effort_target(client, features_df) after scraping
    3. Use 'effort_hours_real' as your ML target instead of 'effort_target'

SETUP:
    pip install requests python-dateutil numpy pandas rich
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dateutil import parser as dparser
from collections import defaultdict
from rich.console import Console

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────
SESSION_GAP_H    = 2.0    # commits < 2h apart → same work session
DEFAULT_SOLO_MIN = 30     # single-commit session → assume 30 min of work
MAX_SESSION_H    = 8.0    # cap any session at 8h (avoids overnight outliers)
MAX_PAGES        = 10     # pages of commits to fetch per repo (100/page → 1000 commits)
MIN_EFFORT_H     = 10     # filter: repos with < 10 estimated hours are noise
MAX_EFFORT_H     = 500_000  # filter: > 500k hours is almost certainly an error


# ══════════════════════════════════════════════════════════════════════════════
# CORE SESSION RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_sessions(commit_timestamps: list[float]) -> float:
    """
    Given a sorted list of Unix timestamps for one contributor,
    reconstruct work sessions and return total estimated hours.

    A session ends when the gap to the next commit exceeds SESSION_GAP_H.
    A solo commit (gap before and after > SESSION_GAP_H) gets DEFAULT_SOLO_MIN.
    All sessions are capped at MAX_SESSION_H.
    """
    if not commit_timestamps:
        return 0.0

    timestamps = sorted(commit_timestamps)
    total_h = 0.0
    session_start = timestamps[0]
    session_last  = timestamps[0]

    for ts in timestamps[1:]:
        gap_h = (ts - session_last) / 3600.0
        if gap_h <= SESSION_GAP_H:
            # Still in the same session
            session_last = ts
        else:
            # Session ended — compute duration
            duration_h = (session_last - session_start) / 3600.0
            if duration_h < (DEFAULT_SOLO_MIN / 60):
                duration_h = DEFAULT_SOLO_MIN / 60   # solo commit minimum
            total_h += min(duration_h, MAX_SESSION_H)
            # Start new session
            session_start = ts
            session_last  = ts

    # Don't forget the last open session
    duration_h = (session_last - session_start) / 3600.0
    if duration_h < (DEFAULT_SOLO_MIN / 60):
        duration_h = DEFAULT_SOLO_MIN / 60
    total_h += min(duration_h, MAX_SESSION_H)

    return round(total_h, 2)


# ══════════════════════════════════════════════════════════════════════════════
# GITHUB DATA FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_commit_timestamps(client, full_name: str) -> dict[str, list[float]]:
    """
    Fetch up to MAX_PAGES × 100 commits and group their timestamps by author.

    Returns:
        { 'login_or_email': [unix_timestamp, ...], ... }

    Uses the /commits endpoint (always synchronous, no 202).
    Filters out bot accounts.
    """
    commits_by_author = defaultdict(list)
    page = 1

    while page <= MAX_PAGES:
        batch = client._get(
            f"/repos/{full_name}/commits",
            {"per_page": 100, "page": page}
        )
        if not isinstance(batch, list) or not batch:
            break

        for commit in batch:
            # Identify author — prefer login (GitHub account) over email
            author_login = None
            if commit.get("author") and commit["author"].get("login"):
                login = commit["author"]["login"]
                if login.endswith("[bot]") or "bot" in login.lower():
                    continue
                author_login = login
            else:
                # Fall back to git author email
                git_author = commit.get("commit", {}).get("author", {})
                email = git_author.get("email", "")
                if not email or "noreply" in email or "bot" in email:
                    continue
                author_login = email

            # Parse commit timestamp
            git_author = commit.get("commit", {}).get("author", {})
            date_str = git_author.get("date", "")
            if not date_str:
                continue
            try:
                ts = dparser.parse(date_str).timestamp()
                commits_by_author[author_login].append(ts)
            except Exception:
                continue

        if len(batch) < 100:
            break   # last page reached
        page += 1
        time.sleep(0.3)

    return dict(commits_by_author)


def estimate_effort_hours(client, full_name: str) -> dict:
    """
    Full pipeline for one repo:
        1. Fetch commit timestamps grouped by contributor
        2. Reconstruct sessions per contributor
        3. Sum across all contributors

    Returns a dict with the effort estimate and diagnostics.
    Returns None if the estimate is outside plausible bounds.
    """
    commits_by_author = fetch_commit_timestamps(client, full_name)

    if not commits_by_author:
        console.print(f"[yellow]  ⚠ No commits fetched for {full_name}[/]")
        return None

    contributor_hours = {}
    for author, timestamps in commits_by_author.items():
        h = reconstruct_sessions(timestamps)
        if h > 0:
            contributor_hours[author] = h

    if not contributor_hours:
        return None

    total_hours    = sum(contributor_hours.values())
    n_contributors = len(contributor_hours)
    median_h_per_contributor = float(np.median(list(contributor_hours.values())))
    n_commits_sampled = sum(len(v) for v in commits_by_author.values())

    # Plausibility filter
    if total_hours < MIN_EFFORT_H or total_hours > MAX_EFFORT_H:
        console.print(
            f"[dim]  → effort outlier filtered: {full_name} → {total_hours:.0f}h[/]"
        )
        return None

    return {
        "effort_hours_real":          round(total_hours, 1),
        "effort_contributors_seen":   n_contributors,
        "effort_commits_sampled":     n_commits_sampled,
        "effort_median_h_per_contrib": median_h_per_contributor,
        "effort_pages_fetched":       min(n_commits_sampled // 100 + 1, MAX_PAGES),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH ENRICHMENT — run after your existing scrape
# ══════════════════════════════════════════════════════════════════════════════

def add_real_effort_target(client, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes your existing features DataFrame (output of merge_all.py),
    fetches commit-based effort estimates for each repo,
    and returns an enriched DataFrame with 'effort_hours_real' as the new target.

    Repos where effort cannot be estimated are dropped.

    Usage:
        from effort_estimator import add_real_effort_target
        from scraper import GitHubClient
        import pandas as pd

        client = GitHubClient(GH_TOKEN)
        df = pd.read_csv('features_merged.csv')
        df_enriched = add_real_effort_target(client, df)
        df_enriched.to_csv('features_with_real_effort.csv', index=False)
    """
    console.rule("[bold blue]Phase 1 — Commit-Based Effort Estimation[/]")
    console.print(f"Processing {len(features_df)} repos...")

    effort_records = []
    failed = 0

    for i, (_, row) in enumerate(features_df.iterrows()):
        full_name = row["full_name"]
        console.print(
            f"[dim]({i+1}/{len(features_df)})[/] [cyan]{full_name}[/]...",
            end=" "
        )

        result = estimate_effort_hours(client, full_name)

        if result:
            result["full_name"] = full_name
            effort_records.append(result)
            console.print(
                f"[green]✓[/] {result['effort_hours_real']:.0f}h  "
                f"({result['effort_contributors_seen']} contribs, "
                f"{result['effort_commits_sampled']} commits)"
            )
        else:
            failed += 1
            console.print("[red]✗ skipped[/]")

        time.sleep(0.5)   # gentle rate limiting

    console.rule("Summary")
    console.print(f"  [green]Enriched : {len(effort_records)}[/]")
    console.print(f"  [red]Skipped  : {failed}[/]")

    if not effort_records:
        raise RuntimeError("No effort estimates produced. Check your token and repo names.")

    effort_df = pd.DataFrame(effort_records)

    # Merge back onto features
    enriched = features_df.merge(effort_df, on="full_name", how="inner")
    console.print(f"\n[bold]✓ Enriched DataFrame: {enriched.shape[0]} repos retained[/]")

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compare_targets(enriched_df: pd.DataFrame):
    """
    Compare the new commit-based target against the old synthetic target.
    Prints correlation and plots distributions side by side.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    old = enriched_df["effort_target"]
    new = enriched_df["effort_hours_real"]

    corr, p = stats.spearmanr(old, new)
    print(f"\nSpearman correlation (synthetic vs real): r={corr:.3f}  p={p:.4f}")
    print(
        f"{'Strong agreement' if corr > 0.7 else 'Moderate agreement' if corr > 0.4 else 'Weak agreement — targets diverge significantly'}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Distribution comparison
    axes[0].hist(np.log1p(old), bins=40, alpha=0.6, label='Synthetic', color='steelblue')
    axes[0].hist(np.log1p(new), bins=40, alpha=0.6, label='Commit-based', color='darkorange')
    axes[0].set_title('Target Distributions (log scale)')
    axes[0].set_xlabel('log(hours + 1)')
    axes[0].legend()

    # Scatter
    axes[1].scatter(np.log1p(old), np.log1p(new), alpha=0.4, s=15, color='teal')
    axes[1].set_xlabel('log(synthetic target)')
    axes[1].set_ylabel('log(commit-based target)')
    axes[1].set_title(f'Synthetic vs Commit-Based\nSpearman r={corr:.3f}')

    # Commit-based distribution
    axes[2].hist(np.log1p(new), bins=40, color='darkorange', edgecolor='white')
    axes[2].set_title('Commit-Based Effort Distribution')
    axes[2].set_xlabel('log(hours + 1)')

    plt.tight_layout()
    plt.savefig('target_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: target_comparison.png")

    return corr


def describe_new_target(enriched_df: pd.DataFrame):
    """Print summary statistics for the new target."""
    t = enriched_df["effort_hours_real"]
    print("\n── Commit-Based Effort Target Stats ──")
    print(f"  Count  : {len(t)}")
    print(f"  Mean   : {t.mean():,.0f} h")
    print(f"  Median : {t.median():,.0f} h")
    print(f"  Std    : {t.std():,.0f} h")
    print(f"  Min    : {t.min():,.0f} h")
    print(f"  Max    : {t.max():,.0f} h")
    print(f"  Skew   : {t.skew():.2f}")
    print(f"  Log skew: {np.log1p(t).skew():.2f}  (should be < 1.0 for log-normal)")


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE SCRIPT — run directly to enrich your existing CSV
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import sys

    # ── Config ────────────────────────────────────────────────────────────────
    GH_TOKEN    = os.getenv("GH_TOKEN")
    INPUT_CSV   = "features_merged.csv"       # your existing merged file
    OUTPUT_CSV  = "features_with_real_effort.csv"

    if not GH_TOKEN:
        print("ERROR: GH_TOKEN environment variable not set.")
        print("  export GH_TOKEN=your_token_here")
        sys.exit(1)

    # Import your existing client
    try:
        from scraper import GitHubClient
    except ImportError:
        print("ERROR: scraper.py not found in the same directory.")
        sys.exit(1)

    client = GitHubClient(GH_TOKEN)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} repos from {INPUT_CSV}")

    # ── Run enrichment ────────────────────────────────────────────────────────
    enriched = add_real_effort_target(client, df)

    # ── Validate ──────────────────────────────────────────────────────────────
    describe_new_target(enriched)
    corr = compare_targets(enriched)

    # ── Save ──────────────────────────────────────────────────────────────────
    enriched.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved enriched dataset → {OUTPUT_CSV}")
    print(f"  Shape: {enriched.shape}")
    print(f"\nNext step: update your ML notebook to use 'effort_hours_real' as the target.")
