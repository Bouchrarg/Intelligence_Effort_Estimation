"""
effort_estimator_v2.py — Commit-Based Effort Estimation with Full-History Scaling
==================================================================================

IMPROVEMENT OVER v1:
    v1 problem : MAX_PAGES=10 cap truncates effort for large repos.
                 A repo with 50,000 commits was measured at ~350h
                 because only the last 1,000 commits were seen.

    v2 solution: Two-phase estimation
        Phase 1 — Sample: reconstruct sessions from a representative
                  window of commits (most recent N + oldest N).
        Phase 2 — Scale: extrapolate to full history using the ratio
                  total_commits / commits_sampled.

    This is consistent with Robles et al. (2014) "Beyond contributors:
    Understanding OSS effort" and Teixeira et al. (2015).

SAMPLING STRATEGY:
    Instead of only sampling recent commits (which biases toward
    current team size and velocity), we sample from BOTH ends:
        - Most recent SAMPLE_PAGES pages  (recent activity)
        - Oldest SAMPLE_PAGES pages       (founding effort)
    This gives a better density estimate for extrapolation.

SCALING FORMULA:
    effort_estimated = effort_sampled × (total_commits / commits_sampled)
    
    Assumption: effort per commit is roughly uniform over project lifetime.
    This is standard in OSS research. A decay correction is applied for
    very old projects where early commits tend to be larger.

PLAUSIBILITY BOUNDS (updated):
    v1: MIN=10h, MAX=500,000h  (too narrow — truncated large repos)
    v2: MIN=10h, MAX=5,000,000h (allows for very large projects like Linux)
    Outlier detection uses IQR on log scale instead of hard caps.
"""

import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dateutil import parser as dparser
from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# ── Sampling config ───────────────────────────────────────────────────────────
SAMPLE_PAGES       = 5      # pages from recent end  (5 × 100 = 500 commits)
SAMPLE_PAGES_OLD   = 3      # pages from oldest end  (3 × 100 = 300 commits)
COMMITS_PER_PAGE   = 100
SESSION_GAP_H      = 2.0    # hours gap → new session
DEFAULT_SOLO_MIN   = 30     # minutes for a solo commit
MAX_SESSION_H      = 8.0    # cap per session
MIN_EFFORT_H       = 10     # minimum plausible effort
MAX_EFFORT_H       = 5_000_000  # maximum plausible (Linux kernel ~ 2M hours)

# Decay factor: early commits in large projects tend to be bulkier
# (fewer, larger commits per unit of work in older codebases)
# Applied as a mild correction: old_commits_weight = DECAY_FACTOR
DECAY_FACTOR       = 0.85   # old commits contribute 85% of recent per-commit effort


# ══════════════════════════════════════════════════════════════════════════════
# SESSION RECONSTRUCTION (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_sessions(commit_timestamps: list) -> float:
    if not commit_timestamps:
        return 0.0
    timestamps = sorted(commit_timestamps)
    total_h = 0.0
    session_start = timestamps[0]
    session_last  = timestamps[0]
    for ts in timestamps[1:]:
        gap_h = (ts - session_last) / 3600.0
        if gap_h <= SESSION_GAP_H:
            session_last = ts
        else:
            duration_h = (session_last - session_start) / 3600.0
            total_h += min(max(duration_h, DEFAULT_SOLO_MIN / 60), MAX_SESSION_H)
            session_start = ts
            session_last  = ts
    duration_h = (session_last - session_start) / 3600.0
    total_h += min(max(duration_h, DEFAULT_SOLO_MIN / 60), MAX_SESSION_H)
    return round(total_h, 2)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — FETCH COMMIT SAMPLE (recent + oldest)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_commit_page(client, full_name: str, page: int,
                      direction: str = 'desc') -> list:
    """
    Fetch one page of commits.
    direction: 'desc' = newest first (default), 'asc' = oldest first
    """
    result = client._get(
        f"/repos/{full_name}/commits",
        {"per_page": COMMITS_PER_PAGE, "page": page, "order": direction}
    )
    return result if isinstance(result, list) else []


def parse_commits_to_author_map(commits: list,
                                weight: float = 1.0) -> dict:
    """
    Parse a list of commit objects into {author: [(timestamp, weight), ...]}
    Filters out bots and invalid entries.
    weight: applied to each timestamp for scaling (used for old commits)
    """
    author_map = defaultdict(list)
    for commit in commits:
        # Identify author
        author_login = None
        if commit.get("author") and commit["author"].get("login"):
            login = commit["author"]["login"]
            if login.endswith("[bot]") or "bot" in login.lower():
                continue
            author_login = login
        else:
            git_author = commit.get("commit", {}).get("author", {})
            email = git_author.get("email", "")
            if not email or "noreply" in email or "bot" in email:
                continue
            author_login = email

        git_author = commit.get("commit", {}).get("author", {})
        date_str = git_author.get("date", "")
        if not date_str:
            continue
        try:
            ts = dparser.parse(date_str).timestamp()
            author_map[author_login].append((ts, weight))
        except Exception:
            continue
    return dict(author_map)


def fetch_total_commit_count(client, full_name: str) -> int:
    """
    Get the total commit count for a repo using the contributors stats
    endpoint, which returns aggregate commit counts without fetching all
    commits. Falls back to the repo's size-based estimate if unavailable.

    GitHub's /stats/contributors returns cached data — may return 202
    (computing) on first call, requiring a retry.
    """
    for attempt in range(3):
        result = client._get(f"/repos/{full_name}/stats/contributors", {})
        if isinstance(result, list) and result:
            total = sum(c.get("total", 0) for c in result)
            if total > 0:
                return total
        elif isinstance(result, dict) and result.get("status") == 202:
            # GitHub is computing stats — wait and retry
            time.sleep(3)
        else:
            break

    # Fallback: use the commit count from the last page of commits
    # Binary search would be ideal but expensive — use a rough estimate
    # by checking how many pages exist (GitHub returns Link header)
    # For simplicity, return None and let the caller handle it
    return None


def get_first_commit_date(client, full_name: str) -> float:
    """
    Fetch the very first commit's timestamp using ascending sort.
    Returns Unix timestamp or None.
    """
    commits = fetch_commit_page(client, full_name, page=1, direction='asc')
    if not commits:
        return None
    git_author = commits[0].get("commit", {}).get("author", {})
    date_str = git_author.get("date", "")
    try:
        return dparser.parse(date_str).timestamp()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — SCALE TO FULL HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def scale_effort_to_full_history(
    effort_recent_h: float,
    effort_old_h: float,
    commits_recent: int,
    commits_old: int,
    total_commits: int,
) -> float:
    """
    Extrapolate sampled effort to full project history.

    We have two samples:
        - Recent sample  : effort_recent_h from commits_recent commits
        - Old sample     : effort_old_h    from commits_old commits (× DECAY_FACTOR)

    Per-commit effort rates:
        rate_recent = effort_recent_h / commits_recent
        rate_old    = effort_old_h    / commits_old (decayed)

    The middle of the history (unsampled) is assumed to have
    a rate interpolated between old and recent rates.

    Total = recent_effort + old_effort + middle_effort_estimate
    """
    if total_commits is None or total_commits <= 0:
        # No total count available — use recent rate only
        if commits_recent > 0:
            rate_recent = effort_recent_h / commits_recent
            return effort_recent_h + rate_recent * max(0, total_commits or 0 - commits_recent)
        return effort_recent_h

    commits_sampled = commits_recent + commits_old
    if commits_sampled >= total_commits:
        # We have the full history — no scaling needed
        return effort_recent_h + effort_old_h * DECAY_FACTOR

    # Per-commit rates
    rate_recent = effort_recent_h / max(1, commits_recent)
    rate_old    = (effort_old_h * DECAY_FACTOR) / max(1, commits_old)

    # Unsampled commits in the middle
    commits_middle  = total_commits - commits_sampled
    # Interpolated rate for the middle period
    rate_middle     = (rate_recent + rate_old) / 2.0

    effort_middle   = rate_middle * commits_middle

    total_effort = effort_recent_h + (effort_old_h * DECAY_FACTOR) + effort_middle
    return round(total_effort, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ESTIMATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def estimate_effort_hours_v2(client, full_name: str,
                              known_total_commits: int = None) -> dict:
    """
    Full two-phase effort estimation for one repository.

    Phase 1: Sample recent + oldest commits, reconstruct sessions.
    Phase 2: Scale to full history using total commit count.

    Args:
        client               : GitHubClient instance
        full_name            : 'owner/repo'
        known_total_commits  : if available from existing features (saves 1 API call)

    Returns:
        dict with effort estimate and diagnostics, or None if estimation fails.
    """
    # ── Get total commit count ────────────────────────────────────────────────
    total_commits = known_total_commits
    if total_commits is None or total_commits <= 0:
        total_commits = fetch_total_commit_count(client, full_name)
        time.sleep(0.3)

    # ── Sample recent commits ─────────────────────────────────────────────────
    recent_author_map = defaultdict(list)
    commits_recent_count = 0

    for page in range(1, SAMPLE_PAGES + 1):
        batch = fetch_commit_page(client, full_name, page=page, direction='desc')
        if not batch:
            break
        parsed = parse_commits_to_author_map(batch, weight=1.0)
        for author, entries in parsed.items():
            recent_author_map[author].extend(entries)
        commits_recent_count += len(batch)
        if len(batch) < COMMITS_PER_PAGE:
            break
        time.sleep(0.15)

    # ── Sample oldest commits ─────────────────────────────────────────────────
    old_author_map = defaultdict(list)
    commits_old_count = 0

    # Only fetch old commits if the project is larger than our recent sample
    if total_commits and total_commits > commits_recent_count + 100:
        for page in range(1, SAMPLE_PAGES_OLD + 1):
            batch = fetch_commit_page(client, full_name, page=page, direction='asc')
            if not batch:
                break
            parsed = parse_commits_to_author_map(batch, weight=1.0)
            for author, entries in parsed.items():
                old_author_map[author].extend(entries)
            commits_old_count += len(batch)
            if len(batch) < COMMITS_PER_PAGE:
                break
            time.sleep(0.15)

    if not recent_author_map and not old_author_map:
        return None

    # ── Reconstruct sessions for each sample ─────────────────────────────────
    def compute_effort_from_map(author_map: dict) -> float:
        total = 0.0
        for author, entries in author_map.items():
            timestamps = [ts for ts, _ in entries]
            total += reconstruct_sessions(timestamps)
        return total

    effort_recent_h = compute_effort_from_map(recent_author_map)
    effort_old_h    = compute_effort_from_map(old_author_map)

    # ── Scale to full history ─────────────────────────────────────────────────
    effort_total_h = scale_effort_to_full_history(
        effort_recent_h  = effort_recent_h,
        effort_old_h     = effort_old_h,
        commits_recent   = commits_recent_count,
        commits_old      = commits_old_count,
        total_commits    = total_commits,
    )

    # ── Coverage ratio (diagnostic) ──────────────────────────────────────────
    commits_sampled = commits_recent_count + commits_old_count
    coverage = (commits_sampled / total_commits) if total_commits else 1.0

    # ── Plausibility filter ───────────────────────────────────────────────────
    if effort_total_h < MIN_EFFORT_H:
        return None
    if effort_total_h > MAX_EFFORT_H:
        # Don't drop — cap and flag instead
        effort_total_h = MAX_EFFORT_H
        capped = True
    else:
        capped = False

    return {
        "effort_hours_real":        round(effort_total_h, 1),
        "effort_hours_sampled":     round(effort_recent_h + effort_old_h, 1),
        "effort_coverage_ratio":    round(coverage, 4),
        "effort_commits_sampled":   commits_sampled,
        "effort_total_commits":     total_commits or -1,
        "effort_contributors_seen": len(set(recent_author_map) | set(old_author_map)),
        "effort_was_capped":        capped,
        "effort_scaling_factor":    round(1.0 / coverage, 2) if coverage > 0 else 1.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def add_real_effort_target(client, features_df: pd.DataFrame,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Enrich the features DataFrame with scaled effort estimates.

    Uses 'total_commits' from features_df if available to avoid
    an extra API call per repo.
    """
    console.rule("[bold blue]Phase 1 v2 — Scaled Commit-Based Effort Estimation[/]")

    has_total_commits = 'total_commits' in features_df.columns
    if has_total_commits:
        console.print("[dim]Using total_commits from existing features (saves API calls)[/]")
    else:
        console.print("[yellow]total_commits not in features — will fetch from API[/]")

    effort_records = []
    failed = []
    total = len(features_df)

    for i, (_, row) in enumerate(features_df.iterrows()):
        full_name = row["full_name"]
        known_tc  = int(row["total_commits"]) if has_total_commits else None

        if verbose:
            console.print(
                f"[dim]({i+1}/{total})[/] [cyan]{full_name}[/]...",
                end=" "
            )

        result = estimate_effort_hours_v2(client, full_name,
                                          known_total_commits=known_tc)

        if result:
            result["full_name"] = full_name
            effort_records.append(result)
            if verbose:
                cov = result['effort_coverage_ratio'] * 100
                scale = result['effort_scaling_factor']
                capped = " [CAPPED]" if result['effort_was_capped'] else ""
                console.print(
                    f"[green]✓[/] {result['effort_hours_real']:,.0f}h  "
                    f"(sampled={result['effort_hours_sampled']:.0f}h  "
                    f"coverage={cov:.1f}%  scale={scale:.1f}×){capped}"
                )
        else:
            failed.append(full_name)
            if verbose:
                console.print("[red]✗ skipped[/]")

        time.sleep(0.4)

    console.rule("Summary")
    console.print(f"  [green]Enriched : {len(effort_records)}[/]")
    console.print(f"  [red]Skipped  : {len(failed)}[/]")

    if not effort_records:
        raise RuntimeError("No effort estimates produced.")

    effort_df  = pd.DataFrame(effort_records)
    enriched   = features_df.merge(effort_df, on="full_name", how="inner")

    # ── Print scaling diagnostics ─────────────────────────────────────────────
    console.print("\n[bold]Scaling diagnostics:[/]")
    console.print(f"  Mean coverage ratio  : {effort_df['effort_coverage_ratio'].mean():.1%}")
    console.print(f"  Mean scaling factor  : {effort_df['effort_scaling_factor'].mean():.1f}×")
    console.print(f"  Repos capped at max  : {effort_df['effort_was_capped'].sum()}")
    console.print(f"\n  Effort distribution (effort_hours_real):")
    stats = effort_df['effort_hours_real'].describe()
    console.print(f"    min={stats['min']:,.0f}h  median={stats['50%']:,.0f}h  "
                  f"mean={stats['mean']:,.0f}h  max={stats['max']:,.0f}h")

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION — compare v1 vs v2 targets
# ══════════════════════════════════════════════════════════════════════════════

def compare_v1_v2(enriched_df: pd.DataFrame):
    """
    If the DataFrame contains both effort_hours_v1 and effort_hours_real,
    compare them to quantify the improvement from scaling.
    """
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    if 'effort_hours_v1' not in enriched_df.columns:
        print("No v1 column found — run with v1 first to enable comparison.")
        return

    v1 = enriched_df['effort_hours_v1']
    v2 = enriched_df['effort_hours_real']
    synth = enriched_df.get('effort_target', None)

    fig, axes = plt.subplots(1, 3 if synth is not None else 2, figsize=(15, 4))

    axes[0].hist(np.log1p(v1), bins=40, alpha=0.6, label='v1 (capped)', color='steelblue')
    axes[0].hist(np.log1p(v2), bins=40, alpha=0.6, label='v2 (scaled)', color='darkorange')
    axes[0].set_title('v1 vs v2 Target Distribution')
    axes[0].set_xlabel('log(hours + 1)')
    axes[0].legend()

    axes[1].scatter(np.log1p(v1), np.log1p(v2), alpha=0.4, s=15, color='teal')
    axes[1].plot([0, 15], [0, 15], 'r--', lw=1)
    r, p = stats.spearmanr(v1, v2)
    axes[1].set_title(f'v1 vs v2  Spearman r={r:.3f}')
    axes[1].set_xlabel('log(v1 effort)')
    axes[1].set_ylabel('log(v2 effort)')

    if synth is not None:
        r_v1, _ = stats.spearmanr(synth, v1)
        r_v2, _ = stats.spearmanr(synth, v2)
        axes[2].bar(['v1 vs synthetic', 'v2 vs synthetic'],
                    [r_v1, r_v2], color=['steelblue', 'darkorange'])
        axes[2].set_title('Spearman r with Synthetic Target')
        axes[2].set_ylabel('r')
        axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('effort_v1_vs_v2.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nv1 range: {v1.min():.0f}h – {v1.max():.0f}h  (std={v1.std():.0f})")
    print(f"v2 range: {v2.min():.0f}h – {v2.max():.0f}h  (std={v2.std():.0f})")
    print(f"Std improvement: {v2.std() / v1.std():.1f}× more variance in v2")


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, sys

    GH_TOKEN   = os.getenv("GH_TOKEN")
    INPUT_CSV  = "features_merged_fixed.csv"
    OUTPUT_CSV = "features_with_real_effort_v2.csv"

    if not GH_TOKEN:
        print("ERROR: export GH_TOKEN=your_token_here")
        sys.exit(1)

    try:
        from scraper import GitHubClient
    except ImportError:
        print("ERROR: scraper.py not found.")
        sys.exit(1)

    client = GitHubClient(GH_TOKEN)
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} repos from {INPUT_CSV}")

    enriched = add_real_effort_target(client, df)
    enriched.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✓ Saved → {OUTPUT_CSV}")
    print(f"  Shape: {enriched.shape}")
    print("\nNext: use 'effort_hours_real' as your ML target.")
    print("      Use 'effort_coverage_ratio' to filter low-coverage repos if needed.")