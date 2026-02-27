#!/usr/bin/env python3
"""
NFL Combine Career Predictor
Compares a player's combine results to historical players at the same position
and uses Claude AI (with adaptive thinking) to predict career trajectory.

Usage:
    python nfl_combine_analyzer.py "Caleb Williams"
    python nfl_combine_analyzer.py "Bo Nix" --comps 15
    python nfl_combine_analyzer.py "Marvin Harrison Jr" --years 2010 2024
"""

import ssl
import sys
import argparse
import anthropic
import numpy as np
import pandas as pd

# macOS Python 3.13 SSL cert fix
try:
    import certifi
    ssl._create_default_https_context = ssl.create_default_context
except ImportError:
    pass

try:
    import nfl_data_py as nfl
except ImportError:
    print("ERROR: nfl_data_py not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Combine metric labels for display
# nflverse column names: forty, bench, vertical, broad_jump, cone, shuttle
# ---------------------------------------------------------------------------
METRIC_LABELS = {
    "ht":          "Height",
    "wt":          "Weight (lbs)",
    "forty":       "40-Yard Dash (s)",
    "vertical":    "Vertical Jump (in)",
    "broad_jump":  "Broad Jump (in)",
    "bench":       "Bench Press (reps)",
    "shuttle":     "20-Yard Shuttle (s)",
    "cone":        "3-Cone Drill (s)",
}

# Metrics that matter most by position group
POSITION_METRICS: dict[str, list[str]] = {
    "QB":  ["ht", "wt", "forty", "vertical", "broad_jump", "shuttle", "cone"],
    "RB":  ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "FB":  ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "WR":  ["ht", "wt", "forty", "vertical", "broad_jump", "shuttle", "cone"],
    "TE":  ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "OT":  ["ht", "wt", "bench", "vertical", "broad_jump", "shuttle", "cone"],
    "OG":  ["ht", "wt", "bench", "vertical", "broad_jump", "shuttle", "cone"],
    "C":   ["ht", "wt", "bench", "vertical", "broad_jump", "shuttle", "cone"],
    "DE":  ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "DT":  ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "NT":  ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "OLB": ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "ILB": ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "MLB": ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "LB":  ["ht", "wt", "forty", "vertical", "broad_jump", "bench", "shuttle", "cone"],
    "CB":  ["ht", "wt", "forty", "vertical", "broad_jump", "shuttle", "cone"],
    "S":   ["ht", "wt", "forty", "vertical", "broad_jump", "shuttle", "cone"],
    "SS":  ["ht", "wt", "forty", "vertical", "broad_jump", "shuttle", "cone"],
    "FS":  ["ht", "wt", "forty", "vertical", "broad_jump", "shuttle", "cone"],
    "K":   ["ht", "wt"],
    "P":   ["ht", "wt"],
    "LS":  ["ht", "wt"],
}

DEFAULT_METRICS = ["ht", "wt", "forty", "vertical", "broad_jump", "shuttle", "cone"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def height_to_inches(h) -> float | None:
    """Convert height strings like '6-3' or '6-03' to total inches."""
    if pd.isna(h):
        return None
    s = str(h).strip()
    if "-" in s:
        parts = s.split("-")
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except (ValueError, IndexError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def load_combine_data(start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    print(f"  Loading combine data for {start_year}–{end_year}...")
    df = nfl.import_combine_data(years)

    # Normalise column names to lower-case
    df.columns = [c.lower() for c in df.columns]

    # Rename common variants
    renames = {"player": "player_name", "year": "season", "position": "pos"}
    df.rename(columns={k: v for k, v in renames.items() if k in df.columns}, inplace=True)

    # Ensure season is numeric
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")

    # Convert height from "6-3" format to total inches (numeric)
    if "ht" in df.columns:
        df["ht_display"] = df["ht"]          # keep original for display
        df["ht"] = df["ht"].apply(height_to_inches)

    # Ensure all metric columns are numeric
    for col in ["wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Player lookup
# ---------------------------------------------------------------------------

def find_player(df: pd.DataFrame, name: str) -> pd.Series | None:
    mask = df["player_name"].str.contains(name, case=False, na=False)
    results = df[mask].copy()

    if results.empty:
        return None

    if len(results) > 1:
        print(f"\n  Multiple matches for '{name}':")
        for _, row in results.iterrows():
            season = int(row["season"]) if pd.notna(row.get("season")) else "?"
            print(f"    - {row['player_name']} ({row.get('pos', '?')}, {season})")
        print("  Using the most recent entry.\n")
        results = results.sort_values("season", ascending=False)

    return results.iloc[0]


# ---------------------------------------------------------------------------
# Position metrics
# ---------------------------------------------------------------------------

def get_metrics_for_position(pos: str) -> list[str]:
    pos_upper = pos.upper().strip()
    if pos_upper in POSITION_METRICS:
        return POSITION_METRICS[pos_upper]
    # Fuzzy match
    for key in POSITION_METRICS:
        if key in pos_upper or pos_upper in key:
            return POSITION_METRICS[key]
    return DEFAULT_METRICS


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def find_similar_players(
    df: pd.DataFrame,
    target: pd.Series,
    n: int = 10,
) -> pd.DataFrame:
    pos = str(target.get("pos", "")).upper().strip()
    metrics = get_metrics_for_position(pos)

    # Same position, excluding target player-season
    pos_df = df[df["pos"].str.upper().str.strip() == pos].copy()
    exclude_mask = (
        (pos_df["player_name"] == target["player_name"]) &
        (pos_df["season"] == target.get("season"))
    )
    pos_df = pos_df[~exclude_mask]

    if pos_df.empty:
        return pd.DataFrame()

    # Find metrics where we have a valid target value
    valid_metrics = []
    target_vals: list[float] = []
    for m in metrics:
        if m not in df.columns:
            continue
        val = target.get(m)
        if pd.notna(val) and float(val) != 0:
            valid_metrics.append(m)
            target_vals.append(float(val))

    if not valid_metrics:
        return pos_df.head(n)

    target_arr = np.array(target_vals, dtype=float)

    # Per-metric standard deviations (for normalisation)
    stds = pos_df[valid_metrics].std(ddof=0).replace(0, 1).values.astype(float)

    distances: list[float] = []
    for _, row in pos_df.iterrows():
        row_vals = [row.get(m) for m in valid_metrics]
        if any(pd.isna(v) or float(v) == 0 for v in row_vals):
            distances.append(float("inf"))
            continue
        row_arr = np.array(row_vals, dtype=float)
        diff = (row_arr - target_arr) / stds
        distances.append(float(np.linalg.norm(diff)))

    pos_df = pos_df.copy()
    pos_df["_distance"] = distances
    return pos_df.sort_values("_distance").head(n).drop(columns=["_distance"])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_player_card(player: pd.Series, metrics: list[str]) -> str:
    lines: list[str] = []
    name = player.get("player_name", "Unknown")
    pos  = player.get("pos", "?")
    season = int(player["season"]) if pd.notna(player.get("season")) else "?"

    dr = player.get("draft_round")
    dp = player.get("draft_ovr")
    if pd.notna(dr):
        draft_info = f"Round {int(dr)}, Overall Pick #{int(dp)}" if pd.notna(dp) else f"Round {int(dr)}"
    else:
        draft_info = "Undrafted / Unknown"

    lines.append(f"Player   : {name}")
    lines.append(f"Position : {pos}")
    lines.append(f"Year     : {season}")
    lines.append(f"Draft    : {draft_info}")
    lines.append("Combine  :")
    for m in metrics:
        label = METRIC_LABELS.get(m, m)
        if m == "ht":
            # Prefer the original feet-inches display string
            val_str = str(player.get("ht_display", "N/A"))
            if val_str == "nan" or not val_str:
                val_str = "N/A"
        else:
            val = player.get(m)
            val_str = f"{val}" if pd.notna(val) and val != 0 else "N/A"
        lines.append(f"  {label:<25} {val_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

def analyze_with_claude(
    target: pd.Series,
    similar_players: pd.DataFrame,
    metrics: list[str],
) -> None:
    client = anthropic.Anthropic()

    target_card = format_player_card(target, metrics)

    comp_cards: list[str] = []
    for _, player in similar_players.iterrows():
        comp_cards.append(format_player_card(player, metrics))

    comps_text = "\n\n---\n\n".join(comp_cards)
    pos = target.get("pos", "?")

    prompt = f"""You are a veteran NFL scout and draft analyst with encyclopedic knowledge of NFL combine history, draft results, and player career trajectories from 2000 to the present.

Your task: evaluate the TARGET PLAYER's NFL Combine profile, compare it to historically similar players at the same position, and produce a detailed career prediction.

═══════════════════════════════════════
TARGET PLAYER
═══════════════════════════════════════
{target_card}

═══════════════════════════════════════
MOST SIMILAR HISTORICAL PLAYERS
(ranked by combine metric proximity for {pos})
═══════════════════════════════════════
{comps_text}

═══════════════════════════════════════
ANALYSIS REQUIRED
═══════════════════════════════════════

Draw on your knowledge of the listed comparable players' actual NFL careers to anchor your predictions.

1. COMBINE PROFILE ASSESSMENT
   - Grade each recorded metric (elite / above-average / average / below-average / concern) relative to historical {pos} prospects
   - Note any standout athletic traits or red flags

2. HISTORICAL COMP ANALYSIS
   - For each comparable player above, briefly describe their actual NFL career outcome
   - Identify patterns: which combine traits led to success vs. bust at this position?

3. CAREER CEILING & FLOOR
   - Best-case scenario (who is the optimistic comp?)
   - Realistic expectation
   - Worst-case scenario (who is the cautionary tale?)

4. POSITION-SPECIFIC PREDICTORS
   - Which metrics are most predictive of {pos} success, and how does this player score on them?

5. FINAL VERDICT
   - Draft-value assessment
   - Career outlook: Bust / Depth Player / Starter / Pro Bowl / Elite
   - Confidence rating (Low / Medium / High) based on combine signal strength
   - One-paragraph summary

Be specific, be honest, and use the historical comps as evidence.
"""

    print("\n" + "=" * 60)
    print("  CLAUDE AI ANALYSIS  (adaptive thinking enabled)")
    print("=" * 60 + "\n")

    thinking_shown = False

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if hasattr(event, "content_block"):
                    if event.content_block.type == "thinking" and not thinking_shown:
                        print("[Claude is reasoning through historical data...]\n")
                        thinking_shown = True
            elif event.type == "content_block_delta":
                if hasattr(event, "delta") and event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)

    print("\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NFL Combine Career Predictor — powered by Claude AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("player", help="Player name (or partial name) to analyse")
    parser.add_argument(
        "--comps", type=int, default=10,
        help="Number of historical comps to find (default: 10)",
    )
    parser.add_argument(
        "--start-year", type=int, default=2000,
        help="First combine year to include (default: 2000)",
    )
    parser.add_argument(
        "--end-year", type=int, default=2024,
        help="Last combine year to include (default: 2024)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  NFL COMBINE CAREER PREDICTOR — powered by Claude AI")
    print("=" * 60)

    # Load data
    df = load_combine_data(args.start_year, args.end_year)
    print(f"  Loaded {len(df):,} combine records.\n")

    # Find target player
    print(f"  Searching for: {args.player}")
    target = find_player(df, args.player)
    if target is None:
        print(f"\nERROR: No combine record found for '{args.player}'.")
        print("  Tip: Try a partial last name, e.g. 'Mahomes' or 'Burrow'")
        sys.exit(1)

    pos     = target.get("pos", "?")
    season  = int(target["season"]) if pd.notna(target.get("season")) else "?"
    metrics = get_metrics_for_position(pos)

    print(f"  Found: {target['player_name']} | {pos} | {season} Combine\n")

    # Show target stats
    print("=" * 60)
    print("  TARGET PLAYER — COMBINE STATS")
    print("=" * 60)
    print(format_player_card(target, metrics))

    # Find comps
    print(f"\n  Finding {args.comps} closest historical {pos} comps...")
    similar = find_similar_players(df, target, n=args.comps)

    if similar.empty:
        print(f"\nWARNING: No comparable {pos} players found. Analysis may be limited.")
    else:
        print(f"  Found {len(similar)} comps.")
        print("\n  TOP COMPS (by combine similarity):")
        for i, (_, row) in enumerate(similar.iterrows(), 1):
            s = int(row["season"]) if pd.notna(row.get("season")) else "?"
            print(f"    {i:2}. {row['player_name']:<28} ({s})")

    # Claude analysis
    analyze_with_claude(target, similar, metrics)

    print("=" * 60)
    print("  Analysis complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
