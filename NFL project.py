from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look nicer by default
sns.set_theme(style="whitegrid", context="talk")

import nflreadpy as nfl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


def last_n_seasons(n: int = 10) -> list[int]:
    """Return a list of the last n season years, ending at the current year."""
    current_year = datetime.now().year
    start = current_year - n + 1
    return list(range(start, current_year + 1))


def _first_existing_column(df, candidates: Sequence[str]) -> str | None:
    """Return the first column name from candidates that exists on df, else None."""
    cols = set(getattr(df, "columns", []))
    for c in candidates:
        if c in cols:
            return c
    return None


def filter_rows_for_teams(df, teams: Iterable[str]):
    """Filter a (Polars) DataFrame to rows matching any of the given teams."""
    import polars as pl

    teams = [t.upper().strip() for t in teams]

    # Common single-team columns
    team_col = _first_existing_column(
        df,
        [
            "team",
            "team_abbr",
            "abbr",
            "posteam",
            "defteam",
            "home_team",
            "away_team",
        ],
    )

    # If schedules-like (home/away), filter on either
    home_col = _first_existing_column(df, ["home_team", "home_team_abbr"])
    away_col = _first_existing_column(df, ["away_team", "away_team_abbr"])

    if home_col and away_col:
        return df.filter(pl.col(home_col).is_in(teams) | pl.col(away_col).is_in(teams))

    if team_col:
        return df.filter(pl.col(team_col).is_in(teams))

    return df


def filter_head_to_head_games(schedules_df, team_a: str, team_b: str):
    """Filter schedules to only games where team_a played team_b."""
    import polars as pl

    team_a = team_a.upper().strip()
    team_b = team_b.upper().strip()

    home_col = _first_existing_column(schedules_df, ["home_team", "home_team_abbr"])
    away_col = _first_existing_column(schedules_df, ["away_team", "away_team_abbr"])

    if not (home_col and away_col):
        return schedules_df

    return schedules_df.filter(
        ((pl.col(home_col) == team_a) & (pl.col(away_col) == team_b))
        | ((pl.col(home_col) == team_b) & (pl.col(away_col) == team_a))
    )


def dedupe_and_sort_head_to_head(head_to_head_df):
    """De-duplicate head-to-head games and sort to keep y and X aligned."""
    import polars as pl

    df = head_to_head_df

    # Try a few common game-id columns to dedupe
    gid = _first_existing_column(df, ["game_id", "gsis", "old_game_id"])
    season_col = _first_existing_column(df, ["season", "year"])
    week_col = _first_existing_column(df, ["week"])

    if gid:
        df = df.unique(subset=[gid], keep="first")

    # Sort for stable alignment
    sort_cols = []
    if season_col:
        sort_cols.append(season_col)
    if week_col:
        sort_cols.append(week_col)
    if gid:
        sort_cols.append(gid)

    if sort_cols:
        df = df.sort(sort_cols)

    return df


def scale_numeric_features(df):
    import polars as pl

    numeric_df = df.select(pl.selectors.numeric())
    if numeric_df.width == 0:
        raise ValueError("No numeric columns found to scale.")

    X = numeric_df.to_numpy()

    # Impute missing values before scaling/modeling
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, numeric_df.columns


# -----------------------------
# Labels (y)
# -----------------------------
def build_binary_outcome_from_schedules(head_to_head_df, team1: str):
    import polars as pl

    home_score = _first_existing_column(head_to_head_df, ["home_score", "home_points", "score_home"])
    away_score = _first_existing_column(head_to_head_df, ["away_score", "away_points", "score_away"])
    home_team = _first_existing_column(head_to_head_df, ["home_team", "home_team_abbr"])
    away_team = _first_existing_column(head_to_head_df, ["away_team", "away_team_abbr"])

    if not all([home_score, away_score, home_team, away_team]):
        raise ValueError("Required schedule columns not found to build target y.")

    team1 = team1.upper().strip()

    df = head_to_head_df.with_columns(
        pl.when(
            ((pl.col(home_team) == team1) & (pl.col(home_score) > pl.col(away_score)))
            | ((pl.col(away_team) == team1) & (pl.col(away_score) > pl.col(home_score)))
        )
        .then(1)
        .otherwise(0)
        .alias("y")
    )

    return df.select("y").to_numpy().ravel()


# -----------------------------
# Feature Engineering (X)
# -----------------------------
def build_matchup_features_with_team_stats(head_to_head_df, team_stats_df, team1: str, team2: str):
   ###   - team_stats is forced to ONE row per (team, season) to prevent join row explosion. ###
    import polars as pl

    team1 = team1.upper().strip()
    team2 = team2.upper().strip()

    df = head_to_head_df

    # Columns from schedules
    season_col = _first_existing_column(df, ["season", "year"])
    home_team_col = _first_existing_column(df, ["home_team", "home_team_abbr"])
    away_team_col = _first_existing_column(df, ["away_team", "away_team_abbr"])
    if not all([season_col, home_team_col, away_team_col]):
        raise ValueError("Schedules data missing season/home/away columns needed for feature building.")

    # Add home indicator for team1 (do this BEFORE dropping any team columns)
    df = df.with_columns((pl.col(home_team_col) == team1).cast(pl.Int8).alias("team1_is_home"))

    # Normalize season column name for joining
    if season_col != "season":
        df = df.rename({season_col: "season"})

    # Add fixed team1/team2 columns for joining
    df = df.with_columns([pl.lit(team1).alias("team1"), pl.lit(team2).alias("team2")])

    # ---- Prepare team stats keys ----
    ts = team_stats_df
    ts_team_col = _first_existing_column(ts, ["team", "team_abbr", "abbr"])
    ts_season_col = _first_existing_column(ts, ["season", "year"])
    if not all([ts_team_col, ts_season_col]):
        raise ValueError("Team stats data missing team/season columns needed for joining.")

    # Select keys + numeric stats (exclude key columns from numeric selector to avoid duplicates)
    ts = ts.select(
        [
            pl.col(ts_team_col),
            pl.col(ts_season_col),
            pl.selectors.numeric().exclude([ts_team_col, ts_season_col]),
        ]
    ).rename({ts_team_col: "team", ts_season_col: "season"})

    # Restrict to just the two teams to reduce size
    ts = ts.filter(pl.col("team").is_in([team1, team2]))

    # 🔒 Force ONE row per (team, season) to prevent join row explosion
    ts = ts.group_by(["team", "season"]).agg(pl.all().exclude(["team", "season"]).mean())
    ts = ts.unique(subset=["team", "season"], keep="first")

    # ---- Join team stats for team1 and team2 (season-level) ----
    t1 = ts.rename({c: f"t1_{c}" for c in ts.columns if c not in {"team", "season"}})
    t2 = ts.rename({c: f"t2_{c}" for c in ts.columns if c not in {"team", "season"}})

    df = df.join(t1, left_on=["team1", "season"], right_on=["team", "season"], how="left")
    df = df.join(t2, left_on=["team2", "season"], right_on=["team", "season"], how="left")

    # ---- Build delta features ----
    delta_exprs = []
    for c in df.columns:
        if c.startswith("t1_"):
            base = c[3:]
            other = f"t2_{base}"
            if other in df.columns:
                delta_exprs.append((pl.col(c) - pl.col(other)).alias(f"delta_{base}"))
    df = df.with_columns(delta_exprs)

    # ---- Drop leakage / IDs / labels ----
    drop_cols = set()
    for c in df.columns:
        cl = c.lower()

        # Drop raw joined stats after delta creation
        if c.startswith("t1_") or c.startswith("t2_"):
            drop_cols.add(c)

        # Drop team identifiers / join helpers
        if cl in {"team", "season", "team1", "team2", "home_team", "away_team", "home_team_abbr", "away_team_abbr"}:
            drop_cols.add(c)

        # Drop obvious labels/outcomes
        if "score" in cl or "points" in cl or cl == "result":
            drop_cols.add(c)

        # Drop IDs / dates
        if cl.endswith("_id") or cl in {"gsis", "game_id", "old_game_id", "id", "index"}:
            drop_cols.add(c)
        if "timestamp" in cl or cl in {"gameday", "game_date", "date"}:
            drop_cols.add(c)

        # Drop betting/market keywords if present
        if any(k in cl for k in ["moneyline", "spread", "odds", "total", "line"]):
            drop_cols.add(c)

    df = df.drop([c for c in drop_cols if c in df.columns])

    # Keep only numeric features (context + deltas)
    df = df.select(pl.selectors.numeric())

    # Scale + return
    return scale_numeric_features(df)



# I made some pretty labels for the plots cause otherwise they output ugly names #

def prettify_feature_name(name: str) -> str:
    """Turn model feature keys into readable labels for plots."""
    n = name.strip()

    # Prefix handling
    prefix = ""
    if n.startswith("delta_"):
        prefix = "Δ "
        n = n[len("delta_"):]

    # Common tokens / abbreviations
    token_map = {
        "epa": "Expected Points Added",
        ##### Measures Offensive Efficiency #####
        "qbr": "Quarterback Rating",
        ##### Quarterback Rating #####
        "cpoe": "Completion Percentage over Expectation",
        ##### Completion Percentage over Expected (higher with better QB) #####
        "ypa": "Yards per Attempt",
        ##### Yards per Passing Attempt #####
        "ypp": "Yards Per Play",
        ##### Yards per Offensive Play #####
        "td": "TD",
        "tds": "TDs",
        ##### Representation of Offensive Historical Dominance #####
        "int": "Interception",
        "ints": "Interceptions",
        ##### Volatility and Lost Posessions #####
        "fg": "Field Goal",
        "fgs": "Field Goals",
        ##### Mixed reading, either drives stall out or late game plays #####
        "xp": "Extra Point",
        "xps": "Extra Points",
        ##### Basically just TDs, might get compiled with completion #####
        "yd": "Yd",
        "yds": "Yds",
        "yards": "Yards",
        ##### Measures offensive pressure without situational regard #####
        "pct": "%",
        ##### normalization for other stats (3rd down conversions --> 3rd down conversion pct)
        "wp": "Win Probability",
        ##### Essentially encoded win percentage/likelihood #####
        "wpa": "Win probability added",
        ##### Win Probability added by Plays (does this team make plays when it mattered most?) #####
        "sr": "Success Rate",
        ##### Consistency Metric #####
        "tfl": "Tackles for Loss",
        ##### Tackles for loss, defensive pressure effectiveness #####
        "qb": "QB",
        "wr": "WR",
        "rb": "RB",
        "te": "TE",
        "st": "ST",
        "def": "Defense",
        "off": "Offense",
        ##### Positions #####
        "team1": "Team 1",
        "team2": "Team 2",
    }

    # Special-case a few known engineered fields
    if name == "team1_is_home":
        return "Team 1 is Home (1/0)"

    parts = [p for p in n.replace("__", "_").split("_") if p]
    pretty_parts: list[str] = []
    for p in parts:
        low = p.lower()
        if low in token_map:
            pretty_parts.append(token_map[low])
        elif low.isdigit():
            pretty_parts.append(low)
        else:
            # Title-case normal words, keep common stat suffixes readable
            pretty_parts.append(low.capitalize())

    out = " ".join(pretty_parts).replace(" %", "%")
    return (prefix + out).strip()


# -----------------------------
# Visualization + Reporting
# -----------------------------
def print_top_features(model, feature_names, top_n: int = 10):
    """Print top-N features by absolute coefficient magnitude."""
    coefs = model.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1][:top_n]

    print(f"\nTop {top_n} features influencing Team 1 win probability:")
    print("--------------------------------------------------")
    for i, idx in enumerate(order, start=1):
        direction = "favors Team 1" if coefs[idx] > 0 else "favors Team 2"
        label = prettify_feature_name(str(feature_names[idx]))
        print(f"{i:>2}. {label:<35} {coefs[idx]:>7.2f}   ({direction})")


def plot_logistic_coeff_heatmap(model, feature_names, title: str, top_n: int = 35):
    """Plot a cleaner coefficient heatmap (sorted by absolute magnitude)."""
    coefs = model.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1]

    # Show only the most influential features to keep labels readable
    order = order[: min(top_n, len(order))]

    sorted_features = np.array(feature_names, dtype=object)[order]
    sorted_coefs = coefs[order]

    pretty_labels = [prettify_feature_name(str(n)) for n in sorted_features]

    # Dynamic height so labels don't overlap
    height = max(6, 0.35 * len(pretty_labels) + 2)
    plt.figure(figsize=(11, height))

    ax = sns.heatmap(
        sorted_coefs.reshape(-1, 1),
        yticklabels=pretty_labels,
        xticklabels=["Impact"],
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=11)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    seasons = last_n_seasons(10)

    print("Enter TWO NFL teams using abbreviations (examples: KC, SF, PIT, DAL, PHI, BUF, LAR, SEA)")
    team1 = input("Team 1: ").strip().upper()
    team2 = input("Team 2: ").strip().upper()

    if not team1 or not team2:
        raise SystemExit("You must enter two team abbreviations.")

    print(f"\nLoading data for seasons: {seasons[0]}–{seasons[-1]}\n")

    # 1) Schedules / results
    schedules = nfl.load_schedules(seasons)
    schedules_for_teams = filter_rows_for_teams(schedules, [team1, team2])
    head_to_head = filter_head_to_head_games(schedules, team1, team2)
    head_to_head = dedupe_and_sort_head_to_head(head_to_head)

    print("Schedules loaded.")
    print(f"- Games involving either team: {schedules_for_teams.height}")
    print(f"- Head-to-head games: {head_to_head.height}\n")

    # 2) Team stats
    team_stats = nfl.load_team_stats(seasons)
    team_stats_for_teams = filter_rows_for_teams(team_stats, [team1, team2])

    print("Team stats loaded.")
    print(f"- Rows for selected teams: {team_stats_for_teams.height}\n")

    # Scale numeric team stats (not required for matchup model, but kept for info)
    try:
        X_team_scaled, team_feature_names = scale_numeric_features(team_stats_for_teams)
        print("Team stats scaled.")
        print(f"- Scaled feature count: {len(team_feature_names)}")
        print(f"- Scaled matrix shape: {X_team_scaled.shape}\n")
    except ValueError as e:
        print(f"Team stats scaling skipped: {e}\n")

    # 3) Logistic regression: head-to-head winner
    try:
        if head_to_head.height == 0:
            raise ValueError("No head-to-head games available for modeling.")

        y = build_binary_outcome_from_schedules(head_to_head, team1)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError(
                f"Need at least 2 outcome classes to train. Got classes={classes} with counts={counts}. "
                "Try swapping team1/team2, expanding the date range, or using a different matchup."
            )

        X, x_feature_names = build_matchup_features_with_team_stats(head_to_head, team_stats, team1, team2)

        # Safety check: match sample counts
        if X.shape[0] != len(y):
            raise ValueError(f"X and y length mismatch after feature build: X={X.shape[0]}, y={len(y)}")

        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)

        delta_count = sum(1 for n in x_feature_names if n.startswith("delta_"))

        print("Logistic regression model trained.")
        print(f"- Training samples: {len(y)}")
        print(f"- Class balance: {dict(zip(classes.tolist(), counts.tolist()))}")
        print(f"- Number of features: {X.shape[1]}")
        print(f"- Delta (team1-team2) stat features: {delta_count}\n")

        print_top_features(model, x_feature_names, top_n=10)
        plot_logistic_coeff_heatmap(
            model,
            x_feature_names,
            title=f"Most Influential Football Features: {team1} vs {team2}",
            top_n=35,
        )

    except Exception as e:
        print(f"Logistic regression skipped: {e}\n")

    # 4) Player stats (optional, informational)
    player_stats = nfl.load_player_stats(seasons)
    player_stats_for_teams = filter_rows_for_teams(player_stats, [team1, team2])

    print("Player stats loaded.")
    print(f"- Total player-stat rows for selected teams: {player_stats_for_teams.height}\n")

    try:
        X_player_scaled, player_feature_names = scale_numeric_features(player_stats_for_teams)
        print("Player stats scaled.")
        print(f"- Scaled feature count: {len(player_feature_names)}")
        print(f"- Scaled matrix shape: {X_player_scaled.shape}\n")
    except ValueError as e:
        print(f"Player stats scaling skipped: {e}\n")

    print("Head-to-head sample:")
    print(head_to_head.head(5))


if __name__ == "__main__":
    main()