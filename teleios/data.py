"""
Data layer. Downloads the free sportsdataverse ESPN parquet releases (the
same source Prop Edge uses), caches them in data/cache/, and cleans them into
two tidy tables keyed by ESPN game_id and canonical team codes:

  games()         one row per game: date, tip time, home/away, score,
                  neutral site, where it was played
  player_games()  one row per player per game: minutes, box score, DNP

Season labels are the year a season ENDS: 2026-27 is 2027.
A failed download keeps the last cached copy and is logged in
data/source_status.json. Nothing is ever made up.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

from teleios import teams

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
STATUS = DATA / "source_status.json"

CURRENT_SEASON = 2027
SEASONS = (2024, 2025, 2026, 2027)
REGULAR, PLAYOFFS, PLAY_IN = 2, 3, 5

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
URLS = {
    "schedule": BASE + "/espn_nba_schedules/nba_schedule_{s}.parquet",
    "player_box": BASE + "/espn_nba_player_boxscores/player_box_{s}.parquet",
}
FILES = {"schedule": "nba_schedule_{s}.parquet", "player_box": "player_box_{s}.parquet"}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _status() -> dict:
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {}


def download(kind: str, season: int) -> dict:
    """Fetch one file into the cache. Never raises; never replaces a good
    cached file with a broken one."""
    url = URLS[kind].format(s=season)
    dest = CACHE / FILES[kind].format(s=season)
    rec = {"url": url, "checked_at": now_utc()}
    try:
        r = requests.get(url, timeout=90)
        rec["http"] = r.status_code
        rec["last_modified"] = r.headers.get("Last-Modified")
        r.raise_for_status()
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        pl.read_parquet(tmp, n_rows=1)
        tmp.replace(dest)
        rec["state"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["state"] = "failed (kept cached copy)" if dest.exists() else "failed"
        rec["error"] = str(e)[:200]
    st = _status()
    st[f"{kind}_{season}"] = rec
    STATUS.write_text(json.dumps(st, indent=2, sort_keys=True))
    return rec


def refresh(history: bool = False) -> list[dict]:
    """Daily: re-download the current season. history=True also fetches
    older seasons that aren't cached yet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for s in SEASONS:
        for kind in FILES:
            cached = (CACHE / FILES[kind].format(s=s)).exists()
            if s == CURRENT_SEASON or (history and not cached) or not cached:
                out.append(download(kind, s))
    return out


def _read(kind: str) -> pl.DataFrame:
    frames = []
    for s in SEASONS:
        p = CACHE / FILES[kind].format(s=s)
        if p.exists():
            frames.append(pl.read_parquet(p))
    return pl.concat(frames, how="diagonal_relaxed")


def _canon_col(col: str) -> pl.Expr:
    return pl.col(col).map_elements(lambda x: teams.canon(x, "espn"), return_dtype=pl.Utf8)


def games() -> pl.DataFrame:
    s = _read("schedule")
    s = s.filter(pl.col("season_type").is_in([REGULAR, PLAYOFFS, PLAY_IN]))
    s = s.with_columns(
        home=_canon_col("home_abbreviation"),
        away=_canon_col("away_abbreviation"),
    ).filter(pl.col("home").is_not_null() & pl.col("away").is_not_null())  # drops All-Star teams
    s = s.select(
        pl.col("game_id").cast(pl.Int64),
        pl.col("season").cast(pl.Int32),
        pl.col("season_type").cast(pl.Int32),
        pl.col("game_date").alias("date"),
        pl.col("game_date_time").dt.convert_time_zone("UTC").alias("tip_utc"),
        "home", "away",
        pl.col("home_score").cast(pl.Int32),
        pl.col("away_score").cast(pl.Int32),
        pl.col("status_type_completed").fill_null(False).alias("completed"),
        pl.col("status_type_name").alias("status"),
        pl.col("neutral_site").fill_null(False).alias("neutral"),
        pl.col("venue_address_city").alias("city"),
    ).unique("game_id", keep="last").sort(["tip_utc", "game_id"])
    return s


def player_games() -> pl.DataFrame:
    p = _read("player_box")
    p = p.with_columns(team=_canon_col("team_abbreviation"))
    p = p.filter(pl.col("team").is_not_null() & pl.col("season_type").is_in([REGULAR, PLAYOFFS, PLAY_IN]))
    return p.select(
        pl.col("game_id").cast(pl.Int64),
        pl.col("game_date").alias("date"),
        "team",
        pl.col("athlete_id").cast(pl.Int64).alias("pid"),
        pl.col("athlete_display_name").alias("player"),
        pl.col("minutes").cast(pl.Float64).fill_null(0.0).alias("min"),
        pl.col("did_not_play").fill_null(False).alias("dnp"),
        pl.col("starter").fill_null(False),
        pl.col("points").cast(pl.Float64).fill_null(0.0).alias("pts"),
        pl.col("rebounds").cast(pl.Float64).fill_null(0.0).alias("reb"),
        pl.col("offensive_rebounds").cast(pl.Float64).fill_null(0.0).alias("oreb"),
        pl.col("assists").cast(pl.Float64).fill_null(0.0).alias("ast"),
        pl.col("steals").cast(pl.Float64).fill_null(0.0).alias("stl"),
        pl.col("blocks").cast(pl.Float64).fill_null(0.0).alias("blk"),
        pl.col("turnovers").cast(pl.Float64).fill_null(0.0).alias("tov"),
        pl.col("fouls").cast(pl.Float64).fill_null(0.0).alias("pf"),
        pl.col("field_goals_made").cast(pl.Float64).fill_null(0.0).alias("fgm"),
        pl.col("field_goals_attempted").cast(pl.Float64).fill_null(0.0).alias("fga"),
        pl.col("three_point_field_goals_made").cast(pl.Float64).fill_null(0.0).alias("fg3m"),
        pl.col("free_throws_made").cast(pl.Float64).fill_null(0.0).alias("ftm"),
        pl.col("free_throws_attempted").cast(pl.Float64).fill_null(0.0).alias("fta"),
        pl.col("plus_minus").cast(pl.Float64).fill_null(0.0).alias("pm"),
    ).unique(["game_id", "pid"], keep="last")


def venue(row: dict) -> tuple[float, float]:
    """Where a game was played: the home arena, or the neutral city."""
    if row.get("neutral") and row.get("city") in teams.NEUTRAL_CITIES:
        return teams.NEUTRAL_CITIES[row["city"]]
    return teams.location(row["home"])
