"""
Shot profiles for the analyst: for every team and every regular player,
where the shots come from, how well they're made, and how much of the
result is shot quality vs shot making.

  shot quality   xPTS per shot: what a league-average shooter would score on
                 these exact shots (zone x shot type)
  shot making    actual points per shot minus xPTS per shot (+ = hitting
                 more than the shots deserve: skill, or a hot streak)

Written to data/shot_profiles.json. Season = this season so far; recent =
the team's last 10 games.
"""

from __future__ import annotations

import json

import polars as pl

from teleios import data, pbp

OUT = data.DATA / "shot_profiles.json"
ZONES = pbp.ZONES


def _profile(df: pl.DataFrame) -> dict:
    n = df.height
    if n == 0:
        return {"fga": 0}
    out = {"fga": n, "pts_per_shot": round(df["pts"].mean(), 3), "xpts_per_shot": round(df["xpts"].mean(), 3)}
    out["shot_making"] = round(out["pts_per_shot"] - out["xpts_per_shot"], 3)
    made = df.filter(pl.col("made"))
    out["assisted_share"] = round(made["assisted"].mean(), 3) if made.height else None
    zones = {}
    for z in ZONES:
        d = df.filter(pl.col("zone") == z)
        zones[z] = {"share": round(d.height / n, 3), "fg_pct": round(d["made"].mean(), 3) if d.height else None,
                    "fga": d.height}
    out["zones"] = zones
    fam = df.group_by("family").agg(pl.len().alias("n"), pl.col("made").mean().alias("pct")).sort("n", descending=True)
    out["shot_types"] = {f: {"share": round(c / n, 3), "fg_pct": round(p, 3)} for f, c, p in fam.iter_rows() if c >= 5}
    return out


def build() -> dict:
    sh = pbp.shots()
    if sh.is_empty():
        return {}
    cur = sh.filter((pl.col("season") == data.CURRENT_SEASON) & (pl.col("season_type") == 2))
    names = dict(data.player_games().select(["pid", "player"]).unique("pid").iter_rows())
    out = {"updated": data.now_utc(), "season": "2026-27", "league": _profile(cur), "teams": {}, "players": []}
    for t in sorted(cur["team"].unique().to_list()):
        off = cur.filter(pl.col("team") == t)
        de = cur.filter(pl.col("opp") == t)
        last10 = off["game_id"].unique().sort()[-10:].to_list()
        out["teams"][t] = {
            "offense_season": _profile(off),
            "offense_last10": _profile(off.filter(pl.col("game_id").is_in(last10))),
            "defense_season": _profile(de),
            "defense_last10": _profile(de.filter(pl.col("game_id").is_in(last10))),
        }
    for pid, d in cur.group_by("pid"):
        if d.height < 60:
            continue
        p = _profile(d)
        p.update(pid=pid[0], player=names.get(pid[0]), team=d.sort("date")["team"][-1], games=d["game_id"].n_unique())
        out["players"].append(p)
    out["players"].sort(key=lambda x: -x["fga"])
    OUT.write_text(json.dumps(out, indent=1))
    return out
