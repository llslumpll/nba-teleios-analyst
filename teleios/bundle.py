"""
Packs everything the analyst chat needs into one file, chat/data.json:
ratings, injuries, this week's predictions with their reasons, the
scorecard, the patterns notebook, experiments, shot profiles, and this
season's results and player game logs.

  python -m teleios.bundle
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import polars as pl

from teleios import data

OUT = data.ROOT / "chat" / "data.json"


def _j(name: str, default=None):
    try:
        return json.loads((data.DATA / name).read_text())
    except Exception:  # noqa: BLE001
        return default


DROP = {"PLAYER_ID", "TEAM_ID", "TEAM_NAME", "PLAYER_NAME", "TEAM_ABBREVIATION", "PLAYER_LAST_TEAM_ID",
        "PLAYER_LAST_TEAM_ABBREVIATION", "TEAM_CITY", "TEAM_NICKNAME", "SEASON_YEAR", "W", "L", "PLAYER_POSITION",
        "AGE", "GP_RANK", "G", "FG2A_FREQUENCY", "FG3A_FREQUENCY", "FGA_FREQUENCY"}


def tracking() -> dict:
    """NBA.com camera-tracking stats saved by the Mac tracker:
    {"status", "players": {name: {table: {stat: value}}}, "teams": {code: {...}}}"""
    import csv
    from teleios import teams as T
    folder = data.DATA / "tracking" / "latest"
    out = {"status": _j("tracking/status.json", {}), "players": {}, "teams": {}}
    if not folder.exists():
        return out
    for f in sorted(folder.glob("*.csv")):
        kind, table = f.stem.split("_", 1)
        with open(f) as fh:
            for row in csv.DictReader(fh):
                vals = {}
                for k, v in row.items():
                    if k in DROP or v in ("", None):
                        continue
                    try:
                        vals[k.lower()] = round(float(v), 3)
                    except ValueError:
                        continue
                if kind == "player":
                    key = row.get("PLAYER_NAME") or row.get("PLAYER_NAME".lower())
                    rec = out["players"].setdefault(key, {"team": T.canon(row.get("TEAM_ABBREVIATION") or row.get("PLAYER_LAST_TEAM_ABBREVIATION"), "kalshi")})
                else:
                    key = T.canon(row.get("TEAM_ABBREVIATION"), "kalshi") or T.canon(row.get("TEAM_NAME"), "espn")
                    if not key:
                        full = {v[0]: c for c, v in T.TEAMS.items()}
                        key = full.get(row.get("TEAM_NAME"))
                    if not key:
                        continue
                    rec = out["teams"].setdefault(key, {})
                rec[table] = vals
    return out


def main():
    today = date.today()
    preds = {}
    for f in sorted((data.DATA / "predictions").glob("*.json")):
        d = f.stem
        if d >= str(today - timedelta(days=14)):
            preds[d] = list(json.loads(f.read_text()).values())

    pproj = {}
    for f in sorted((data.DATA / "player_projections").glob("*.json")) if (data.DATA / "player_projections").exists() else []:
        if f.stem >= str(today - timedelta(days=3)):
            day = json.loads(f.read_text())
            for g in day.values():
                for p in g.get("players", []):
                    p.pop("range50", None)
            pproj[f.stem] = list(day.values())

    games = data.games().filter(pl.col("season") == data.CURRENT_SEASON)
    done = games.filter(pl.col("completed") & (pl.col("home_score") + pl.col("away_score") > 0))
    back = {g["game_id"]: g for g in (_j("backfill_2027.json", {}) or {}).get("games", [])}
    results = []
    for g in done.sort("tip_utc").iter_rows(named=True):
        b = back.get(g["game_id"], {})
        results.append([str(g["date"]), g["home"], g["away"], g["home_score"], g["away_score"],
                        b.get("p_home"), b.get("market_p_home")])
    upcoming = [[str(g["date"]), g["home"], g["away"], str(g["tip_utc"])]
                for g in games.filter(~pl.col("completed") & (pl.col("date") <= today + timedelta(days=7))).sort("tip_utc").iter_rows(named=True)]

    pg = data.player_games().filter(pl.col("date") >= pl.date(2026, 10, 1)).sort("date")
    opp = {}
    for g in done.iter_rows(named=True):
        opp[(g["game_id"], g["home"])] = g["away"]
        opp[(g["game_id"], g["away"])] = g["home"]
    logs = {}
    names = {}
    for r in pg.iter_rows(named=True):
        names[r["pid"]] = r["player"]
        logs.setdefault(r["pid"], []).append([
            str(r["date"]), r["team"], opp.get((r["game_id"], r["team"])), "DNP" if r["dnp"] else round(r["min"]),
            int(r["pts"]), int(r["reb"]), int(r["ast"]), int(r["stl"]), int(r["blk"]), int(r["tov"]),
            int(r["fgm"]), int(r["fga"]), int(r["fg3m"]), int(r["ftm"]), int(r["fta"]), int(r["pm"])])

    ratings = _j("ratings.json", {})
    bundle = {
        "built": data.now_utc(),
        "season": "2026-27",
        "model": {"params": _j("model_params.json", {}), "situation_effects": ratings.get("situation_effects"),
                  "how": ("margin = home court + rest/back-to-back + travel + altitude + team term + "
                          "sum(projected minutes/48 x player rating), home minus away; "
                          "P(home wins) = Phi(margin/sigma). Player ratings: game-level regularised plus-minus "
                          "pulled toward a box-score prior, recency-weighted. Out/Doubtful players removed, "
                          "Questionable keep 65% of minutes.")},
        "teams": ratings.get("teams", {}),
        "players": ratings.get("players", []),
        "injuries": _j("injuries.json", {}),
        "predictions": preds,
        "player_projections": pproj,
        "upcoming": upcoming,
        "results": {"cols": ["date", "home", "away", "home_score", "away_score", "teleios_p_home", "fanduel_close_p_home"],
                    "rows": results},
        "player_logs": {"cols": ["date", "team", "opp", "min", "pts", "reb", "ast", "stl", "blk", "tov",
                                 "fgm", "fga", "fg3m", "ftm", "fta", "plus_minus"],
                        "names": {str(k): v for k, v in names.items()},
                        "logs": {str(k): v for k, v in logs.items()}},
        "shot_profiles": _j("shot_profiles.json", {}),
        "tracking": tracking(),
        "scorecard": _j("scorecard.json", {}),
        "notebook": _j("notebook.json", {}),
        "experiments": _j("experiments.json", {}),
        "last_runs": (_j("run_log.json", []) or [])[-3:],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(bundle, separators=(",", ":"), default=str))
    print(f"chat/data.json: {OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
