"""
Daily player projections: for every game in today's and tomorrow's slate,
project each available player's stat line (with 50% and 80% ranges),
freeze it (each line means "if he plays"; Questionable players are flagged), keep updating until tip-off, lock it, and grade it once the box
score lands. A player who doesn't play is void, not a miss.

  data/player_projections/<date>.json   {game_id: {...game..., players: [...]}}
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl

from teleios import calibrate, data
from teleios.players import COMBOS, STATS, PlayerModel, PParams, chance_over, interval
from teleios.sources import load_kalshi_props, norm_name

DIR = data.DATA / "player_projections"
PARAMS = data.DATA / "player_params.json"
SHOW = ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "pra")


def load_params() -> PParams:
    try:
        return PParams(**json.loads(PARAMS.read_text()))
    except Exception:  # noqa: BLE001
        return PParams()


def _load(day: str) -> dict:
    try:
        return json.loads((DIR / f"{day}.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save(day: str, recs: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / f"{day}.json").write_text(json.dumps(recs, indent=1, sort_keys=True, default=str))


def project(games: pl.DataFrame, pg: pl.DataFrame, slate: list[dict]) -> dict[int, list[dict]]:
    """slate: [{game_id, date, home, away, margin, lineups: {team: {pid: minutes}}, questionable: {team: [pid]}}]
    Returns game_id -> player projections."""
    if not slate:
        return {}
    names = dict(pg.select(["pid", "player"]).unique("pid").iter_rows())
    fake = []
    for g in slate:
        for team, lu in g["lineups"].items():
            for pid in lu:
                fake.append({"game_id": g["game_id"], "date": g["date"], "team": team, "pid": pid,
                             "player": names.get(pid, str(pid)), "min": 1e-6, "dnp": False, "starter": False,
                             **{c: 0.0 for c in ("pts", "reb", "oreb", "ast", "stl", "blk", "tov", "pf", "fgm",
                                                 "fga", "fg3m", "ftm", "fta", "pm")}})
    fdf = pl.DataFrame(fake).select(pg.columns).cast(pg.schema)
    pg2 = pl.concat([pg.filter(~pl.col("game_id").is_in([g["game_id"] for g in slate])), fdf])
    games2 = games.with_columns(pl.when(pl.col("game_id").is_in([g["game_id"] for g in slate]))
                                .then(pl.lit(False)).otherwise(pl.col("completed")).alias("completed"))
    m = PlayerModel(games2, pg2, load_params())
    margins = {str(g["game_id"]): g["margin"] for g in slate}
    pr = m.project_rows(margins=margins)
    want = {g["game_id"] for g in slate}
    qmap = {(g["game_id"], pid) for g in slate for t, l in g["questionable"].items() for pid in l}
    out: dict[int, list[dict]] = {}
    cols = list(STATS)
    for i in np.where(np.isin(m.gid, list(want)))[0]:
        gid, pid = int(m.gid[i]), int(m.pid[i])
        if (gid, pid) not in {(g["game_id"], p) for g in slate for lu in g["lineups"].values() for p in lu}:
            continue
        # A player who sits is void, so each line is "if he plays": a Questionable
        # player keeps his full projection and is flagged.
        q = (gid, pid) in qmap
        f = 1.0
        mu = {s: float(pr["mu"][i, j]) for j, s in enumerate(cols)}
        for name, parts in COMBOS.items():
            mu[name] = sum(mu[x] for x in parts)
        rec = {"pid": pid, "player": names.get(pid), "team": m.team[i], "opp": m.opp[i],
               "minutes": round(float(pr["min"][i]), 1), "questionable": q, "proj": {}, "range50": {}, "range80": {}}
        for s in SHOW:
            var = m.p.var.get(s)
            rec["proj"][s] = round(mu[s], 2)
            if var:
                st = s if s in STATS else "pts"
                rec["range50"][s] = [round(x, 1) for x in interval(st, mu[s], var, 0.5)]
                rec["range80"][s] = [round(x, 1) for x in interval(st, mu[s], var, 0.8)]
        out.setdefault(gid, []).append(rec)
    for gid in out:
        out[gid].sort(key=lambda r: -r["minutes"])
    return out


def season_avgs(pg: pl.DataFrame, before: date) -> dict[int, dict]:
    """Season-to-date per-game averages: the naive baseline every projection is compared with."""
    cur = pg.filter((pl.col("date") >= pl.date(2026, 10, 1)) & (pl.col("date") < before) & ~pl.col("dnp") & (pl.col("min") > 0))
    a = cur.group_by("pid").agg(*[pl.col(s).mean() for s in STATS], g=pl.len())
    out = {}
    for r in a.iter_rows(named=True):
        d = {s: round(r[s], 2) for s in STATS}
        d["pra"] = round(r["pts"] + r["reb"] + r["ast"], 2)
        out[r["pid"]] = d
    return out


MARKET_STATS = ("pts", "reb", "ast", "fg3m", "pra")


def attach_market(players: list[dict], day: str, home: str, away: str, ladders: dict, var: dict) -> None:
    """Kalshi's main line for each player stat, its price for OVER, and
    Teleios's own chance of OVER at that same line (raw, and corrected by
    teleios/calibrate.py once a correction has passed its held-out test)."""
    cal = calibrate.load()
    for p in players:
        mk = {}
        for s in MARKET_STATS:
            lad = ladders.get((norm_name(p["player"]), s, day, home, away))
            if not lad or not lad.get("main"):
                continue
            line = lad["main"]["line"]
            st = s if s in STATS else "pts"
            raw = chance_over(st, p["proj"][s], var[s], line)
            mk[s] = {"line": line, "market_p_over": round(lad["main"]["mid"], 4),
                     "teleios_p_over": round(calibrate.apply(raw, s, cal), 4),   # corrected if a correction has earned its place
                     "teleios_p_over_raw": round(raw, 4)}
        if mk:
            p["market"] = mk


def update(games: pl.DataFrame, pg: pl.DataFrame, slate: list[dict], now) -> int:
    """Write/refresh projections for un-tipped games in the slate."""
    if not slate:
        return 0
    proj = project(games, pg, slate)
    var = load_params().var
    ladders = {(l_["player_key"] if "player_key" in l_ else norm_name(l_["player"]), l_["stat"], l_["date"], l_["home"], l_["away"]): l_
               for l_ in load_kalshi_props().get("ladders", [])}
    n = 0
    by_day: dict[str, list] = {}
    for g in slate:
        by_day.setdefault(str(g["date"]), []).append(g)
    for day, gl in by_day.items():
        recs = _load(day)
        base = season_avgs(pg, date.fromisoformat(day))
        for g in gl:
            k = str(g["game_id"])
            if recs.get(k, {}).get("locked"):
                continue
            players = proj.get(g["game_id"], [])
            for p in players:
                p["season_avg"] = base.get(p["pid"])
            attach_market(players, day, g["home"], g["away"], ladders, var)
            recs[k] = {"game_id": g["game_id"], "date": day, "home": g["home"], "away": g["away"],
                       "tip_utc": g["tip_utc"], "frozen_at": data.now_utc(), "locked": False, "players": players}
            n += len(players)
        _save(day, recs)
    return n


def lock_started(games: pl.DataFrame, now) -> None:
    tips = {str(g["game_id"]): g["tip_utc"] for g in games.select(["game_id", "tip_utc"]).iter_rows(named=True)}
    for f in DIR.glob("*.json") if DIR.exists() else []:
        recs = json.loads(f.read_text())
        ch = False
        for k, r in recs.items():
            t = tips.get(k)
            if not r.get("locked") and t is not None and t <= now:
                r["locked"] = True
                ch = True
        if ch:
            f.write_text(json.dumps(recs, indent=1, sort_keys=True, default=str))


def grade(games: pl.DataFrame, pg: pl.DataFrame) -> int:
    done = set(games.filter(pl.col("completed") & (pl.col("home_score") + pl.col("away_score") > 0))["game_id"].to_list())
    box = {(r["game_id"], r["pid"]): r for r in pg.filter(pl.col("game_id").is_in(list(done))).iter_rows(named=True)}
    n = 0
    for f in sorted(DIR.glob("*.json")) if DIR.exists() else []:
        recs = json.loads(f.read_text())
        ch = False
        for k, r in recs.items():
            if r.get("graded") or int(k) not in done:
                continue
            for p in r["players"]:
                b = box.get((int(k), p["pid"]))
                if not b or b["dnp"] or not b["min"]:
                    p["result"] = "void (did not play)"
                    continue
                act = {s: b[s] for s in STATS}
                act["pra"] = act["pts"] + act["reb"] + act["ast"]
                p["actual"] = {s: act[s] for s in SHOW}
                p["actual_minutes"] = round(b["min"], 1)
                p["in80"] = {s: (p["range80"][s][0] <= act[s] <= p["range80"][s][1]) for s in SHOW if s in p["range80"]}
                for s, mk in (p.get("market") or {}).items():
                    mk["went_over"] = act[s] > mk["line"]
                n += 1
            r["graded"] = True
            r["locked"] = True
            ch = True
        if ch:
            f.write_text(json.dumps(recs, indent=1, sort_keys=True, default=str))
    return n


def scorecard() -> dict:
    """Live average miss per stat (players projected 20+ min), next to the
    season-average baseline on the same players, and range coverage."""
    rows = []
    for f in sorted(DIR.glob("*.json")) if DIR.exists() else []:
        for r in json.loads(f.read_text()).values():
            for p in r.get("players", []):
                if "actual" in p and p["minutes"] >= 20:
                    rows.append(p)
    if not rows:
        return {"player_games": 0}
    out = {"player_games": len(rows), "stats": {}, "vs_kalshi": vs_market()}
    for s in SHOW:
        t = [abs(p["proj"][s] - p["actual"][s]) for p in rows]
        b = [abs(p["season_avg"][s] - p["actual"][s]) for p in rows if p.get("season_avg")]
        bt = [abs(p["proj"][s] - p["actual"][s]) for p in rows if p.get("season_avg")]
        cov = [p["in80"][s] for p in rows if s in p.get("in80", {})]
        out["stats"][s] = {"teleios_avg_miss": round(float(np.mean(t)), 3),
                           "season_avg_miss_same_players": round(float(np.mean(b)), 3) if b else None,
                           "teleios_on_those_players": round(float(np.mean(bt)), 3) if bt else None,
                           "in_80pct_range": round(float(np.mean(cov)), 3) if cov else None}
    return out


def vs_market() -> dict:
    """On every graded player stat with a Kalshi main line: whose chance of
    OVER was more accurate (Brier), and when Teleios and the market disagreed
    on the side, who was right."""
    import math
    rows = []
    for f in sorted(DIR.glob("*.json")) if DIR.exists() else []:
        for r in json.loads(f.read_text()).values():
            for p in r.get("players", []):
                for s, mk in (p.get("market") or {}).items():
                    if "went_over" in mk:
                        rows.append((s, mk))
    out = {"note": "Kalshi main line (tight quote nearest 50%); OVER chance at that line. Brier: lower is better."}
    for s in ("all",) + MARKET_STATS:
        rr = [mk for st, mk in rows if s in ("all", st)]
        if not rr:
            continue
        y = np.array([1.0 if mk["went_over"] else 0.0 for mk in rr])
        t = np.array([mk["teleios_p_over"] for mk in rr])
        m = np.array([mk["market_p_over"] for mk in rr])
        diff = (t - y) ** 2 - (m - y) ** 2
        se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        z = float(diff.mean() / se) if se and se == se and se > 0 else 0.0
        dis = [(tt, mm, yy) for tt, mm, yy in zip(t, m, y) if (tt >= 0.5) != (mm >= 0.5)]
        out[s] = {"lines": len(rr), "teleios_brier": round(float(((t - y) ** 2).mean()), 4),
                  "market_brier": round(float(((m - y) ** 2).mean()), 4), "z": round(z, 2),
                  "verdict": "Teleios ahead" if z < -2 else "market ahead" if z > 2 else "too close to call yet",
                  "over_rate": round(float(y.mean()), 3),
                  "disagreements": len(dis), "teleios_right_when_disagreeing": sum(1 for tt, mm, yy in dis if (tt >= 0.5) == (yy == 1.0))}
    return out
