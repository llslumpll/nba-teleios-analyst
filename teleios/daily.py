"""
The daily run (GitHub Actions calls this several times a day).

  1. refresh box scores + schedule, injuries, rosters, Kalshi odds
  2. grade every frozen prediction whose game has finished
  3. fit the model on everything finished so far
  4. predict today's and tomorrow's games that haven't tipped
     (a prediction keeps updating until tip-off, then it is locked forever)
  5. write the scorecard, ratings and the patterns notebook

  python -m teleios.daily              full run
  python -m teleios.daily --no-fetch   use cached data only
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from teleios import data, notebook, pbp, player_daily, profiles, scorecard, sources, teams
from teleios.backtest import offseason_before
from teleios.model import Engine, Params, learn_box_beta, phi

ET = ZoneInfo("America/New_York")
PRED_DIR = data.DATA / "predictions"
PARAMS_PATH = data.DATA / "model_params.json"
BETA_PATH = data.DATA / "box_prior.json"
RATINGS_PATH = data.DATA / "ratings.json"
QUESTIONABLE_SHARE = 0.65   # expected share of minutes for a Questionable player
STALE_DAYS = 10             # not seen for this long and not injured: half minutes


def load_params() -> Params:
    try:
        d = json.loads(PARAMS_PATH.read_text())
        return Params(**{k: v for k, v in d.items() if k in Params.__dataclass_fields__})
    except Exception:  # noqa: BLE001
        return Params()


def box_beta(engine: Engine, today: date) -> dict:
    off = offseason_before(today)
    try:
        saved = json.loads(BETA_PATH.read_text())
        if saved.get("learned_at") == str(off) and saved.get("half_life") == engine.p.half_life:
            return saved["beta"]
    except Exception:  # noqa: BLE001
        pass
    beta = learn_box_beta(engine, off)
    BETA_PATH.write_text(json.dumps({"learned_at": str(off), "half_life": engine.p.half_life, "beta": beta}, indent=1))
    return beta


# ---------------------------------------------------------------------------
# Lineups from rosters + injury report
# ---------------------------------------------------------------------------

def build_lineup(engine: Engine, team: str, asof: date, rosters: dict, inj_by_pid: dict, inj_by_name: dict,
                 pg_last: dict) -> tuple[dict, list, list]:
    a = asof.toordinal()
    roster = [p for p in rosters.get("players", []) if p.get("team") == team and p.get("pid")]
    if not roster:  # fall back: players whose latest game was for this team
        roster = [{"pid": pid, "player": engine.names.get(pid), "team": team}
                  for pid, (t, _d) in pg_last.items() if t == team and (a - _d) <= 30]
    mins = engine.projected_minutes([p["pid"] for p in roster], a)
    lineup, out, flagged = {}, [], []
    for p in roster:
        pid = p["pid"]
        if pid not in pg_last:  # never played in our data (e.g. unused rookie)
            continue
        inj = inj_by_pid.get(pid) or inj_by_name.get((p.get("player") or "").lower())
        m = mins[pid]
        if inj and inj.get("class") == "out":
            out.append({"pid": pid, "player": p.get("player"), "status": inj.get("status"), "minutes": round(m, 1)})
            continue
        if inj and inj.get("class") == "questionable":
            flagged.append({"pid": pid, "player": p.get("player"), "status": inj.get("status")})
            m *= QUESTIONABLE_SHARE
        if a - pg_last[pid][1] > STALE_DAYS and not inj:
            m *= 0.5
        lineup[pid] = m
    return engine.scale_to_team(lineup), out, flagged


# ---------------------------------------------------------------------------

def load_day(day: str) -> dict:
    try:
        return json.loads((PRED_DIR / f"{day}.json").read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_day(day: str, recs: dict) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (PRED_DIR / f"{day}.json").write_text(json.dumps(recs, indent=1, sort_keys=True, default=str))


def grade(games: pl.DataFrame) -> int:
    done = {g["game_id"]: g for g in games.filter(pl.col("completed")).iter_rows(named=True)}
    n = 0
    for f in sorted(PRED_DIR.glob("*.json")):
        recs = json.loads(f.read_text())
        changed = False
        for k, r in recs.items():
            if r.get("result") or int(k) not in done:
                continue
            g = done[int(k)]
            if (g["home_score"] or 0) + (g["away_score"] or 0) == 0:
                continue
            m = g["home_score"] - g["away_score"]
            sgn = 1 if r["pick"] == r["home"] else -1
            r["result"] = {"home_score": g["home_score"], "away_score": g["away_score"], "margin": m,
                           "winner": g["home"] if m > 0 else g["away"],
                           "hit": (r["pick"] == (g["home"] if m > 0 else g["away"])),
                           # margin tracking: what Teleios projected vs what happened, from the pick's side
                           "projected_pick_margin": round(sgn * r["margin"], 1),
                           "actual_pick_margin": sgn * m,
                           "margin_error": round(m - r["margin"], 1),   # + = home did better than projected
                           "margin_miss": round(abs(m - r["margin"]), 1),
                           "graded_at": data.now_utc()}
            sp = r.get("spread")
            if sp and sp.get("home_line") is not None:
                cov = m + sp["home_line"]
                home_res = "push" if cov == 0 else ("covered" if cov > 0 else "missed")
                r["result"]["spread"] = {
                    "home_line": sp["home_line"], "home": home_res,
                    "pick": sp.get("pick"),
                    "pick_result": (None if not sp.get("pick") else "push" if cov == 0 else
                                    "win" if (cov > 0) == (sp["pick"] == r["home"]) else "loss")}
            r["locked"] = True
            changed = True
            n += 1
        if changed:
            f.write_text(json.dumps(recs, indent=1, sort_keys=True, default=str))
    return n


def main(argv=None):
    argv = argv or sys.argv[1:]
    fetch = "--no-fetch" not in argv
    log = {"run_at": data.now_utc()}
    if fetch:
        log["box_scores"] = [f'{r["url"].rsplit("/", 1)[-1]}: {r["state"]}' for r in data.refresh()]
        log["play_by_play"] = [f'{r["season"]}: {r["state"]}' for r in pbp.refresh()]
        rosters = sources.fetch_rosters()
        injuries = sources.fetch_injuries()
        kalshi = sources.fetch_kalshi()
        props = sources.fetch_kalshi_props()
        _d0 = datetime.now(timezone.utc).astimezone(ET).date()
        spreads = sources.fetch_spreads([str(_d0), str(_d0 + timedelta(days=1))])
        log["spreads"] = {"error": spreads.get("error"), "games": len(spreads.get("games", [])),
                          "diagnostics": spreads.get("diagnostics")}
        log["kalshi_props"] = {"error": props.get("error"), "ladders": len(props.get("ladders", [])),
                               "with_main_line": sum(1 for l_ in props.get("ladders", []) if l_.get("main")),
                               "diagnostics": props.get("diagnostics")}
    else:
        rosters, injuries, kalshi = sources.load_rosters(), sources.load_injuries(), sources.load_kalshi()
        spreads = sources.load_spreads()
    log["sources"] = {
        "rosters": {"fetched_at": rosters.get("fetched_at"), "error": rosters.get("error"), "players": len(rosters.get("players", []))},
        "injuries": {"fetched_at": injuries.get("fetched_at"), "error": injuries.get("error"), "players": len(injuries.get("players", []))},
        "kalshi": {"fetched_at": kalshi.get("fetched_at"), "error": kalshi.get("error"), "games": len(kalshi.get("games", []))},
    }

    games, pg = data.games(), data.player_games()
    log["graded"] = grade(games)

    now = datetime.now(timezone.utc)
    today = now.astimezone(ET).date()
    eng = Engine(games, pg, load_params(), pbp.luck_by_game())
    eng.p.box_beta = box_beta(eng, today)
    st = eng.fit(today + timedelta(days=1))  # everything finished so far

    played = pg.filter(~pl.col("dnp") & (pl.col("min") > 0)).sort("date")
    pg_last = {pid: (t, d.toordinal()) for pid, t, d in played.select(["pid", "team", "date"]).iter_rows()}
    inj_by_pid = {p["pid"]: p for p in injuries.get("players", []) if p.get("pid")}
    inj_by_name = {(p.get("player") or "").lower(): p for p in injuries.get("players", [])}
    kal = {(g["date"], g["home"], g["away"]): g for g in kalshi.get("games", [])}
    spr = {(g["date"], g["home"], g["away"]): g for g in spreads.get("games", [])}

    upcoming = games.filter(
        pl.col("date").is_in([today, today + timedelta(days=1)]) & ~pl.col("completed")
    )
    made = 0
    slate = []
    for day, day_games in upcoming.group_by("date", maintain_order=True):
        dkey = str(day[0])
        recs = load_day(dkey)
        for g in day_games.iter_rows(named=True):
            k = str(g["game_id"])
            tip = g["tip_utc"]
            if k in recs and (recs[k].get("locked") or (tip and tip <= now)):
                recs[k]["locked"] = True
                continue
            if tip and tip <= now:
                continue  # never predict after tip-off
            hl, h_out, h_q = build_lineup(eng, g["home"], day[0], rosters, inj_by_pid, inj_by_name, pg_last)
            al, a_out, a_q = build_lineup(eng, g["away"], day[0], rosters, inj_by_pid, inj_by_name, pg_last)
            if not hl or not al:
                continue
            pr = eng.predict(st, g["game_id"], g["home"], g["away"], hl, al)
            ph = pr["p_home"]
            pick = g["home"] if ph >= 0.5 else g["away"]
            mk = kal.get((dkey, g["home"], g["away"]))
            prev = recs.get(k, {})
            sp = spr.get((dkey, g["home"], g["away"]))
            spread = None
            if sp and sp.get("home_line") is not None:
                hl_ = float(sp["home_line"])
                if pr["margin"] + hl_ != 0:
                    side = g["home"] if pr["margin"] + hl_ > 0 else g["away"]
                    side_line = hl_ if side == g["home"] else -hl_
                else:
                    side, side_line = None, None
                spread = {"home_line": hl_, "source": sp.get("source"),
                          "market_p_home_cover": sp.get("market_p_home_cover"),
                          "teleios_p_home_cover": round(phi((pr["margin"] + hl_) / eng.p.sigma), 4),
                          "pick": side, "pick_line": side_line}
            elif prev.get("spread"):
                spread = prev["spread"]   # keep the last line seen if the source drops it

            def top(parts):
                return [{"player": eng.names.get(pid), "minutes": round(mn, 1), "rating": round(v, 2),
                         "impact": round(c, 2)} for pid, mn, v, c in parts[:8]]

            rec = {
                "game_id": g["game_id"], "date": dkey, "tip_utc": str(tip),
                "home": g["home"], "away": g["away"], "neutral": g["neutral"],
                "p_home": round(ph, 4), "margin": round(pr["margin"], 2),
                "pick": pick, "chance": round(max(ph, 1 - ph), 4),
                "market_p_home": mk.get("p_home") if mk else None,
                "spread": spread,
                "edge_vs_market": (round(ph - mk["p_home"], 4) if mk and mk.get("p_home") is not None else None),
                "why": {
                    "home_court_and_situation": {c: round(v, 2) for c, v in pr["situation"].items() if abs(v) > 0.005},
                    "rest_days": {g["home"]: pr["rest_home"], g["away"]: pr["rest_away"]},
                    "strength": {g["home"]: round(pr["home_strength"], 2), g["away"]: round(pr["away_strength"], 2)},
                    "team_term": {g["home"]: round(st["team"][g["home"]], 2), g["away"]: round(st["team"][g["away"]], 2)},
                    "top_players": {g["home"]: top(pr["home_parts"]), g["away"]: top(pr["away_parts"])},
                    "out": {g["home"]: h_out, g["away"]: a_out},
                    "questionable": {g["home"]: h_q, g["away"]: a_q},
                },
                "first": prev.get("first") or {"at": data.now_utc(), "p_home": round(ph, 4),
                                                "market_p_home": mk.get("p_home") if mk else None,
                                                "spread": spread},
                "frozen_at": data.now_utc(),
                "locked": False,
            }
            recs[k] = rec
            made += 1
            slate.append({"game_id": g["game_id"], "date": day[0], "home": g["home"], "away": g["away"],
                          "tip_utc": str(tip), "margin": pr["margin"],
                          "lineups": {g["home"]: hl, g["away"]: al},
                          "questionable": {g["home"]: [x["pid"] for x in h_q], g["away"]: [x["pid"] for x in a_q]}})
        save_day(dkey, recs)
    log["predicted"] = made
    player_daily.lock_started(games, now)
    log["players_graded"] = player_daily.grade(games, pg)
    try:
        log["player_projections"] = player_daily.update(games, pg, slate, now)
    except Exception as e:  # noqa: BLE001  (a player-model failure must not stop game predictions)
        log["player_projections"] = f"failed: {type(e).__name__}: {str(e)[:200]}"

    write_ratings(eng, st, today, rosters, inj_by_pid, inj_by_name, pg_last)
    profiles.build()
    sc = scorecard.build()
    log["scorecard"] = {k: sc.get("overall", {}).get(k) for k in ("games", "accuracy", "brier")}
    nb = notebook.build()
    log["notebook_entries"] = len(nb.get("entries", []))
    log["unmatched_team_names"] = teams.unmatched()

    hist_path = data.DATA / "run_log.json"
    try:
        hist = json.loads(hist_path.read_text())
    except Exception:  # noqa: BLE001
        hist = []
    hist = (hist + [log])[-200:]
    hist_path.write_text(json.dumps(hist, indent=1, default=str))
    print(json.dumps(log, indent=1, default=str))


def write_ratings(eng: Engine, st: dict, today: date, rosters, inj_by_pid, inj_by_name, pg_last) -> None:
    """Every team's strength tonight (with injuries) and fully healthy, plus
    each player's rating, for the analyst to read."""
    out = {"as_of": data.now_utc(), "games_in_fit": st["n_games"],
           "situation_effects": {k: round(v, 3) for k, v in st["sit"].items()},
           "sigma": eng.p.sigma, "teams": {}, "players": []}
    no_inj = {}
    for t in teams.CODES:
        lu, outs, q = build_lineup(eng, t, today, rosters, inj_by_pid, inj_by_name, pg_last)
        healthy, _o, _q = build_lineup(eng, t, today, rosters, no_inj, no_inj, pg_last)
        s_now, parts = eng.team_strength(st, t, lu)
        s_full, _p = eng.team_strength(st, t, healthy)
        out["teams"][t] = {"name": teams.TEAMS[t][0], "strength_now": round(s_now, 2),
                           "strength_healthy": round(s_full, 2), "team_term": round(st["team"][t], 2),
                           "out": [o["player"] for o in outs], "questionable": [x["player"] for x in q],
                           "rotation": [{"player": eng.names.get(pid), "minutes": round(m, 1), "rating": round(v, 2)}
                                        for pid, m, v, _c in parts]}
    a = today.toordinal()
    for pid, (team, last) in pg_last.items():
        j = eng.pidx.get(pid)
        if j is None or a - last > 120:
            continue
        out["players"].append({"pid": pid, "player": eng.names.get(pid), "team": team,
                               "rating": round(float(st["player"][j]), 2),
                               "box_prior": round(float(st["prior"][j]), 2),
                               "recent_minutes_weight": round(float(st["minutes_seen"][j]), 0)})
    out["players"].sort(key=lambda x: -x["rating"])
    rank = sorted(out["teams"], key=lambda t: -out["teams"][t]["strength_now"])
    for i, t in enumerate(rank, 1):
        out["teams"][t]["rank_now"] = i
    RATINGS_PATH.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
