"""
The scorecard: how good are Teleios's game predictions, really?

Main score (prediction quality):
  accuracy   how often the favourite it picked won
  Brier      mean squared error of the win chance (0 = perfect, 0.25 = coin flip)
  log loss   punishes confident misses hard (0.693 = coin flip)
  calibration  when it says 70%, do those teams win ~70%?

Benchmark: on the same games, the market's chance (Kalshi live; FanDuel
closing line for the backfill) scored the same way. Lower Brier than the
market on enough games = Teleios knows something the market doesn't.

Live predictions and the backfill are always kept apart: the backfill knew
who played each game (as a final injury report would) and was made after
the fact, so it can't count as a live record.
"""

from __future__ import annotations

import json
import math

import numpy as np

from teleios import data

PRED_DIR = data.DATA / "predictions"
BACKFILL = data.DATA / "backfill_2027.json"
OUT = data.DATA / "scorecard.json"


def _metrics(rows: list[dict], key: str = "p_home") -> dict:
    rows = [r for r in rows if r.get(key) is not None]
    if not rows:
        return {"games": 0}
    p = np.clip(np.array([r[key] for r in rows], float), 1e-4, 1 - 1e-4)
    won = np.array([1.0 if r["margin"] > 0 else 0.0 for r in rows])
    return {
        "games": len(rows),
        "accuracy": round(float(((p >= 0.5) == (won > 0.5)).mean()), 4),
        "brier": round(float(((p - won) ** 2).mean()), 4),
        "log_loss": round(float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean()), 4),
    }


def _margin(rows: list[dict]) -> dict:
    """How close the projected margin was: average miss, share within 5 and
    10 points, and bias (+ = home teams did better than projected)."""
    rows = [r for r in rows if r.get("pred_margin") is not None]
    if not rows:
        return {"games": 0}
    err = np.array([r["margin"] - r["pred_margin"] for r in rows], float)
    return {"games": len(rows), "avg_miss": round(float(np.abs(err).mean()), 2),
            "median_miss": round(float(np.median(np.abs(err))), 2),
            "within_5": round(float((np.abs(err) <= 5).mean()), 3),
            "within_10": round(float((np.abs(err) <= 10).mean()), 3),
            "bias_home": round(float(err.mean()), 2)}


def _calibration(rows: list[dict], key: str = "p_home") -> list[dict]:
    out = []
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    fav = []
    for r in rows:
        if r.get(key) is None:
            continue
        ph = r[key]
        fav.append((max(ph, 1 - ph), 1.0 if (r["margin"] > 0) == (ph >= 0.5) else 0.0))
    for lo, hi in bins:
        b = [x for x in fav if lo <= x[0] < hi]
        if b:
            out.append({"range": f"{int(lo*100)}-{min(int(hi*100), 100)}%", "games": len(b),
                        "said": round(float(np.mean([x[0] for x in b])), 3),
                        "won": round(float(np.mean([x[1] for x in b])), 3)})
    return out


def _vs_market(rows: list[dict], market_key: str, label: str = "market") -> dict:
    both = [r for r in rows if r.get("p_home") is not None and r.get(market_key) is not None]
    if not both:
        return {"games": 0}
    t, m = _metrics(both, "p_home"), _metrics(both, market_key)
    p = np.array([r["p_home"] for r in both])
    q = np.array([r[market_key] for r in both])
    won = np.array([1.0 if r["margin"] > 0 else 0.0 for r in both])
    diff = (p - won) ** 2 - (q - won) ** 2   # negative = Teleios better
    se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    z = float(diff.mean() / se) if se and se == se and se > 0 else 0.0
    disagree = [r for r in both if (r["p_home"] >= 0.5) != (r[market_key] >= 0.5)]
    return {
        "games": len(both),
        "teleios": t, label: m,
        "brier_gap": round(float(diff.mean()), 4),
        "verdict": ("Teleios ahead" if z < -2 else f"{label} ahead" if z > 2 else "too close to call yet"),
        "z": round(z, 2),
        "disagreements": {
            "games": len(disagree),
            "teleios_right": sum(1 for r in disagree if (r["margin"] > 0) == (r["p_home"] >= 0.5)),
        },
    }


def _spread(rows: list[dict]) -> dict:
    """Against the spread: Teleios takes whichever side of the main line its
    projected margin favours; a win is that side covering. Also compares its
    chance of the home team covering with the market's (Kalshi), when quoted."""
    rr = [r for r in rows if (r.get("spread_result") or {}).get("pick_result")]
    if not rr:
        return {"games": 0}
    w = sum(1 for r in rr if r["spread_result"]["pick_result"] == "win")
    lo = sum(1 for r in rr if r["spread_result"]["pick_result"] == "loss")
    pu = len(rr) - w - lo
    out = {"games": len(rr), "wins": w, "losses": lo, "pushes": pu,
           "win_rate": round(w / (w + lo), 4) if w + lo else None,
           "breakeven_note": "About 52.4% is needed to beat standard -110 betting prices."}
    by = {}
    for r in rr:
        edge = abs(r["pred_margin"] + r["spread"]["home_line"])
        b = "0-2 pts off the line" if edge < 2 else "2-4 pts" if edge < 4 else "4+ pts"
        x = by.setdefault(b, [0, 0])
        if r["spread_result"]["pick_result"] != "push":
            x[0 if r["spread_result"]["pick_result"] == "win" else 1] += 1
    out["by_disagreement"] = {k: {"wins": v[0], "losses": v[1]} for k, v in sorted(by.items())}
    q = [r for r in rr if r["spread"].get("market_p_home_cover") is not None and r["spread_result"].get("home") != "push"]
    if q:
        y = np.array([1.0 if r["spread_result"]["home"] == "covered" else 0.0 for r in q])
        t = np.array([r["spread"]["teleios_p_home_cover"] for r in q])
        m = np.array([r["spread"]["market_p_home_cover"] for r in q])
        out["vs_kalshi"] = {"games": len(q), "teleios_brier": round(float(((t - y) ** 2).mean()), 4),
                            "market_brier": round(float(((m - y) ** 2).mean()), 4)}
    return out


def player_log() -> dict:
    """Every graded player projection (played games only): projection,
    actual, and Teleios's over/under call on each Kalshi line."""
    from teleios import player_daily
    cols = ["date", "player", "team", "opp", "proj_min", "min",
            "pts", "pts_a", "reb", "reb_a", "ast", "ast_a", "fg3m", "fg3m_a", "pra", "pra_a", "in80", "calls"]
    rows = []
    d = player_daily.DIR
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        for r in json.loads(f.read_text()).values():
            for p in r.get("players", []):
                if "actual" not in p:
                    continue
                pj, ac = p["proj"], p["actual"]
                calls = []
                for st, mk in (p.get("market") or {}).items():
                    if "went_over" not in mk:
                        continue
                    a = ac.get(st)
                    if a == mk["line"]:
                        continue
                    side = "o" if mk["teleios_p_over"] >= 0.5 else "u"
                    calls.append([st, mk["line"], side, round(mk["teleios_p_over"], 3), round(mk["market_p_over"], 3),
                                  1 if (side == "o") == bool(mk["went_over"]) else 0])
                inr = p.get("in80") or {}
                rows.append([r["date"], p["player"], p["team"], p.get("opp"), round(p["minutes"], 1), p.get("actual_minutes"),
                             round(pj["pts"], 1), ac["pts"], round(pj["reb"], 1), ac["reb"], round(pj["ast"], 1), ac["ast"],
                             round(pj["fg3m"], 1), ac["fg3m"], round(pj["pra"], 1), ac["pra"],
                             sum(1 for v in inr.values() if v), calls])
    rows.sort(key=lambda x: (x[0], x[6]), reverse=True)   # newest first, then biggest projected scorers
    calls = [c for r in rows for c in r[17]]
    summary = {"player_games": len(rows), "lines_called": len(calls),
               "calls_right": sum(c[5] for c in calls)}
    by = {}
    for c in calls:
        x = by.setdefault(c[0], [0, 0])
        x[0] += c[5]
        x[1] += 1
    summary["calls_by_stat"] = {k: {"right": v[0], "lines": v[1]} for k, v in by.items()}
    return {"cols": cols, "rows": rows, "summary": summary,
            "note": "calls = [stat, line, o/u, Teleios over chance, market over chance, 1 if right]; in80 = stats (of 5) inside the 80% range"}


def live_rows() -> list[dict]:
    rows = []
    for f in sorted(PRED_DIR.glob("*.json")):
        for r in json.loads(f.read_text()).values():
            if r.get("result"):
                rows.append({"game_id": r["game_id"], "date": r["date"], "home": r["home"], "away": r["away"],
                             "p_home": r["p_home"], "market_p_home": r.get("market_p_home"),
                             "margin": r["result"]["margin"], "pred_margin": r["margin"],
                             "home_score": r["result"].get("home_score"), "away_score": r["result"].get("away_score"),
                             "spread": r.get("spread"), "spread_result": (r["result"].get("spread") or {}),
                             "rest_home": (r.get("why") or {}).get("rest_days", {}).get(r["home"]),
                             "rest_away": (r.get("why") or {}).get("rest_days", {}).get(r["away"])})
    return rows


def backfill_rows() -> list[dict]:
    try:
        return json.loads(BACKFILL.read_text())["games"]
    except Exception:  # noqa: BLE001
        return []


def build() -> dict:
    live = live_rows()
    back = backfill_rows()
    out = {
        "updated": data.now_utc(),
        "overall": _metrics(live),
        "live": {
            "note": "Frozen before tip-off with that day's injury report, graded after the game.",
            "metrics": _metrics(live),
            "calibration": _calibration(live),
            "vs_kalshi": _vs_market(live, "market_p_home", "kalshi"),
            "margin": _margin(live),
            "spread": _spread(live),
            "last_30_days": _metrics([r for r in live if r["date"] >= _days_ago(30)]),
        },
        "backfill_2026_27": {
            "note": ("Made after the fact, walk-forward (only earlier games used), knowing who played. "
                     "Benchmarks = FanDuel closing chance and Prop Edge's published chance (which is partly blended with the market), as Prop Edge saved them."),
            "metrics": _metrics(back),
            "margin": _margin(back),
            "calibration": _calibration(back),
            "vs_fanduel_close": _vs_market(back, "market_p_home", "fanduel_close"),
            "vs_prop_edge": _vs_market(back, "propedge_p_home", "prop_edge"),
            "prop_edge": _metrics(back, "propedge_p_home"),
        },
    }
    from teleios import player_daily
    out["players_live"] = {"note": "Players projected 20+ minutes; average miss vs the season-average baseline on the same players.",
                           **player_daily.scorecard()}
    try:
        pb = json.loads((data.DATA / "players_backtest.json").read_text())
        out["players_backtest_2025_26"] = {k: pb[k] for k in ("tuned_on", "tested_on", "test_avg_miss", "prop_edge_published", "ranges")}
        out["players_backtest_2025_26"]["season_avg_baseline"] = pb["baselines"]["season_avg"]
    except Exception:  # noqa: BLE001
        pass
    try:
        out["backtest_2025_26"] = {k: v for k, v in json.loads((data.DATA / "backtest.json").read_text()).items()
                                   if k in ("tuned_on", "tested_on", "chosen", "test", "calibration", "baselines")}
    except Exception:  # noqa: BLE001
        pass
    out["game_log"] = {"cols": ["date", "away", "home", "pick", "chance", "proj_pick_margin", "actual_pick_margin",
                                "margin_miss", "hit", "market_pick_chance", "spread_pick", "spread_line", "spread_result",
                                "home_score", "away_score"],
                       "rows": [[r["date"], r["away"], r["home"], (r["home"] if r["p_home"] >= 0.5 else r["away"]),
                                 round(max(r["p_home"], 1 - r["p_home"]), 3),
                                 round(r["pred_margin"] if r["p_home"] >= 0.5 else -r["pred_margin"], 1),
                                 r["margin"] if r["p_home"] >= 0.5 else -r["margin"],
                                 round(abs(r["margin"] - r["pred_margin"]), 1),
                                 (r["margin"] > 0) == (r["p_home"] >= 0.5),
                                 (None if r.get("market_p_home") is None else
                                  round(r["market_p_home"] if r["p_home"] >= 0.5 else 1 - r["market_p_home"], 3)),
                                 (r.get("spread") or {}).get("pick"), (r.get("spread") or {}).get("pick_line"),
                                 (r.get("spread_result") or {}).get("pick_result"),
                                 r.get("home_score"), r.get("away_score")]
                                for r in sorted(live, key=lambda x: (x["date"], x["game_id"]), reverse=True)]}
    out["player_log"] = player_log()
    OUT.write_text(json.dumps(out, indent=1))
    return out


def _days_ago(n: int) -> str:
    from datetime import datetime, timedelta, timezone
    return str((datetime.now(timezone.utc) - timedelta(days=n)).date())
