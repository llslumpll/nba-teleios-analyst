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


def live_rows() -> list[dict]:
    rows = []
    for f in sorted(PRED_DIR.glob("*.json")):
        for r in json.loads(f.read_text()).values():
            if r.get("result"):
                rows.append({"game_id": r["game_id"], "date": r["date"], "home": r["home"], "away": r["away"],
                             "p_home": r["p_home"], "market_p_home": r.get("market_p_home"),
                             "margin": r["result"]["margin"], "pred_margin": r["margin"],
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
            "last_30_days": _metrics([r for r in live if r["date"] >= _days_ago(30)]),
        },
        "backfill_2026_27": {
            "note": ("Made after the fact, walk-forward (only earlier games used), knowing who played. "
                     "Benchmarks = FanDuel closing chance and Prop Edge's published chance (which is partly blended with the market), as Prop Edge saved them."),
            "metrics": _metrics(back),
            "calibration": _calibration(back),
            "vs_fanduel_close": _vs_market(back, "market_p_home", "fanduel_close"),
            "vs_prop_edge": _vs_market(back, "propedge_p_home", "prop_edge"),
            "prop_edge": _metrics(back, "propedge_p_home"),
        },
    }
    try:
        out["backtest_2025_26"] = {k: v for k, v in json.loads((data.DATA / "backtest.json").read_text()).items()
                                   if k in ("tuned_on", "tested_on", "chosen", "test", "calibration", "baselines")}
    except Exception:  # noqa: BLE001
        pass
    OUT.write_text(json.dumps(out, indent=1))
    return out


def _days_ago(n: int) -> str:
    from datetime import datetime, timedelta, timezone
    return str((datetime.now(timezone.utc) - timedelta(days=n)).date())
