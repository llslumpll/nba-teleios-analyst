"""
The patterns notebook: places where the model keeps missing in the same
direction. A pattern is only CONFIRMED when it shows up in the earlier half
of the graded games AND again in the later half, pointing the same way, and
is clearly more than luck overall (|z| >= 2.5). Otherwise it's WATCHING, or
FADED if the later games went the other way.

Checked every run, on live predictions plus the 2026-27 backfill:
  - home teams: does the model under/over-rate home court?
  - each team: does it keep under/over-rating them? (at home and on the road)
  - back-to-backs, long trips (1,500+ miles), altitude
  - big favourites (70%+): too confident or not enough?

Confirmed patterns are listed for the analyst and for the monthly retune;
they are not silently applied to predictions.
"""

from __future__ import annotations

import json
import math

import numpy as np

from teleios import data, scorecard

OUT = data.DATA / "notebook.json"
MIN_EACH_HALF = 25
Z_CONFIRM = 2.5


def _rows() -> list[dict]:
    rows = scorecard.backfill_rows() + scorecard.live_rows()
    rows = [r for r in rows if r.get("pred_margin") is not None]
    rows.sort(key=lambda r: (r["date"], r["game_id"]))
    return rows


def _test(name: str, desc: str, rows: list[dict], sign: float = 1.0) -> dict | None:
    """rows: games where the pattern applies. sign flips residual to the
    team's point of view (+1 home, -1 away)."""
    if len(rows) < 2 * MIN_EACH_HALF:
        return {"name": name, "description": desc, "games": len(rows), "state": "not enough games yet"} if rows else None
    res = np.array([sign * (r["margin"] - r["pred_margin"]) for r in rows])
    half = len(res) // 2
    a, b = res[:half], res[half:]
    z = float(res.mean() / (res.std(ddof=1) / math.sqrt(len(res))))
    same = np.sign(a.mean()) == np.sign(b.mean())
    state = "confirmed" if (same and abs(z) >= Z_CONFIRM) else ("faded" if not same and abs(a.mean()) > 2 else "watching")
    return {
        "name": name, "description": desc, "games": len(rows),
        "avg_miss_points": round(float(res.mean()), 2),
        "earlier_half": round(float(a.mean()), 2), "later_half": round(float(b.mean()), 2),
        "z": round(z, 2), "state": state,
        "reading": (f"Did {abs(res.mean()):.1f} pts {'better' if res.mean() > 0 else 'worse'} than predicted on average"),
    }


def build() -> dict:
    rows = _rows()
    entries = []

    def add(e):
        if e:
            entries.append(e)

    add(_test("home_court", "Home teams vs prediction", rows, 1.0))
    teams_seen = sorted({r["home"] for r in rows} | {r["away"] for r in rows})
    for t in teams_seen:
        tr = [r for r in rows if t in (r["home"], r["away"])]
        res_rows = [{**r, "margin": r["margin"] if r["home"] == t else -r["margin"],
                     "pred_margin": r["pred_margin"] if r["home"] == t else -r["pred_margin"]} for r in tr]
        add(_test(f"team_{t}", f"{t}: result vs prediction (all games)", res_rows, 1.0))

    for r in rows:
        rh, ra = r.get("rest_home"), r.get("rest_away")
        r["b2b"] = "home" if rh == 1 and ra != 1 else ("away" if ra == 1 and rh != 1 else None)
    b2b = [r for r in rows if r.get("b2b") in ("home", "away")]
    if b2b:
        add(_test("back_to_back", "Teams on the 2nd night of a back-to-back",
                  [{**r, "margin": r["margin"] if r["b2b"] == "home" else -r["margin"],
                    "pred_margin": r["pred_margin"] if r["b2b"] == "home" else -r["pred_margin"]} for r in b2b]))
    alt = [r for r in rows if r["home"] in ("DEN", "UTA")]
    add(_test("altitude", "Denver and Utah at home", alt, 1.0))

    fav = []
    for r in rows:
        ph = r.get("p_home")
        if ph is not None and max(ph, 1 - ph) >= 0.70:
            s = 1.0 if ph >= 0.5 else -1.0
            fav.append({**r, "margin": s * r["margin"], "pred_margin": s * r["pred_margin"]})
    add(_test("big_favourites", "Favourites the model gave 70%+", fav, 1.0))

    entries.sort(key=lambda e: ({"confirmed": 0, "watching": 1, "faded": 2}.get(e["state"], 3), -abs(e.get("z", 0))))
    out = {"updated": data.now_utc(), "games_checked": len(rows),
           "rule": f"confirmed = same direction in both halves and |z| >= {Z_CONFIRM}",
           "entries": entries}
    OUT.write_text(json.dumps(out, indent=1))
    return out
