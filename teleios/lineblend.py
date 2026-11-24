"""
Line blend: borrow a little of what the betting market knows.

Prop Edge blends about 20% of the betting line into its player projections,
and on Nov 23 its projections were more accurate than Teleios's on points,
PRA and assists. This tests the same idea for Teleios, automatically.

For each player stat with a Kalshi line, the market's center is where the
line plus its price put the expected value:
    center = line + z(market chance of over) * spread
(spread from Teleios's own variance model at that line). The blended
projection is
    blended = (1 - w) * Teleios + w * center,   one w per stat, 0 to 0.8

Daily, after grading (same rules as the over/under correction):
  - wait for MIN_NIGHTS nights and MIN_LINES graded lines
  - pick each stat's w on the older nights (lowest average miss)
  - test on the newest nights (>= 30% of lines), never seen while picking w
  - adopt only if the average miss improves by >= 0.5% across stats and no
    stat gets more than 1% worse; drop again if it stops earning its place

Only players with a betting line are blended; others keep Teleios's own
projection. The unblended projection is always kept as proj_raw.
Active setting: data/line_blend.json. Log: data/line_blend_log.json.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy import stats as sst

from teleios import data

DIR = data.DATA / "player_projections"
ACTIVE = data.DATA / "line_blend.json"
LOG = data.DATA / "line_blend_log.json"
MIN_NIGHTS = 3
MIN_LINES = 1200
HELD_SHARE = 0.30
STATS = ("pts", "reb", "ast", "fg3m", "pra")
GRID = [round(x, 2) for x in np.arange(0.0, 0.81, 0.1)]


def center(line: float, market_p_over: float, var: list[float]) -> float:
    v = max((var[0] * line + var[1] * line * line) * (var[2] if len(var) > 2 else 1.0), 1e-6)
    z = sst.norm.ppf(min(max(market_p_over, 0.02), 0.98))
    return float(line + z * math.sqrt(v))


def load() -> dict:
    try:
        return json.loads(ACTIVE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def blend(proj: float, line: float, market_p_over: float, stat: str, var: list[float], cfg: dict | None = None) -> float:
    cfg = cfg if cfg is not None else load()
    w = (cfg.get("w") or {}).get(stat, 0.0) if cfg.get("active") else 0.0
    if not w:
        return proj
    return (1 - w) * proj + w * center(line, market_p_over, var)


def rows(var: dict) -> list[tuple]:
    out = []
    for f in sorted(DIR.glob("*.json")) if DIR.exists() else []:
        for g in json.loads(f.read_text()).values():
            for p in g.get("players", []):
                if "actual" not in p:
                    continue
                for s, m in (p.get("market") or {}).items():
                    if s not in STATS or s not in var:
                        continue
                    raw = (p.get("proj_raw") or {}).get(s, p["proj"][s])
                    out.append((f.stem, s, float(raw), center(m["line"], m["market_p_over"], var[s]), float(p["actual"][s])))
    return out


def update(var: dict) -> dict:
    R = rows(var)
    nights = sorted({r[0] for r in R})
    rec = {"run_at": data.now_utc(), "lines": len(R), "nights": len(nights)}
    if len(nights) < MIN_NIGHTS or len(R) < MIN_LINES:
        rec["state"] = f"waiting: {len(nights)}/{MIN_NIGHTS} nights, {len(R)}/{MIN_LINES} lines"
        return _log(rec)
    held, n = [], 0
    for d in reversed(nights):
        held.append(d)
        n += sum(1 for r in R if r[0] == d)
        if n >= HELD_SHARE * len(R) and len(held) < len(nights):
            break
    tr = [r for r in R if r[0] not in held]
    te = [r for r in R if r[0] in held]
    if not tr:
        rec["state"] = "waiting: not enough older nights to fit on"
        return _log(rec)
    w, res = {}, {}
    for s in STATS:
        a = np.array([(r[2], r[3], r[4]) for r in tr if r[1] == s])
        b = np.array([(r[2], r[3], r[4]) for r in te if r[1] == s])
        if len(a) < 50 or len(b) < 30:
            w[s] = 0.0
            continue
        mae = {g: float(np.abs((1 - g) * a[:, 0] + g * a[:, 1] - a[:, 2]).mean()) for g in GRID}
        w[s] = min(mae, key=mae.get)
        raw = float(np.abs(b[:, 0] - b[:, 2]).mean())
        new = float(np.abs((1 - w[s]) * b[:, 0] + w[s] * b[:, 1] - b[:, 2]).mean())
        res[s] = {"w": w[s], "held_out_raw_miss": round(raw, 3), "held_out_blended_miss": round(new, 3),
                  "change": round((raw - new) / raw, 4)}
    gains = [v["change"] for v in res.values()]
    rec.update(held_out_nights=held, fit_nights=[d for d in nights if d not in held], held_out_lines=len(te), by_stat=res)
    ok = bool(gains) and float(np.mean(gains)) >= 0.005 and min(gains) >= -0.01
    if ok:
        # refit each w on all nights, keep stats whose held-out test didn't hurt
        wa = {}
        for s in STATS:
            a = np.array([(r[2], r[3], r[4]) for r in R if r[1] == s])
            if s not in res or len(a) < 80 or res[s]["change"] < 0:
                wa[s] = 0.0
                continue
            mae = {g: float(np.abs((1 - g) * a[:, 0] + g * a[:, 1] - a[:, 2]).mean()) for g in GRID}
            wa[s] = min(mae, key=mae.get)
        ACTIVE.write_text(json.dumps({"active": True, "w": wa, "held_out_test": res, "fitted_on_nights": nights,
                                      "adopted_at": data.now_utc()}, indent=1))
        rec["state"] = f"adopted: average miss {float(np.mean(gains)) * 100:.1f}% lower on held-out nights"
    else:
        if load().get("active"):
            ACTIVE.write_text(json.dumps({"active": False, "dropped_at": data.now_utc(),
                                          "why": "no longer beats Teleios's own projections on held-out nights"}, indent=1))
        rec["state"] = (f"kept Teleios's own projections: held-out change {float(np.mean(gains)) * 100:.1f}% "
                        f"(need +0.5% with no stat worse than -1%)" if gains else "kept: too few lines per stat")
    return _log(rec)


def _log(rec: dict) -> dict:
    try:
        log = json.loads(LOG.read_text())
    except Exception:  # noqa: BLE001
        log = []
    LOG.write_text(json.dumps((log + [rec])[-120:], indent=1))
    return rec
