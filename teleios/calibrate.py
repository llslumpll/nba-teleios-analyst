"""
Over/under calibration: corrects Teleios's chance of a player going OVER a
betting line, learned from its own graded results.

Why: on the first night, Teleios's projections sat right on the betting
lines for rebounds, assists and threes, yet it still called the under 55-60%
of the time. Its spread of outcomes is too wide and lopsided, so raw
"chance of over" runs low.

How (runs every day, after grading):
  1. collect every graded player line: Teleios's raw chance of over, the
     stat, and whether it went over
  2. wait for at least MIN_NIGHTS nights and MIN_LINES lines
  3. hold out the most recent nights (at least 30% of lines): the fit never
     sees them
  4. fit candidates on the older nights (logistic recalibration):
       A  logit(p') = a + b * logit(p)
       B  same, plus a small per-stat shift (pulled toward 0)
  5. score raw and each candidate on the held-out nights (Brier: lower is
     better). Adopt the best candidate only if it beats raw by at least
     MIN_GAIN; then refit it on all nights and save it
  6. otherwise keep raw chances (and drop any earlier correction that no
     longer earns its place)

Every run is logged in data/calibration_log.json; the active correction is
data/ou_calibration.json. Projections and ranges are unchanged: only the
chance of over/under at a betting line is corrected.
"""

from __future__ import annotations

import json
import math

import numpy as np

from teleios import data

DIR = data.DATA / "player_projections"
ACTIVE = data.DATA / "ou_calibration.json"
LOG = data.DATA / "calibration_log.json"
MIN_NIGHTS = 3
MIN_LINES = 1200
HELD_SHARE = 0.30
MIN_GAIN = 0.002
STATS = ("pts", "reb", "ast", "fg3m", "pra")


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def rows() -> list[tuple]:
    out = []
    for f in sorted(DIR.glob("*.json")) if DIR.exists() else []:
        for g in json.loads(f.read_text()).values():
            for p in g.get("players", []):
                for s, m in (p.get("market") or {}).items():
                    if "went_over" in m:
                        raw = m.get("teleios_p_over_raw", m["teleios_p_over"])
                        out.append((f.stem, s, float(raw), 1.0 if m["went_over"] else 0.0))
    return out


def _design(logit_p, stats, per_stat: bool):
    cols = [np.ones_like(logit_p), logit_p]
    if per_stat:
        cols += [(stats == s).astype(float) for s in STATS]
    return np.column_stack(cols)


def _fit(X, y, ridge_stat=2.0, iters=50):
    """Logistic regression by Newton steps; per-stat columns get a ridge."""
    w = np.zeros(X.shape[1])
    w[1] = 1.0
    lam = np.zeros(X.shape[1])
    lam[2:] = ridge_stat
    for _ in range(iters):
        p = _sig(X @ w)
        g = X.T @ (p - y) + lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(lam + 1e-6)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def brier(p, y):
    return float(((p - y) ** 2).mean())


def apply(p_raw: float, stat: str, cal: dict | None = None) -> float:
    """Corrected chance of OVER (unchanged if no correction is active)."""
    cal = cal if cal is not None else load()
    if not cal or not cal.get("active"):
        return p_raw
    x = cal["a"] + cal["b"] * float(_logit(np.array([p_raw]))[0]) + cal.get("stat_shift", {}).get(stat, 0.0)
    return float(1 / (1 + math.exp(-x)))


def load() -> dict:
    try:
        return json.loads(ACTIVE.read_text())
    except Exception:  # noqa: BLE001
        return {}


def update() -> dict:
    R = rows()
    nights = sorted({r[0] for r in R})
    rec = {"run_at": data.now_utc(), "lines": len(R), "nights": len(nights)}
    if len(nights) < MIN_NIGHTS or len(R) < MIN_LINES:
        rec["state"] = f"waiting: {len(nights)}/{MIN_NIGHTS} nights, {len(R)}/{MIN_LINES} lines"
        return _log(rec)
    # hold out the newest nights covering >= HELD_SHARE of lines
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
    arr = lambda rr: (np.array([r[2] for r in rr]), np.array([r[1] for r in rr]), np.array([r[3] for r in rr]))
    ptr, str_, ytr = arr(tr)
    pte, ste, yte = arr(te)
    raw_b = brier(pte, yte)
    cands = {}
    for name, per in (("A_global", False), ("B_per_stat", True)):
        w = _fit(_design(_logit(ptr), str_, per), ytr)
        pc = _sig(_design(_logit(pte), ste, per) @ w)
        cands[name] = {"held_out_brier": round(brier(pc, yte), 4), "per_stat": per}
    best = min(cands, key=lambda k: cands[k]["held_out_brier"])
    gain = raw_b - cands[best]["held_out_brier"]
    rec.update(held_out_nights=held, fit_nights=[d for d in nights if d not in held],
               held_out_lines=len(te), raw_held_out_brier=round(raw_b, 4), candidates=cands, best=best,
               gain=round(gain, 4))
    if gain >= MIN_GAIN:
        p_all, s_all, y_all = arr(R)
        per = cands[best]["per_stat"]
        w = _fit(_design(_logit(p_all), s_all, per), y_all)
        cal = {"active": True, "kind": best, "a": float(w[0]), "b": float(w[1]),
               "stat_shift": ({s: float(v) for s, v in zip(STATS, w[2:])} if per else {}),
               "fitted_on_nights": nights, "lines": len(R),
               "held_out_test": {"raw_brier": round(raw_b, 4), "corrected_brier": cands[best]["held_out_brier"],
                                 "nights": held}, "adopted_at": data.now_utc()}
        ACTIVE.write_text(json.dumps(cal, indent=1))
        rec["state"] = f"adopted {best}: held-out Brier {raw_b:.4f} -> {cands[best]['held_out_brier']:.4f}"
    else:
        if load().get("active"):
            ACTIVE.write_text(json.dumps({"active": False, "dropped_at": data.now_utc(),
                                          "why": f"no longer beats raw on held-out nights (gain {gain:.4f})"}, indent=1))
        rec["state"] = f"kept raw: best candidate gained only {gain:.4f} on held-out nights (need {MIN_GAIN})"
    return _log(rec)


def _log(rec: dict) -> dict:
    try:
        log = json.loads(LOG.read_text())
    except Exception:  # noqa: BLE001
        log = []
    LOG.write_text(json.dumps((log + [rec])[-120:], indent=1))
    return rec


if __name__ == "__main__":
    print(json.dumps(update(), indent=1))
