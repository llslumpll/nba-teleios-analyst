"""
Walk-forward backtest. For every game day of a season: fit on games
finished before that day, predict that day's games, score them.

Lineups: the backtest knows WHO played (as the injury report would tell you
before tip-off) but not how many minutes: each player gets his projected
minutes from earlier games, scaled to 240 per team. Live predictions do the
same with the injury report.

  python -m teleios.backtest            tune on 2024-25, test on 2025-26
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from datetime import date

import numpy as np
import polars as pl

from teleios import data
from teleios.model import Engine, Params, learn_box_beta, phi

OUT = data.DATA / "backtest.json"


def lineups_played(pg: pl.DataFrame) -> dict[int, dict[str, list[int]]]:
    out: dict[int, dict[str, list[int]]] = {}
    for gid, team, pid in pg.filter(~pl.col("dnp") & (pl.col("min") > 0)).select(["game_id", "team", "pid"]).iter_rows():
        out.setdefault(gid, {}).setdefault(team, []).append(pid)
    return out


def offseason_before(d: date) -> date:
    """July 1 before a date: when the box prior is re-learned."""
    y = d.year if d.month >= 7 else d.year - 1
    return date(y, 7, 1)


def run_season(engine: Engine, season: int, played: dict, season_types=(2, 3, 5),
               until: date | None = None, beta_cache: dict | None = None) -> list[dict]:
    g = engine.games.filter(
        (pl.col("season") == season) & pl.col("completed") & pl.col("season_type").is_in(list(season_types))
        & (pl.col("home_score") + pl.col("away_score") > 0)
    )
    if until:
        g = g.filter(pl.col("date") < until)
    preds = []
    beta_cache = beta_cache if beta_cache is not None else {}
    for day, day_games in g.group_by("date", maintain_order=True):
        day = day[0]
        off = offseason_before(day)
        if off not in beta_cache:
            beta_cache[off] = learn_box_beta(engine, off)
        engine.p.box_beta = beta_cache[off]
        st = engine.fit(day)
        a = day.toordinal()
        for row in day_games.iter_rows(named=True):
            lu = played.get(row["game_id"])
            if not lu or row["home"] not in lu or row["away"] not in lu:
                continue
            hl = engine.scale_to_team(engine.projected_minutes(lu[row["home"]], a))
            al = engine.scale_to_team(engine.projected_minutes(lu[row["away"]], a))
            pr = engine.predict(st, row["game_id"], row["home"], row["away"], hl, al)
            preds.append({
                "game_id": row["game_id"], "date": str(day), "season_type": row["season_type"],
                "home": row["home"], "away": row["away"],
                "margin": pr["margin"], "actual": row["home_score"] - row["away_score"],
            })
    return preds


def score(preds: list[dict], sigma: float) -> dict:
    if not preds:
        return {}
    m = np.array([p["margin"] for p in preds])
    a = np.array([p["actual"] for p in preds])
    p = np.array([phi(x / sigma) for x in m])
    won = (a > 0).astype(float)
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    return {
        "games": len(preds),
        "accuracy": float(((p > 0.5) == (won > 0.5)).mean()),
        "brier": float(((p - won) ** 2).mean()),
        "log_loss": float(-(won * np.log(pc) + (1 - won) * np.log(1 - pc)).mean()),
        "margin_mae": float(np.abs(m - a).mean()),
        "margin_rmse": float(np.sqrt(((m - a) ** 2).mean())),
        "home_win_rate": float(won.mean()),
    }


def best_sigma(preds: list[dict]) -> float:
    m = np.array([p["margin"] for p in preds])
    a = np.array([p["actual"] for p in preds])
    won = (a > 0).astype(float)
    best = None
    for s in np.arange(9.0, 17.01, 0.25):
        p = np.clip([phi(x / s) for x in m], 1e-4, 1 - 1e-4)
        ll = float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean())
        if best is None or ll < best[1]:
            best = (float(s), ll)
    return best[0]


def calibration(preds: list[dict], sigma: float, bins=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> list[dict]:
    """Favourite's stated chance vs how often favourites actually won."""
    rows = []
    fav = []
    for p in preds:
        ph = phi(p["margin"] / sigma)
        c = max(ph, 1 - ph)
        w = (p["actual"] > 0) if ph >= 0.5 else (p["actual"] < 0)
        fav.append((c, float(w)))
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = [x for x in fav if lo <= x[0] < hi or (hi == 1.0 and x[0] == 1.0)]
        if b:
            rows.append({"range": f"{int(lo*100)}-{int(hi*100)}%", "games": len(b),
                         "said": round(float(np.mean([x[0] for x in b])), 3),
                         "won": round(float(np.mean([x[1] for x in b])), 3)})
    return rows


# --- baseline: classic Elo with margin of victory (FiveThirtyEight style) ---

def elo_preds(games: pl.DataFrame, seasons: set[int], k=20.0, hca=100.0, carry=0.75) -> list[dict]:
    r: dict[str, float] = {}
    cur_season = None
    out = []
    for g in games.filter(pl.col("completed") & (pl.col("home_score") + pl.col("away_score") > 0)).iter_rows(named=True):
        if g["season"] != cur_season:
            r = {t: 1505 + carry * (v - 1505) for t, v in r.items()}
            cur_season = g["season"]
        rh, ra = r.get(g["home"], 1505.0), r.get(g["away"], 1505.0)
        h = 0.0 if g["neutral"] else hca
        diff = rh + h - ra
        p = 1 / (1 + 10 ** (-diff / 400))
        mov = g["home_score"] - g["away_score"]
        if g["season"] in seasons:
            out.append({"game_id": g["game_id"], "p_home": p, "actual": mov, "margin": diff / 28.0})
        won = 1.0 if mov > 0 else 0.0
        mult = ((abs(mov) + 3) ** 0.8) / (7.5 + 0.006 * (diff if mov > 0 else -diff))
        delta = k * mult * (won - p)
        r[g["home"]], r[g["away"]] = rh + delta, ra - delta
    return out


def score_probs(rows: list[dict]) -> dict:
    p = np.clip(np.array([x["p_home"] for x in rows]), 1e-4, 1 - 1e-4)
    won = np.array([1.0 if x["actual"] > 0 else 0.0 for x in rows])
    return {
        "games": len(rows),
        "accuracy": float(((p > 0.5) == (won > 0.5)).mean()),
        "brier": float(((p - won) ** 2).mean()),
        "log_loss": float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean()),
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    games, pg = data.games(), data.player_games()
    played = lineups_played(pg)

    grid = {
        "half_life": [60.0, 120.0, 240.0],
        "lam_player": [3.0, 6.0, 12.0],
        "lam_team": [6.0, 20.0],
    }
    if "--quick" in argv:
        grid = {"half_life": [120.0], "lam_player": [6.0], "lam_team": [12.0]}

    results = []
    beta_cache: dict = {}
    for hl, lp, lt in itertools.product(*grid.values()):
        p = Params(half_life=hl, lam_player=lp, lam_team=lt)
        eng = Engine(games, pg, p)
        cache_key = (hl,)
        bc = beta_cache.setdefault(cache_key, {})
        tune = run_season(eng, 2025, played, beta_cache=bc)
        s = best_sigma(tune)
        sc = score(tune, s)
        results.append({"params": {"half_life": hl, "lam_player": lp, "lam_team": lt, "sigma": s}, "tune": sc})
        print(f"hl={hl:>5} lp={lp:>5} lt={lt:>5} sigma={s:.2f}  "
              f"acc={sc['accuracy']:.3f} brier={sc['brier']:.4f} ll={sc['log_loss']:.4f} mae={sc['margin_mae']:.2f}", flush=True)

    best = min(results, key=lambda r: r["tune"]["log_loss"])
    bp = best["params"]
    print("best on 2024-25:", bp, flush=True)

    p = Params(**bp)
    eng = Engine(games, pg, p)
    bc = beta_cache.get((bp["half_life"],), {})
    test = run_season(eng, 2026, played, beta_cache=bc)
    test_sc = score(test, p.sigma)
    test_reg = score([x for x in test if x["season_type"] == 2], p.sigma)
    cal = calibration(test, p.sigma)
    elo_test = score_probs(elo_preds(games, {2026}))
    home_only = float(np.mean([1.0 if x["actual"] > 0 else 0.0 for x in test]))

    out = {
        "tuned_on": "2024-25 (all games, walk-forward)",
        "tested_on": "2025-26 (never seen during tuning)",
        "grid": results,
        "chosen": bp,
        "box_beta": eng.p.box_beta,
        "test": test_sc,
        "test_regular_season": test_reg,
        "calibration": cal,
        "baselines": {"elo_538_style": elo_test, "always_home_team": {"accuracy": home_only}},
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("chosen", "test", "test_regular_season", "calibration", "baselines")}, indent=2))


if __name__ == "__main__":
    main()
