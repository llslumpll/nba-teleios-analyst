"""
Walk-forward backtest of the player model.

Tuned on 2024-25 (coordinate search, one setting at a time), tested
unchanged on 2025-26. Sample: regular-season player-games where the model
projected 20+ minutes (the same sample Prop Edge's backtest reports).
The backtest knows who played (as a final injury report would), not how
many minutes. Blowout risk uses the game model's own walk-forward margins.

  python -m teleios.players_backtest
"""

from __future__ import annotations

import copy
import json
import sys

import numpy as np

from teleios import data
from teleios.players import COMBOS, STATS, PlayerModel, PParams, calibrate_scale, coverage, fit_var

OUT = data.DATA / "players_backtest.json"
PARAMS_OUT = data.DATA / "player_params.json"
REPORT = ("pts", "reb", "ast", "fg3m", "pra")
PROP_EDGE = {"pts": 5.10, "reb": 2.00, "ast": 1.54, "fg3m": 1.04, "pra": 6.42}  # Prop Edge README, 2025-26


def with_combos(mu: np.ndarray, y: np.ndarray):
    cols = list(STATS)
    mu2, y2 = [mu], [y]
    for name, parts in COMBOS.items():
        idx = [cols.index(x) for x in parts]
        mu2.append(mu[:, idx].sum(1, keepdims=True))
        y2.append(y[:, idx].sum(1, keepdims=True))
        cols.append(name)
    return np.hstack(mu2), np.hstack(y2), cols


def baselines(m: PlayerModel):
    """Season average to date and last-10 average, per stat."""
    n = m.n
    season_avg = np.full((n, len(STATS)), np.nan)
    last10 = np.full((n, len(STATS)), np.nan)
    hist: dict = {}
    for i in range(n):
        k = (m.pid[i], m.season[i])
        h = hist.get(k)
        if h:
            arr = np.array(h)
            season_avg[i] = arr.mean(0)
            last10[i] = arr[-10:].mean(0)
        hist.setdefault(k, []).append(m.Y[i])
    return season_avg, last10


def evaluate(m: PlayerModel, margins: dict, season: int, extra: dict | None = None) -> dict:
    pr = m.project_rows(margins=margins)
    sel = (m.season == season) & (m.df["season_type"].to_numpy() == 2) & (pr["min"] >= 20.0)
    mu, y, cols = with_combos(pr["mu"][sel], m.Y[sel])
    out = {"rows": int(sel.sum())}
    for j, c in enumerate(cols):
        out[c] = float(np.abs(mu[:, j] - y[:, j]).mean())
    if extra is not None:
        extra.update(mu=mu, y=y, cols=cols, sel=sel, pr=pr)
    return out


def objective(r: dict, stats=REPORT) -> float:
    return sum(r[s] / PROP_EDGE[s] for s in stats)


def main(argv=None):
    argv = argv or sys.argv[1:]
    games, pg = data.games(), data.player_games()
    margins = json.loads((data.DATA / "game_margins_walkforward.json").read_text())
    start = PParams()
    if "--from-saved" in argv and PARAMS_OUT.exists():
        start = PParams(**json.loads(PARAMS_OUT.read_text()))
    m = PlayerModel(games, pg, start)
    p = m.p
    best = evaluate(m, margins, 2025)
    print("start", {k: round(v, 3) for k, v in best.items()}, flush=True)

    def trial(label, setter, values, only=None):
        nonlocal best
        obj = (lambda r: r[only]) if only else objective
        for v in values:
            saved = copy.deepcopy(m.p)
            setter(v)
            r = evaluate(m, margins, 2025)
            if obj(r) < obj(best) - 1e-5 and (only is None or objective(r) <= objective(best) + 1e-4):
                best = r
                print(f"  {label}={v} -> {objective(r):.4f}", {k: round(x, 3) for k, x in r.items() if k != 'rows'}, flush=True)
            else:
                m.p = saved

    rounds = 0 if "--from-saved" in argv else (1 if "--quick" in argv else 2)
    for rnd in range(rounds):
        print(f"round {rnd + 1}", flush=True)
        trial("min_half_life", lambda v: setattr(m.p, "min_half_life", v), [2.0, 3.0, 4.0, 6.0, 8.0])
        trial("team_pull", lambda v: setattr(m.p, "team_pull", v), [0.3, 0.5, 0.7, 0.9, 1.0])
        trial("season_carry", lambda v: setattr(m.p, "season_carry", v), [0.0, 0.15, 0.3, 0.5, 0.9])
        trial("blowout_cut", lambda v: setattr(m.p, "blowout_cut", v), [0.0, 0.08, 0.15, 0.25, 0.35])
        trial("pace_beta", lambda v: setattr(m.p, "pace_beta", v), [0.0, 0.5, 1.0])
        trial("def_beta", lambda v: setattr(m.p, "def_beta", v), [0.0, 0.3, 0.6, 0.8, 1.0])
        trial("team_half_life", lambda v: setattr(m.p, "team_half_life", v), [6.0, 12.0, 25.0])
        for s in STATS:
            trial(f"half_life[{s}]", lambda v, s=s: m.p.rate_half_life.__setitem__(s, v), [10.0, 25.0, 40.0, 60.0, 90.0, 140.0, 220.0], only=s)
            trial(f"k[{s}]", lambda v, s=s: m.p.shrink_k.__setitem__(s, v), [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0], only=s)

    # ranges, fit on the tuning season
    ex: dict = {}
    evaluate(m, margins, 2025, ex)
    for j, c in enumerate(ex["cols"]):
        stat = c if c in STATS else "pts"
        v = fit_var(ex["mu"][:, j], ex["y"][:, j])
        m.p.var[c] = v + [calibrate_scale(stat, ex["mu"][:, j], ex["y"][:, j], v)]

    # test, untouched
    ex = {}
    test = evaluate(m, margins, 2026, ex)
    cover = {}
    for j, c in enumerate(ex["cols"]):
        stat = c if c in STATS else "pts"
        for q in (0.5, 0.8):
            cover.setdefault(c, {})[f"in_{int(q*100)}pct_range"] = round(coverage(stat, ex["mu"][:, j], ex["y"][:, j], m.p.var[c], q), 3)
    sa, l10 = baselines(m)
    sel = ex["sel"]
    base = {}
    for name, arr in (("season_avg", sa), ("last10", l10)):
        a = arr[sel]
        ok = ~np.isnan(a).any(1)
        mu_b, y_b, cols = with_combos(a[ok], m.Y[sel][ok])
        mu_t, _y, _c = with_combos(ex["mu"][ok], m.Y[sel][ok])
        base[name] = {c: round(float(np.abs(mu_b[:, j] - y_b[:, j]).mean()), 3) for j, c in enumerate(cols)}
        base[f"teleios_same_rows_vs_{name}"] = {c: round(float(np.abs(mu_t[:, j] - y_b[:, j]).mean()), 3) for j, c in enumerate(cols)}
    out = {
        "tuned_on": "2024-25 regular season, projected 20+ min",
        "tested_on": "2025-26 regular season, projected 20+ min (never seen in tuning)",
        "params": m.p.to_dict(),
        "test_avg_miss": {k: (round(v, 3) if k != "rows" else v) for k, v in test.items()},
        "prop_edge_published": PROP_EDGE,
        "baselines": base,
        "ranges": cover,
    }
    OUT.write_text(json.dumps(out, indent=1))
    PARAMS_OUT.write_text(json.dumps(m.p.to_dict(), indent=1))
    print(json.dumps({k: out[k] for k in ("test_avg_miss", "prop_edge_published", "baselines", "ranges")}, indent=1))


if __name__ == "__main__":
    main()
