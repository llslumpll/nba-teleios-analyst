"""
Monthly retune: Teleios re-fits its own settings on recent games and adopts
new ones ONLY if they beat the current ones on games the search never saw.

  held-out games  the latest 30% of this season's game days so far
  game model      candidates around the current settings (recency half-life,
                  player pull, team pull), each scored walk-forward on the
                  held-out days; adopt if log loss improves by >= 0.002
  player model    coordinate search on last season + this season's first
                  70%; adopt if the average miss on points, rebounds,
                  assists, threes and PRA (relative to current) improves by
                  >= 0.5% and no one of them gets more than 1% worse

Skips itself until there are 300 held-out games. Every run, adopted or not,
is logged in data/retune_log.json with the old settings, so it can be undone.

  python -m teleios.retune
"""

from __future__ import annotations

import copy
import itertools
import json
from datetime import date

import numpy as np
import polars as pl

from teleios import data, pbp
from teleios.backtest import lineups_played, run_season, score
from teleios.daily import PARAMS_PATH, load_params
from teleios.model import Engine, Params
from teleios.players import STATS, PlayerModel, PParams
from teleios.players_backtest import PARAMS_OUT as PLAYER_PARAMS, with_combos

LOG = data.DATA / "retune_log.json"
MIN_HELD_GAMES = 300
REPORT = ("pts", "reb", "ast", "fg3m", "pra")


def held_out_start(games: pl.DataFrame) -> date | None:
    days = sorted(set(games.filter((pl.col("season") == data.CURRENT_SEASON) & pl.col("completed")
                                   & (pl.col("season_type") == 2))["date"].to_list()))
    if len(days) < 20:
        return None
    return days[int(len(days) * 0.7)]


def retune_games(games, pg, cut: date) -> dict:
    played = lineups_played(pg)
    cur = load_params()
    luck = pbp.luck_by_game()

    def run(p: Params):
        e = Engine(games, pg, p, luck)
        preds = run_season(e, data.CURRENT_SEASON, played, season_types=(2,), since=cut)
        return preds

    base_preds = run(cur)
    if len(base_preds) < MIN_HELD_GAMES:
        return {"state": "skipped", "why": f"only {len(base_preds)} held-out games (need {MIN_HELD_GAMES})"}
    base = score(base_preds, cur.sigma)
    best = (base["log_loss"], cur, base)
    for f_hl, f_lp, f_lt in itertools.product((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)):
        if (f_hl, f_lp, f_lt) == (1.0, 1.0, 1.0):
            continue
        p = copy.deepcopy(cur)
        p.half_life, p.lam_player, p.lam_team = cur.half_life * f_hl, cur.lam_player * f_lp, cur.lam_team * f_lt
        preds = run(p)
        # sigma refit on the same games would leak; keep the current sigma
        sc = score(preds, p.sigma)
        if sc["log_loss"] < best[0]:
            best = (sc["log_loss"], p, sc)
    adopt = best[0] <= base["log_loss"] - 0.002
    out = {"state": "adopted" if adopt else "kept", "held_out_from": str(cut), "current": base, "best_candidate": best[2],
           "old_settings": cur.to_dict(), "new_settings": best[1].to_dict() if adopt else None}
    if adopt:
        d = best[1].to_dict()
        d.pop("box_beta", None)
        PARAMS_PATH.write_text(json.dumps({**d, "chosen_by": f"monthly retune {date.today()}"}, indent=1))
    return out


def retune_players(games, pg, cut: date) -> dict:
    margins = json.loads((data.DATA / "game_margins_walkforward.json").read_text()) if (data.DATA / "game_margins_walkforward.json").exists() else {}
    cur = PParams(**json.loads(PLAYER_PARAMS.read_text())) if PLAYER_PARAMS.exists() else PParams()
    m = PlayerModel(games, pg, copy.deepcopy(cur))
    dates = m.df["date"].to_numpy()
    st = m.df["season_type"].to_numpy()

    def ev(mask_fn):
        pr = m.project_rows(margins=margins)
        sel = mask_fn() & (st == 2) & (pr["min"] >= 20.0)
        mu, y, cols = with_combos(pr["mu"][sel], m.Y[sel])
        return {c: float(np.abs(mu[:, j] - y[:, j]).mean()) for j, c in enumerate(cols)} | {"rows": int(sel.sum())}

    cut64 = np.datetime64(cut)
    train = lambda: ((m.season == data.CURRENT_SEASON - 1) | ((m.season == data.CURRENT_SEASON) & (dates < cut64)))
    held = lambda: (m.season == data.CURRENT_SEASON) & (dates >= cut64)
    base_held = ev(held)
    if base_held["rows"] < 1500:
        return {"state": "skipped", "why": f"only {base_held['rows']} held-out player-games (need 1500)"}
    best_train = ev(train)
    obj = lambda r: sum(r[s] / best_train_ref[s] for s in REPORT)
    best_train_ref = dict(best_train)

    def trial(setter, values, only=None):
        nonlocal best_train
        for v in values:
            saved = copy.deepcopy(m.p)
            setter(v)
            r = ev(train)
            better = (r[only] < best_train[only] - 1e-5) if only else (obj(r) < obj(best_train) - 1e-5)
            if better:
                best_train = r
            else:
                m.p = saved

    trial(lambda v: setattr(m.p, "min_half_life", v), [m.p.min_half_life * f for f in (0.67, 1.5)])
    trial(lambda v: setattr(m.p, "team_pull", v), [max(0.1, m.p.team_pull - 0.2), min(1.0, m.p.team_pull + 0.2)])
    trial(lambda v: setattr(m.p, "def_beta", v), [max(0.0, m.p.def_beta - 0.3), m.p.def_beta + 0.3])
    for s in STATS:
        trial(lambda v, s=s: m.p.rate_half_life.__setitem__(s, v), [m.p.rate_half_life[s] * f for f in (0.6, 1.6)], only=s if s in REPORT else None)
    new_held = ev(held)
    gains = {s: (base_held[s] - new_held[s]) / base_held[s] for s in REPORT}
    avg_gain = float(np.mean(list(gains.values())))
    adopt = avg_gain >= 0.005 and min(gains.values()) >= -0.01
    out = {"state": "adopted" if adopt else "kept", "held_out_from": str(cut), "held_out_rows": base_held["rows"],
           "current_avg_miss": {s: round(base_held[s], 3) for s in REPORT},
           "candidate_avg_miss": {s: round(new_held[s], 3) for s in REPORT},
           "old_settings": cur.to_dict(), "new_settings": m.p.to_dict() if adopt else None}
    if adopt:
        PLAYER_PARAMS.write_text(json.dumps(m.p.to_dict(), indent=1))
    return out


def main():
    games, pg = data.games(), data.player_games()
    cut = held_out_start(games)
    rec = {"run_at": data.now_utc()}
    if cut is None:
        rec["state"] = "skipped: fewer than 20 game days this season"
    else:
        rec["games"] = retune_games(games, pg, cut)
        rec["players"] = retune_players(games, pg, cut)
    try:
        log = json.loads(LOG.read_text())
    except Exception:  # noqa: BLE001
        log = []
    LOG.write_text(json.dumps(log + [rec], indent=1, default=str))
    print(json.dumps(rec, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
