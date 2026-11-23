"""
Backfill the 2026-27 season so far: for every finished game, walk forward
(fit only on games before that day), predict knowing who played, and line
it up with the FanDuel closing chance and Prop Edge's own chance as Prop
Edge saved them. Gives the scorecard and the notebook a history on day one.
Kept separate from live predictions everywhere.

  python -m teleios.backfill
"""

from __future__ import annotations

import json

from teleios import data, pbp
from teleios.backtest import lineups_played, offseason_before
from teleios.daily import box_beta, load_params
from teleios.model import Engine

SEED = data.DATA / "seed" / "propedge_moneyline_2027.json"


def main():
    games, pg = data.games(), data.player_games()
    played = lineups_played(pg)
    eng = Engine(games, pg, load_params(), pbp.luck_by_game())
    try:
        seed = {g["game_id"]: g for g in json.loads(SEED.read_text())["games"]}
    except Exception:  # noqa: BLE001
        seed = {}
    import polars as pl
    g = games.filter((pl.col("season") == data.CURRENT_SEASON) & pl.col("completed")
                     & (pl.col("home_score") + pl.col("away_score") > 0))
    rows = []
    for day, day_games in g.group_by("date", maintain_order=True):
        day = day[0]
        eng.p.box_beta = box_beta(eng, day)
        st = eng.fit(day)
        a = day.toordinal()
        for r in day_games.iter_rows(named=True):
            lu = played.get(r["game_id"])
            if not lu or r["home"] not in lu or r["away"] not in lu:
                continue
            hl = eng.scale_to_team(eng.projected_minutes(lu[r["home"]], a))
            al = eng.scale_to_team(eng.projected_minutes(lu[r["away"]], a))
            pr = eng.predict(st, r["game_id"], r["home"], r["away"], hl, al)
            s = seed.get(r["game_id"], {})
            rows.append({
                "game_id": r["game_id"], "date": str(day), "home": r["home"], "away": r["away"],
                "p_home": round(pr["p_home"], 4), "pred_margin": round(pr["margin"], 2),
                "margin": r["home_score"] - r["away_score"],
                "rest_home": pr["rest_home"], "rest_away": pr["rest_away"],
                "market_p_home": s.get("fanduel_close_p_home"),
                "propedge_p_home": s.get("propedge_p_home"),
            })
    out = {"made_at": data.now_utc(), "season": "2026-27",
           "how": "walk-forward, lineup = players who actually played, minutes projected from earlier games",
           "games": rows}
    (data.DATA / "backfill_2027.json").write_text(json.dumps(out, indent=0))
    print(f"backfilled {len(rows)} games")


if __name__ == "__main__":
    main()
