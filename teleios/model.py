"""
The Teleios game model.

Predicts the HOME team's margin for any game, then turns it into a win
chance with a normal curve: P(home wins) = Phi(margin / sigma).

    margin = home court
           + situation   (rest, back-to-backs, travel, altitude)
           + team term   (coaching / system: what the players don't explain)
           + sum over home players of (minutes / 48) * player rating
           - the same for away players

Player ratings are fit on every past game's final margin with the minutes
each player actually played (a game-level, regularised plus-minus). Each
rating is pulled toward a box-score prior (what his points, rebounds,
assists, steals, blocks, turnovers and shooting usually mean for winning),
and older games count less (recency half-life). Because the team's strength
is the sum of the players who are actually going to play, injuries, rest
days and trades move the prediction automatically.

Every fit uses only games finished before the date being predicted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

from teleios import data, teams

BOX_STATS = ("pts", "reb", "oreb", "ast", "stl", "blk", "tov", "pf", "fga", "fta", "fg3m")


@dataclass
class Params:
    half_life: float = 120.0      # days: how fast old games fade
    window_days: int = 700        # games older than this are ignored
    lam_player: float = 6.0       # pull of each player rating toward its box prior
    lam_team: float = 12.0        # pull of each team term toward 0
    margin_cap: float = 30.0      # blowout margins are capped (garbage time)
    prior_minutes: float = 400.0  # box rates are shrunk toward bench average by this many minutes
    min_half_life: float = 5.0    # games: projected minutes recency
    sigma: float = 12.5           # spread of real margins around the prediction
    luck_weight: float = 0.0      # share of shot-making luck taken out of past margins (0 = off)
    box_beta: dict = field(default_factory=dict)  # per-48 box stat -> rating points

    def to_dict(self):
        return asdict(self)


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Situation: rest, travel, altitude for every team-game
# ---------------------------------------------------------------------------

def situation(games: pl.DataFrame) -> dict[int, dict]:
    """game_id -> situational features, from each team's previous game.
    Uses only the schedule, so it works for future games too."""
    rows = games.sort(["tip_utc", "game_id"]).to_dicts()
    last: dict[str, tuple] = {}  # team -> (date, venue)
    out: dict[int, dict] = {}
    for g in rows:
        v = data.venue(g)
        f = {}
        for side in ("home", "away"):
            t = g[side]
            if t in last:
                prev_date, prev_venue = last[t]
                rest = (g["date"] - prev_date).days
                travel = teams.miles(prev_venue, v)
            else:
                rest, travel = 5, 0.0
            f[f"rest_{side}"] = rest
            f[f"travel_{side}"] = travel
            last[t] = (g["date"], v)
        alt_city = (not g["neutral"]) and g["home"] in ("DEN", "UTA")
        out[g["game_id"]] = {
            "home_court": 0.0 if g["neutral"] else 1.0,
            "b2b": float(f["rest_home"] == 1) - float(f["rest_away"] == 1),
            "rested": float(f["rest_home"] >= 3) - float(f["rest_away"] >= 3),
            "travel": (f["travel_home"] - f["travel_away"]) / 1000.0,
            "altitude": 1.0 if alt_city else 0.0,
            **f,
        }
    return out


SIT_COLS = ("home_court", "b2b", "rested", "travel", "altitude")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, games: pl.DataFrame, pg: pl.DataFrame, params: Params, luck: dict | None = None):
        self.p = params
        self.luck = luck or {}
        self.games = games.sort(["tip_utc", "game_id"])
        self.sit = situation(self.games)
        done = self.games.filter(pl.col("completed") & (pl.col("home_score") + pl.col("away_score") > 0))
        self.done = done
        played = pg.filter(~pl.col("dnp") & (pl.col("min") > 0))
        self.pg = played
        self.pids = sorted(set(played["pid"].to_list()))
        self.pidx = {p: i for i, p in enumerate(self.pids)}
        self.names = dict(zip(played["pid"].to_list(), played["player"].to_list()))
        self.tidx = {t: i for i, t in enumerate(teams.CODES)}
        self.nS, self.nT, self.nP = len(SIT_COLS), len(teams.CODES), len(self.pids)
        self.nCol = self.nS + self.nT + self.nP

        # per-game sparse rows: team of each player-game, signed by home/away
        home_of = dict(zip(done["game_id"].to_list(), done["home"].to_list()))
        away_of = dict(zip(done["game_id"].to_list(), done["away"].to_list()))
        by_game: dict[int, list] = {}
        for gid, team, pid, mins in played.select(["game_id", "team", "pid", "min"]).iter_rows():
            if gid not in home_of:
                continue
            sgn = 1.0 if team == home_of[gid] else (-1.0 if team == away_of[gid] else 0.0)
            if sgn:
                by_game.setdefault(gid, []).append((self.pidx[pid], sgn * mins / 48.0))
        self.g_rows = []
        for g in done.iter_rows(named=True):
            gid = g["game_id"]
            if gid not in by_game:
                continue
            s = self.sit[gid]
            raw = g["home_score"] - g["away_score"]
            lk = self.luck.get(gid, {})
            if self.p.luck_weight and g["home"] in lk and g["away"] in lk:
                raw -= self.p.luck_weight * (lk[g["home"]] - lk[g["away"]])
            y = float(np.clip(raw, -self.p.margin_cap, self.p.margin_cap))
            self.g_rows.append({
                "gid": gid, "date": g["date"], "y": y,
                "sit": np.array([s[c] for c in SIT_COLS]),
                "home": self.tidx[g["home"]], "away": self.tidx[g["away"]],
                "players": by_game[gid],
            })
        self.g_dates = np.array([r["date"].toordinal() for r in self.g_rows])

        # player-game box table with ordinal dates, for priors and minutes
        self.pg_np = {
            "date": np.array([d.toordinal() for d in played["date"].to_list()]),
            "pid": np.array([self.pidx[p] for p in played["pid"].to_list()]),
            "min": played["min"].to_numpy(),
            **{s: played[s].to_numpy() for s in BOX_STATS},
        }

    # -- box prior ---------------------------------------------------------
    def box_rates(self, asof_ord: int) -> np.ndarray:
        """Recency-weighted per-48 box rates (nP x len(BOX_STATS)), shrunk
        toward the average of low-minute players."""
        d = self.pg_np
        m = (d["date"] < asof_ord) & (d["date"] >= asof_ord - self.p.window_days)
        w = 0.5 ** ((asof_ord - d["date"][m]) / self.p.half_life)
        pid = d["pid"][m]
        mins = np.bincount(pid, weights=w * d["min"][m], minlength=self.nP)
        tot = np.stack([np.bincount(pid, weights=w * d[s][m], minlength=self.nP) for s in BOX_STATS], 1)
        bench = mins < np.quantile(mins[mins > 0], 0.4) if (mins > 0).any() else np.ones(self.nP, bool)
        base = (tot[bench & (mins > 0)].sum(0) / max(mins[bench & (mins > 0)].sum(), 1.0)) * 48
        k = self.p.prior_minutes
        rates = (tot * 48 + k * base[None, :]) / (mins[:, None] + k)
        return rates, mins

    def prior(self, asof_ord: int) -> tuple[np.ndarray, np.ndarray]:
        rates, mins = self.box_rates(asof_ord)
        if not self.p.box_beta:
            return np.zeros(self.nP), mins
        beta = np.array([self.p.box_beta.get(s, 0.0) for s in BOX_STATS])
        return rates @ beta + self.p.box_beta.get("_const", 0.0), mins

    # -- fit -----------------------------------------------------------------
    def fit(self, asof) -> dict:
        """Ratings from games finished strictly before `asof` (a date)."""
        a = asof.toordinal()
        sel = np.where((self.g_dates < a) & (self.g_dates >= a - self.p.window_days))[0]
        n = len(sel)
        X = np.zeros((n, self.nCol))
        y = np.zeros(n)
        w = 0.5 ** ((a - self.g_dates[sel]) / self.p.half_life)
        for r, gi in enumerate(sel):
            g = self.g_rows[gi]
            X[r, : self.nS] = g["sit"]
            X[r, self.nS + g["home"]] += 1.0
            X[r, self.nS + g["away"]] -= 1.0
            for j, v in g["players"]:
                X[r, self.nS + self.nT + j] += v
            y[r] = g["y"]
        prior_p, mins = self.prior(a)
        prior = np.concatenate([np.zeros(self.nS + self.nT), prior_p])
        lam = np.concatenate([
            np.full(self.nS, 1e-3),
            np.full(self.nT, self.p.lam_team),
            np.full(self.nP, self.p.lam_player),
        ])
        yr = y - X @ prior
        Xw = X * w[:, None]
        A = X.T @ Xw + np.diag(lam)
        b = np.linalg.solve(A, Xw.T @ yr) + prior
        return {
            "asof": asof,
            "sit": dict(zip(SIT_COLS, b[: self.nS])),
            "team": dict(zip(teams.CODES, b[self.nS: self.nS + self.nT])),
            "player": b[self.nS + self.nT:],
            "prior": prior_p,
            "minutes_seen": mins,
            "n_games": n,
        }

    # -- minutes -------------------------------------------------------------
    def projected_minutes(self, pids: list[int], asof_ord: int) -> dict[int, float]:
        """Recency-weighted minutes over each player's last games played
        (half-life in games), before asof."""
        d = self.pg_np
        out = {}
        for pid in pids:
            j = self.pidx.get(pid)
            if j is None:
                out[pid] = 12.0
                continue
            m = (d["pid"] == j) & (d["date"] < asof_ord) & (d["date"] >= asof_ord - 365)
            if not m.any():
                out[pid] = 12.0
                continue
            order = np.argsort(d["date"][m])[::-1][:20]
            mins = d["min"][m][order]
            wts = 0.5 ** (np.arange(len(mins)) / self.p.min_half_life)
            out[pid] = float((mins * wts).sum() / wts.sum())
        return out

    @staticmethod
    def scale_to_team(mins: dict[int, float], total: float = 240.0) -> dict[int, float]:
        s = sum(mins.values())
        if s <= 0:
            return mins
        f = total / s
        out = {k: min(v * f, 42.0) for k, v in mins.items()}
        # redistribute anything clipped at 42
        left = total - sum(out.values())
        free = [k for k, v in out.items() if v < 42.0]
        if left > 0.5 and free:
            fs = sum(out[k] for k in free)
            for k in free:
                out[k] += left * out[k] / fs
        return out

    # -- predict -------------------------------------------------------------
    def team_strength(self, state: dict, team: str, lineup: dict[int, float]) -> tuple[float, list]:
        r = state["player"]
        parts = []
        tot = state["team"][team]
        for pid, mins in lineup.items():
            j = self.pidx.get(pid)
            val = r[j] if j is not None else float(np.quantile(state["prior"], 0.2))
            parts.append((pid, mins, val, mins / 48.0 * val))
            tot += mins / 48.0 * val
        return tot, sorted(parts, key=lambda x: -x[3])

    def predict(self, state: dict, gid: int, home: str, away: str,
                home_lineup: dict[int, float], away_lineup: dict[int, float]) -> dict:
        s = self.sit[gid]
        sit_val = {c: state["sit"][c] * s[c] for c in SIT_COLS}
        hs, hparts = self.team_strength(state, home, home_lineup)
        as_, aparts = self.team_strength(state, away, away_lineup)
        margin = sum(sit_val.values()) + hs - as_
        return {
            "margin": margin,
            "p_home": phi(margin / self.p.sigma),
            "situation": sit_val,
            "home_strength": hs, "away_strength": as_,
            "home_parts": hparts, "away_parts": aparts,
            "rest_home": s["rest_home"], "rest_away": s["rest_away"],
        }


def learn_box_beta(engine: Engine, asof, ridge: float = 1.0) -> dict:
    """Fit ratings with no prior, then learn how per-48 box rates map to
    those ratings (minutes-weighted). Gives the prior used afterwards."""
    saved = engine.p.box_beta
    engine.p.box_beta = {}
    st = engine.fit(asof)
    engine.p.box_beta = saved
    rates, mins = engine.box_rates(asof.toordinal())
    keep = mins > 300
    Xb = np.column_stack([rates[keep], np.ones(keep.sum())])
    wt = mins[keep]
    yb = st["player"][keep]
    A = Xb.T @ (Xb * wt[:, None]) + ridge * np.eye(Xb.shape[1])
    beta = np.linalg.solve(A, Xb.T @ (yb * wt))
    out = {s: float(b) for s, b in zip(BOX_STATS, beta[:-1])}
    out["_const"] = float(beta[-1])
    return out
