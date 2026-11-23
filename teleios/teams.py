"""
One canonical team code per franchise (the NBA's own three-letter codes),
plus how each data source spells them, and where each team plays.

Arena coordinates are used for travel distance; altitude for the two
mountain arenas (Denver ~5,280 ft, Salt Lake City ~4,300 ft).
"""

from __future__ import annotations

import math

# code: (full name, latitude, longitude, altitude in feet)
TEAMS = {
    "ATL": ("Atlanta Hawks", 33.757, -84.396, 1050),
    "BKN": ("Brooklyn Nets", 40.683, -73.975, 30),
    "BOS": ("Boston Celtics", 42.366, -71.062, 20),
    "CHA": ("Charlotte Hornets", 35.225, -80.839, 750),
    "CHI": ("Chicago Bulls", 41.881, -87.674, 600),
    "CLE": ("Cleveland Cavaliers", 41.497, -81.688, 650),
    "DAL": ("Dallas Mavericks", 32.790, -96.810, 430),
    "DEN": ("Denver Nuggets", 39.749, -105.008, 5280),
    "DET": ("Detroit Pistons", 42.341, -83.055, 600),
    "GSW": ("Golden State Warriors", 37.768, -122.388, 10),
    "HOU": ("Houston Rockets", 29.751, -95.362, 50),
    "IND": ("Indiana Pacers", 39.764, -86.156, 715),
    "LAC": ("LA Clippers", 33.945, -118.343, 100),
    "LAL": ("Los Angeles Lakers", 34.043, -118.267, 300),
    "MEM": ("Memphis Grizzlies", 35.138, -90.051, 250),
    "MIA": ("Miami Heat", 25.781, -80.188, 10),
    "MIL": ("Milwaukee Bucks", 43.045, -87.917, 600),
    "MIN": ("Minnesota Timberwolves", 44.979, -93.276, 830),
    "NOP": ("New Orleans Pelicans", 29.949, -90.082, 10),
    "NYK": ("New York Knicks", 40.751, -73.993, 30),
    "OKC": ("Oklahoma City Thunder", 35.463, -97.515, 1200),
    "ORL": ("Orlando Magic", 28.539, -81.384, 100),
    "PHI": ("Philadelphia 76ers", 39.901, -75.172, 30),
    "PHX": ("Phoenix Suns", 33.446, -112.071, 1090),
    "POR": ("Portland Trail Blazers", 45.532, -122.667, 50),
    "SAC": ("Sacramento Kings", 38.580, -121.500, 30),
    "SAS": ("San Antonio Spurs", 29.427, -98.438, 650),
    "TOR": ("Toronto Raptors", 43.643, -79.379, 250),
    "UTA": ("Utah Jazz", 40.768, -111.901, 4300),
    "WAS": ("Washington Wizards", 38.898, -77.021, 50),
}
CODES = tuple(TEAMS)

# Neutral-site cities seen in ESPN schedules (NBA Cup finals, global games).
NEUTRAL_CITIES = {
    "Las Vegas": (36.103, -115.178),
    "Mexico City": (19.404, -99.096),
    "London": (51.508, 0.003),
    "Paris": (48.839, 2.379),
    "Berlin": (52.507, 13.444),
    "Abu Dhabi": (24.460, 54.600),
    "Manchester": (53.483, -2.200),
}

ALIASES = {
    "espn": {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"},
    "kalshi": {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS",
               "BRK": "BKN", "PHO": "PHX", "CHO": "CHA"},
}

_unmatched: dict[str, set] = {}


def canon(name, source: str = "espn"):
    """Canonical code for a source's spelling, or None (and remembered)."""
    if not name:
        return None
    raw = str(name).strip().upper()
    code = ALIASES.get(source, {}).get(raw, raw)
    if code in TEAMS:
        return code
    _unmatched.setdefault(source, set()).add(raw)
    return None


def unmatched() -> dict:
    return {k: sorted(v) for k, v in _unmatched.items()}


def location(code: str) -> tuple[float, float]:
    _n, lat, lon, _alt = TEAMS[code]
    return lat, lon


def miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in miles."""
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))
