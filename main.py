"""Refresh-aware earners/players forecast for make.fun gta6-trophy-rarity-forecast-01, v2.

What v1 got wrong, and what this program does instead
------------------------------------------------------
An Exophase earn rate is a ratio: earners over players tracked. The two sides
move on different clocks. The tracked pool can multiply overnight when the page
refreshes into a launch crowd, while the earner counts of the old pool barely
move -- so a rate can collapse by a factor of five with its earners *rising*.
v1 saw the shape of this but priced it timidly: on a page that had never
changed it shifted the median only -0.62 logit (the launch-night refresh moved
rates about -1.66 logit) and then charged a sigma above 1.0 for the uncertainty,
paying pinball at every quantile of every trophy.

This program models the ratio and nothing else:

- It recovers earner counts from each distinct page state (rate x players --
  on a settled page these are integers to within the page's 2-decimal
  rounding) and trends log(earners) and log(players) separately against
  log-time, per trophy, with Theil-Sen fits shrunk toward the pool-wide
  medians. The forecast is the ratio of the two extrapolations.
- A history with a single distinct state is a stale page, not a stable one:
  the next refresh replaces a pre-launch-scale pool with the launch crowd.
  The stale branch prices that event directly -- the pool jumps by
  POOL_JUMP x while earners grow only by EARN_GROWTH_STALE x over the first
  refresh (both calibrated on the launch-night refresh the kit's rules
  narrate: 638 tracked players becoming 3,868, with earners nearly frozen),
  and the spread stays tight (SIGMA_STALE), because the event is a regime
  change with a known shape, not a random walk.
- A saturated trophy (rate at the page's ceiling) is held at the ceiling with
  a narrow spread: a trophy every tracked player holds is a fixture of the
  page, and it stayed at 100% through the launch refresh.
- The spread everywhere else is built from the fit's own residuals, slope
  disagreement and a sqrt(horizon) random-walk term, floored low: an honest
  spread around a well-placed median, never a wide one around a guess.

Pure Python 3.13 standard library, offline, deterministic. Reads every
<id>.in in the input directory and writes <id>.out to the output directory,
per the kit contract: {"trophies": [{"id": n, "q": [q05..q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- Stale-page (single distinct state) refresh prior -----------------------
# The tracked pool at the first post-launch refresh, as a multiple of the
# frozen pool (the kit's rules narrate 638 -> 3,868 players, a 6.06x jump).
POOL_JUMP = 6.06
# Earners over that same refresh: the old pool's counts nearly freeze while
# the denominator explodes; a thin layer of new earners arrives with the
# crowd. Calibrated so earners grow about 1.15-1.20x across the jump.
EARN_GROWTH_STALE = 1.175
# Spread charged on the stale branch, in logit units: the refresh's shape is
# known, so this prices only how far the real refresh lands from the prior.
SIGMA_STALE = 0.26

# --- Rich-history trend model ------------------------------------------------
SIGMA_PER_SQRT_DAY = 0.30          # random-walk part of the spread
REFRESH_FLOOR_FEW = 0.30           # spread floor when <= 3 refreshes seen
REFRESH_FLOOR_MANY = 0.16          # spread floor once refreshes are routine
SIGMA_MIN = 0.14
SIGMA_MAX = 0.95
MAX_SLOPE = 3.0                    # per unit of log-time
MAX_FIT_POINTS = 44                # Theil-Sen is quadratic; thin beyond this


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, matching JavaScript's Math.round."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> float:
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def theil_sen(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Robust line fit: (slope, intercept, slope IQR) from pairwise slopes."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return 0.0, ys[0], 0.0
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx > 1e-9:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return 0.0, ys[-1], 0.0
    slope = median(slopes)
    ordered = sorted(slopes)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(3 * len(ordered)) // 4]
    intercept = median([y - slope * x for x, y in zip(xs, ys)])
    return slope, intercept, max(0.0, q3 - q1)


def thin(points: list[tuple[float, float]], cap: int) -> list[tuple[float, float]]:
    """Evenly thin a series to at most `cap` points, always keeping the last."""
    if len(points) <= cap:
        return points
    step = (len(points) - 1) / (cap - 1)
    picked = [points[round(k * step)] for k in range(cap - 1)]
    picked.append(points[-1])
    out = []
    last_x = None
    for x, y in picked:
        if last_x is None or x > last_x + 1e-12:
            out.append((x, y))
            last_x = x
    return out


def distinct_states(history: list[dict]) -> list[int]:
    """Indices of reads where the page actually changed.

    Exophase refreshes on its own schedule; consecutive reads of an unchanged
    page are the recorder's clock, not information. A read is a new state when
    its players count or any rate differs from the previous kept read.
    """
    idx = []
    for i, read in enumerate(history):
        if idx:
            prev = history[idx[-1]]
            if prev.get("players_tracked") == read.get("players_tracked") and \
                    prev.get("rates") == read.get("rates"):
                continue
        idx.append(i)
    return idx


def shrink(slope: float, prior: float, n_obs: int) -> float:
    """Shrink a per-trophy slope toward the pool-wide prior by evidence."""
    w = n_obs / (n_obs + 5.0)
    return max(-MAX_SLOPE, min(MAX_SLOPE, w * slope + (1.0 - w) * prior))


def quantile_row(median_rate: float, sigma: float) -> list:
    centre = logit(median_rate)
    qs = [round4(inv_logit(centre + z * sigma)) for z in Z]
    return sorted(qs)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0

    times = [parse_time(r["slot"]) for r in history]
    hours = [(t - times[0]) / 3600.0 for t in times]
    xs_all = [math.log1p(h) for h in hours]
    x_target = math.log1p(hours[-1] + case["horizon_minutes"] / 60.0)
    dx = max(0.0, x_target - xs_all[-1])

    state_idx = distinct_states(history)
    states = [history[i] for i in state_idx]
    state_xs = [xs_all[i] for i in state_idx]
    n_states = len(states)

    last = states[-1]
    players_last = last.get("players_tracked") or 0
    if not players_last:
        for read in reversed(history):
            if read.get("players_tracked"):
                players_last = read["players_tracked"]
                break

    def last_rate(i: int):
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                return rates[i]
        return None

    rows = []

    # ------------------------------------------------------------------
    # Stale page: one distinct state. The next refresh is a regime change
    # with a known shape -- price the event, not a random walk around the
    # frozen rates.
    # ------------------------------------------------------------------
    if n_states <= 1:
        for i, trophy in enumerate(trophies):
            r_last = last_rate(i)
            if r_last is None:
                r_last = 1.0
            if r_last >= 99.9 or not players_last:
                rows.append({"id": trophy["id"],
                             "q": quantile_row(r_last, 0.05)})
                continue
            earners = r_last / 100.0 * players_last
            e_hat = earners * EARN_GROWTH_STALE
            p_hat = players_last * POOL_JUMP
            # Past the first refresh the two sides keep moving on their own
            # clocks: the pool keeps compounding faster than the earner
            # counts, so a long horizon dilutes further after the jump.
            extra_days = max(0.0, horizon_days - 0.0625)
            if extra_days > 0.0:
                p_hat *= (1.0 + extra_days) ** 0.9
                e_hat *= (1.0 + extra_days) ** 0.55
            med = 100.0 * e_hat / p_hat
            sigma = min(0.60, SIGMA_STALE + 0.10 * math.sqrt(extra_days))
            rows.append({"id": trophy["id"],
                         "q": quantile_row(med, sigma)})
        return {"trophies": rows}

    # ------------------------------------------------------------------
    # Rich history: trend earners and players separately over the distinct
    # refresh states, then divide. No direct rate trend is blended in: the
    # ratio is the model, because the two sides move on different clocks.
    # ------------------------------------------------------------------
    p_points = [(x, math.log(r["players_tracked"]))
                for x, r in zip(state_xs, states) if r.get("players_tracked")]
    if len(p_points) >= 2 and players_last > 0:
        p_fit = thin(p_points, MAX_FIT_POINTS)
        p_slope, _, _ = theil_sen([p[0] for p in p_fit], [p[1] for p in p_fit])
        p_slope = max(0.0, min(MAX_SLOPE, p_slope))
    else:
        p_slope = 0.0
    log_p_hat = math.log(players_last) + p_slope * dx if players_last > 0 else 0.0
    if players_last > 0:
        log_p_hat = min(log_p_hat, math.log(players_last) + math.log(500.0))
        log_p_hat = max(log_p_hat, math.log(players_last * 0.98))
    p_hat = math.exp(log_p_hat) if players_last > 0 else 0.0

    raw = []
    for i in range(len(trophies)):
        pts = []
        last_r = None
        for x, read in zip(state_xs, states):
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            players = read.get("players_tracked")
            if rate is None or not players:
                continue
            last_r = rate
            pts.append((x, math.log(rate / 100.0 * players + 0.5)))
        if last_r is None:
            last_r = last_rate(i)
        fit = thin(pts, MAX_FIT_POINTS)
        if len(fit) >= 2:
            slope, intercept, slope_iqr = theil_sen([p[0] for p in fit],
                                                    [p[1] for p in fit])
            resid = [y - (intercept + slope * x) for x, y in fit]
            resid_sd = math.sqrt(sum(r * r for r in resid) / len(resid))
        else:
            slope, slope_iqr, resid_sd = 0.0, 0.0, 0.0
        raw.append({"last_rate": last_r, "slope": slope,
                    "slope_iqr": slope_iqr, "resid_sd": resid_sd,
                    "n": len(pts)})
    fitted = [r for r in raw if r["n"] >= 2]
    prior_slope = median([r["slope"] for r in fitted]) if fitted else 0.0
    all_last = [r["last_rate"] for r in raw if r["last_rate"] is not None]
    fallback_rate = median(all_last) if all_last else 1.0
    refresh_floor = REFRESH_FLOOR_FEW if n_states <= 3 else REFRESH_FLOOR_MANY

    for i, trophy in enumerate(trophies):
        info = raw[i]
        r_last = info["last_rate"] if info["last_rate"] is not None else fallback_rate
        if r_last >= 99.9:
            rows.append({"id": trophy["id"], "q": quantile_row(r_last, 0.06)})
            continue
        if players_last > 0 and p_hat > 0 and info["n"] >= 1:
            e_slope = shrink(info["slope"], prior_slope, info["n"])
            e_last = r_last / 100.0 * players_last
            e_hat = (e_last + 0.5) * math.exp(max(-1.0, min(3.0, e_slope)) * dx) - 0.5
            e_hat = max(e_hat, e_last * 0.95)      # counts essentially never fall
            e_hat = min(e_hat, p_hat * 0.999)      # nor exceed the pool
            med = 100.0 * e_hat / p_hat if e_hat > 0 else r_last
            # Guard the extrapolation: adoption bends, so the median's logit
            # may not run further than this from the last observed rate,
            # however straight the fitted log-time line looks.
            med = inv_logit(max(logit(r_last) - 3.5,
                                min(logit(r_last) + 2.0, logit(med))))
        else:
            med = r_last
        slope_unc = min(0.4, 0.35 * info["slope_iqr"] * dx)
        resid_term = min(0.45, 0.6 * info["resid_sd"])
        sigma = math.sqrt((SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                          + resid_term ** 2 + slope_unc ** 2
                          + refresh_floor ** 2)
        sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
        rows.append({"id": trophy["id"], "q": quantile_row(med, sigma)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
