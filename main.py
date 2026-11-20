"""GTA VI trophy rarity forecaster v5 (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PS5, its Exophase earn
rate at the case's target, as five quantiles (5/25/50/75/95%).

Method (original, written for this competition):
  A trophy's rate is earners / players_tracked, and the two move for
  different reasons, so they are forecast separately and divided:
  - players_tracked P(t): power law on all reads, power law on the last
    24h, and a logistic in log-time (k=1), combined by a blend that trusts
    the power law when observed growth is moderate and the saturating
    logistic when it is steep. Reads before 0.5h after release are
    excluded from the players fit only; the share fit keeps them with
    the time floored at 0.5h, and the earner fits keep all states
    (the launch point anchors their slope).
    P(target) is never below the last observed count (players tracked
    is cumulative), and no trophy's predicted earners fall below the
    last reliably observed count.
  - earners E_i(t) per trophy: log-log fit of the earner counts
    reconstructed from each read (E = rate * P / 100), with the slope
    shrunk toward the cross-trophy median slope and the intercept
    re-fitted for the blended slope. Points whose rate is at the
    2-decimal rounding floor (rate < 0.03%) are dropped from the fit:
    there the reconstructed earner count is dominated by rounding
    (it grows like P, not like the true earners).
  - per trophy, the MEDIAN rate is either:
      * the logistic share s(t) = S / (1 + (t50/t)^k) fitted (on the logit
        scale) to the trophy's observed rates, when its share has been
        rising strongly (players keep reaching that content faster than
        new players join, and the share saturates); or
      * E_i(target) / P(target) otherwise (dilution dominates, or the
        share's rise is weak/ambiguous so the counts decomposition is
        the honest route).
  - spread: honest and wide where the fits disagree or the history is
    thin, narrow where the trophy's counts are stable and well observed.
  - flat history (every read identical, as in the practice cases):
    Exophase refreshes its page in jumps; the documented launch-night
    refresh took players 638 -> 3868 with earners unchanged, so one
    such refresh is assumed before the target, earners conserved, the
    100% trophy kept at 100%, with a narrow spread.

Standard library only. Deterministic: no randomness, no network, no clock
reads (all times come from the case file).
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# One documented launch-night refresh: players 638 -> 3868, earners
# unchanged (RULES.txt). Used only when the case history is completely flat.
REFRESH_JUMP = 3868.0 / 638.0
FLAT_SIGMA = 0.03
# Rates below this are rounding-dominated (2-decimal rounding is +-0.005
# percentage points, i.e. >= ~17% relative error) and are dropped from fits.
RATE_FLOOR = 0.03
# Tunable model settings (defaults are the production values).
SETTINGS = {
    # Share slope (logit rate vs ln(1+hours)) above which the logistic-share
    # median is trusted over the counts decomposition.
    "share_slope_min": 0.5,
    # Earner-fit anchor: "last" (last reliable point) or "refit" (OLS
    # intercept re-fitted for the blended slope).
    "anchor": "last",
    # Extra spread added per unit of logit disagreement between the
    # share-fit and counts medians (0 = off).
    "widen": 0.0,
    # Rates below this are dropped from the earner/share fits (rounding
    # floor). 0.0 = keep everything.
    "rate_floor": 0.03,
    # Slope shrinkage: "fixed4" (w = n/(n+4)) or "adaptive" (k=2 if the fit
    # is clean, else k=6).
    "shrink": "fixed4",
    # Cap on the P-model log-spread fed into the per-trophy sigma.
    "p_spread_cap": 1.0,
    # Saturation signature: share rose this much from its minimum and is
    # now flat -> trust the logistic-share median.
    "share_rise_min": 1.5,
    "share_flat_slope": 0.3,
    # Implausible E/P rise: counts median implies the share rising this
    # much over the last rate while the share is flat and above its first
    # value -> the earner growth was P-driven, trust the share median.
    "ep_rise_max": 3.0,
    "ep_flat_slope": 0.15,
    # Multiplier on every per-trophy logit spread (calibration).
    "sigma_scale": 0.85,
}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_std, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        resid = math.sqrt(sum((y - my) ** 2 for y in ys) / n) if n else 0.0
        return my, 0.0, resid, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid, n


def parse_time(s):
    """ISO 8601 UTC -> epoch seconds (handles trailing Z and fractions)."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


def distinct_states(case):
    """Collapse consecutive identical reads (Exophase refreshes in jumps, so
    most reads repeat the last refresh's page). Returns a list of
    (t_hours_since_release, players, rates) in time order.

    Reads are sorted by timestamp first (the case history is assumed
    chronological, but a stray out-of-order read must not create a phantom
    state or break the "last state" logic). Reads with an unparseable
    timestamp or a missing/non-positive players_tracked are skipped:
    a single malformed read must not crash the program.
    """
    release = None
    game = case.get("game") or {}
    rd = game.get("release_date")
    if rd:
        try:
            release = parse_time(rd + "T00:00:00Z")
        except (ValueError, TypeError, AttributeError):
            release = None
    hist = case.get("history") or []
    parsed = []
    for read in hist:
        try:
            t = parse_time(read["at"])
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        p = read.get("players_tracked")
        if (not isinstance(p, (int, float)) or isinstance(p, bool)
                or p is None or p <= 0):
            continue
        rates = read.get("rates")
        if not isinstance(rates, (list, tuple)):
            rates = []
        parsed.append((t, p, rates))
    parsed.sort(key=lambda item: item[0])
    states = []
    prev_key = None
    for t, p, rates in parsed:
        if release is None:
            release = t
        hours = (t - release) / 3600.0
        key = (p, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        states.append((hours, p, rates))
    return states, release


def fit_logistic_logtime(ts, log_ys, ks):
    """Fit log_y(t) = log_pinf - log(1 + (t50/t)^k) by grid search over
    (k, t50); log_pinf is the mean residual. Returns (predict_fn, resid)."""
    best = None
    for k in ks:
        t50 = 1.0
        while t50 <= 8000.0:
            vals = []
            for t, ly in zip(ts, log_ys):
                tt = max(t, 0.5)
                vals.append(ly + math.log1p((t50 / tt) ** k))
            mean = sum(vals) / len(vals)
            err = sum((v - mean) ** 2 for v in vals)
            if best is None or err < best[0]:
                best = (err, k, t50, mean)
            t50 *= 1.15
    err, k, t50, log_pinf = best
    resid = math.sqrt(err / len(ts))
    def predict(t):
        tt = max(t, 0.5)
        return log_pinf - math.log1p((t50 / tt) ** k)
    return predict, resid


def fit_players(states, x_tgt, h_tgt):
    """Extrapolate P(target). Three candidate fits, combined with a blend
    that trusts the power law when observed growth is moderate and the
    saturating logistic when it is steep. Returns (P_med, log_spread)."""
    fit_states = [(h, p, r) for h, p, r in states if h >= 0.5] or states
    xs = [math.log1p(max(h, 0.0)) for h, _, _ in fit_states]
    ys = [math.log(max(p, 1)) for _, p, _ in fit_states]
    a, b, _, _ = ols(xs, ys)
    # Players tracked is a cumulative count: its log-log slope cannot be
    # wildly negative or steep; clamp before it feeds the blend or the
    # extrapolation, so one jumpy refresh cannot blow up the forecast.
    # The intercept must be re-fitted for the clamped slope: keeping the
    # OLS intercept with a different slope shifts the whole line (a raw
    # slope of 20 clamped to 4.0 would otherwise collapse log_pow by
    # ~(20-4)*x_tgt).
    b_raw = b
    b = max(-0.1, min(4.0, b))
    if b != b_raw:
        a = sum(y - b * x for x, y in zip(xs, ys)) / len(xs)
    log_pow = a + b * x_tgt
    h_last = fit_states[-1][0]
    xr = [x for x, (h, _, _) in zip(xs, fit_states) if h >= h_last - 24.0]
    yr = [y for y, (h, _, _) in zip(ys, fit_states) if h >= h_last - 24.0]
    if len(xr) >= 2:
        ar, br, _, _ = ols(xr, yr)
        br_raw = br
        br = max(-0.1, min(4.0, br))
        if br != br_raw:
            ar = sum(y - br * x for x, y in zip(xr, yr)) / len(xr)
        log_rec = ar + br * x_tgt
    else:
        log_rec = log_pow
    ts = [max(h, 0.5) for h, _, _ in fit_states]
    # k stays 1.0: letting the grid try k=1.5/2.0 was tested and REJECTED —
    # it fits the steep observed window better in-sample but extrapolates
    # worse (sim mean 0.208 vs 0.151 with k=1 only). The "curve falls
    # below the last count" failure it was meant to fix never occurred
    # in 36 synthetic finals; the P_tgt >= last_p guard below covers it.
    log_fn, _ = fit_logistic_logtime(ts, ys, (1.0,))
    log_log = log_fn(h_tgt)
    # exp() argument clamped: (b - 1.9) / 0.25 overflows only for absurd
    # b, but the clamp costs nothing and makes the blend total.
    w = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, (b - 1.9) / 0.25))))
    log_med = w * log_rec + (1.0 - w) * log_log
    spread = max(0.0, (max(log_pow, log_rec, log_log)
                       - min(log_pow, log_rec, log_log)) / 2.0)
    # Players tracked never decreases: the target count is at least the
    # last observed count, whatever the fits say.
    last_p = fit_states[-1][1]
    return max(last_p, math.exp(log_med)), min(spread, SETTINGS["p_spread_cap"])


_SHARE_FIT_CACHE = {}


def fit_share_logistic(hours, rates_pct, h_tgt):
    """Fit s(t) = S / (1 + (t50/t)^k) to observed rates (in percent) on the
    logit scale, grid-searching (k, t50, S). Returns (median_pct, resid_std)
    or None if the fit fails. Rates below RATE_FLOOR are dropped.

    Speed: logit(r) is precomputed once per point, and for fixed (k, t50)
    the per-point factor A = 1 + (t50/t)^k is hoisted out of the S loop,
    since logit(model) = log(S) - log(A - S). A coarse seed pass sets the
    running best first so the full pass can prune any (k, t50, S) whose
    partial squared error already exceeds it. Without this the grid is
    ~5.5k combos x n points per trophy, which times out (37s) on a
    120-state / 55-rising-trophy case against the 8s limit.
    """
    # Launch points are kept, with the time floored at 0.5h (the
    # log-time domain floor); only the players fit drops h < 0.5.
    pts = [(max(h, 0.5), r) for h, r in zip(hours, rates_pct)
           if r is not None and SETTINGS["rate_floor"] <= r < 100.0]
    if len(pts) < 2:
        return None
    cache_key = (tuple(pts), h_tgt)
    if cache_key in _SHARE_FIT_CACHE:
        return _SHARE_FIT_CACHE[cache_key]
    # S candidates: geometric from 0.002, extended past the old ~0.827 cap
    # (the old grid never reached a saturating share near 100%).
    s_vals = []
    s = 0.002
    while s <= 1.0:
        s_vals.append(s)
        s *= 1.25
    for extra in (0.9, 0.95, 1.0):
        if extra not in s_vals:
            s_vals.append(extra)
    s_vals.sort()
    lr = [logit(r) for _, r in pts]
    npts = len(pts)

    def err_for(k, t50, s, best_err):
        """Squared logit error for one (k, t50, S); aborts early once the
        partial sum cannot beat best_err (remaining terms are >= 0)."""
        log_s = math.log(s)
        err = 0.0
        for _pt, l, a in zip(pts, lr, a_vals):
            d = l - (log_s - math.log(max(a - s, 1e-300)))
            err += d * d
            if best_err is not None and err >= best_err:
                return None
        return err

    best = None
    # Coarse seed pass: establish a good running best so the full pass
    # prunes aggressively.
    for k in (1.0, 1.5):
        t50 = 1.0
        while t50 <= 8000.0:
            a_vals = [1.0 + (t50 / t) ** k for t, _ in pts]
            for s in (0.01, 0.05, 0.2, 0.6, 1.0):
                got = err_for(k, t50, s, best[0] if best else None)
                if got is not None and (best is None or got < best[0]):
                    best = (got, k, t50, s)
            t50 *= 2.0
    for k in (0.7, 1.0, 1.5, 2.0):
        t50 = 1.0
        while t50 <= 8000.0:
            a_vals = [1.0 + (t50 / t) ** k for t, _ in pts]
            for s in s_vals:
                got = err_for(k, t50, s, best[0] if best else None)
                if got is not None and (best is None or got < best[0]):
                    best = (got, k, t50, s)
            t50 *= 1.2
    err, k, t50, s = best
    resid = math.sqrt(err / npts)
    tt = max(h_tgt, 0.5)
    med = 100.0 * s / (1.0 + (t50 / tt) ** k)
    result = (med, resid)
    _SHARE_FIT_CACHE[cache_key] = result
    return result


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    hist = case.get("history") or []
    if not hist:
        raise ValueError(f"{name} has no history")
    states, release = distinct_states(case)
    if not states:
        raise ValueError(f"{name} has no readable history")
    horizon_days = case["horizon_minutes"] / 1440.0
    h_last = states[-1][0]
    # The horizon runs from the CUTOFF (case "at", or the last read's
    # timestamp), not from the last refresh: if the last refresh was
    # 6-12h before the cutoff, anchoring the target at the last refresh
    # extrapolates 6-12h too short.
    h_cutoff = None
    try:
        h_cutoff = (parse_time(case["at"]) - release) / 3600.0
    except (KeyError, ValueError, TypeError, AttributeError):
        h_cutoff = None
    if h_cutoff is None:
        try:
            h_cutoff = (parse_time(hist[-1]["at"]) - release) / 3600.0
        except (KeyError, ValueError, TypeError, AttributeError, IndexError):
            h_cutoff = states[-1][0]
    h_tgt = h_cutoff + case["horizon_minutes"] / 60.0
    x_tgt = math.log1p(max(h_tgt, 0.0))
    last_rates = states[-1][2]
    last_p = states[-1][1]

    flat = len(states) == 1
    if flat:
        p_tgt = last_p * REFRESH_JUMP
        p_log_spread = 0.0
    else:
        p_tgt, p_log_spread = fit_players(states, x_tgt, h_tgt)

    # Per-trophy series. Reliable earner points only (rate >= RATE_FLOOR).
    # NOTE: the early reads are deliberately KEPT here (unlike the
    # players fit, which drops h < 0.5; the share fit keeps them,
    # floored at 0.5h): the launch point is a
    # valuable anchor for the earner slope, and dropping it was tested
    # and made the synthetic finals worse (mean 0.151 -> 0.208 with the
    # k-grid also reverted). All states are used for the earner fits.
    per_trophy = []
    slopes = []
    for i, trophy in enumerate(trophies):
        pts = []
        share_pts = []
        for (h, p, rates) in states:
            if i >= len(rates):
                continue
            r = rates[i]
            if r is None:
                continue
            share_pts.append((h, r))
            if r >= SETTINGS["rate_floor"]:
                e = (r / 100.0) * p
                if e > 0:
                    pts.append((h, math.log1p(max(h, 0.0)), math.log(e)))
        if not pts:
            per_trophy.append((None, share_pts))
            continue
        a, b, resid, n = ols([p[1] for p in pts], [p[2] for p in pts])
        # exclude the 100%-trophy's slope (it tracks P itself) from the
        # shrinkage target
        if not (len(share_pts) > 0 and all(r >= 100.0 for _, r in share_pts if r is not None)):
            # A single reliable point gives a fake slope of exactly 0.0
            # (OLS with n=1); only real fits (n>=2) inform the median.
            if n >= 2:
                slopes.append(b)
        per_trophy.append(((pts, a, b, resid, n), share_pts))
    slopes.sort()
    med_slope = slopes[len(slopes) // 2] if slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = last_rates[i] if i < len(last_rates) else None
        if rate_last is not None and rate_last >= 100.0:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        if flat:
            e_last = (rate_last / 100.0) * last_p if rate_last is not None else 0.0
            med = 100.0 * e_last / p_tgt if p_tgt > 0 else 0.01
            # The narrow flat spread is only honest for a short horizon
            # (the documented launch-night refresh) with a tiny player
            # base; over 4.5 days, or with no observed rate at all, the
            # spread must be wide.
            if rate_last is None:
                sigma = 1.0
            elif horizon_days <= 0.25 and (last_p is not None and last_p < 2000):
                sigma = FLAT_SIGMA
            else:
                sigma = 1.0
        else:
            info, share_pts = per_trophy[i]
            # trend of the share on reliable rates only
            spts = [(math.log1p(max(h, 0.0)), logit(r)) for h, r in share_pts
                    if r is not None and SETTINGS["rate_floor"] <= r < 100.0]
            if len(spts) >= 2:
                _, b_share, _, _ = ols([p[0] for p in spts], [p[1] for p in spts])
            else:
                b_share = 0.0
            # --- counts decomposition (E/P) median ---
            if not info:
                med_ep, sigma_ep = None, None
            else:
                pts, a, b, resid, n = info
                _, x_last, y_last = pts[-1]
                recent = [(x, y) for (h, x, y) in pts if h >= h_last - 24.0]
                if len(recent) >= 2:
                    _, b_rec, _, _ = ols([p[0] for p in recent], [p[1] for p in recent])
                    # Clamp the recent slope BEFORE blending: one jumpy
                    # refresh (rounding noise) must not drag the blend
                    # down and discard the full-history slope.
                    b_rec = max(-0.2, min(4.0, b_rec))
                    b = 0.5 * b + 0.5 * b_rec
                # shrink toward the cross-trophy median slope
                if SETTINGS["shrink"] == "adaptive":
                    k_shrink = 2.0 if resid < 0.3 else 6.0
                else:
                    k_shrink = 4.0
                w = n / (n + k_shrink)
                b = w * b + (1.0 - w) * med_slope
                b = max(-0.2, min(4.0, b))
                if SETTINGS["anchor"] == "refit":
                    # re-fit the intercept for the blended slope
                    a = sum(y - b * x for (_, x, y) in pts) / len(pts)
                    log_e_tgt = a + b * x_tgt
                else:
                    # anchor at the last reliable point
                    log_e_tgt = y_last + b * (x_tgt - x_last)
                # Rounding cap: if the last observed rate is below the
                # rounding floor, the true earner count at the last read
                # is at most ((rate_last + 0.005) / 100) * last_p. If the
                # fit implies more earners than that at the last read,
                # re-anchor the trajectory at the cap (never below the
                # last reliable earner count: earners never decrease).
                if (rate_last is not None and rate_last < SETTINGS["rate_floor"]
                        and last_p is not None and last_p > 0):
                    cap_e = ((rate_last + 0.005) / 100.0) * last_p
                    x_last_state = math.log1p(max(h_last, 0.0))
                    log_e_at_last = log_e_tgt - b * (x_tgt - x_last_state)
                    if log_e_at_last > math.log(max(cap_e, 1e-9)):
                        log_e_at_last = max(math.log(max(cap_e, 1e-9)), y_last)
                        log_e_tgt = log_e_at_last + b * (x_tgt - x_last_state)
                # Earners are cumulative: the target count is at least the
                # last reliably observed count.
                log_e_tgt = max(log_e_tgt, y_last)
                e_tgt = min(math.exp(log_e_tgt), 0.98 * p_tgt)
                med_ep = 100.0 * e_tgt / p_tgt if p_tgt > 0 else 0.01
                sigma_ep = (0.25 + 0.15 * math.sqrt(horizon_days) + 0.4 * resid
                            + 0.35 * p_log_spread + (0.25 if len(states) < 5 else 0.0))
            # --- logistic-share median ---
            med_sh, sigma_sh = None, None
            if b_share > 0:
                got = fit_share_logistic([h for h, _ in share_pts],
                                         [r for _, r in share_pts], h_tgt)
                if got is not None:
                    med_sh, resid_s = got
                    sigma_sh = (0.25 + 0.15 * math.sqrt(horizon_days)
                                + 0.4 * resid_s + 0.2 * p_log_spread
                                + (0.25 if len(states) < 5 else 0.0))
            # --- choose ---
            # features of the share trajectory (reliable rates only)
            rel_rates = [r for _, r in share_pts
                         if r is not None and SETTINGS["rate_floor"] <= r < 100.0]
            rate_first = rel_rates[0] if rel_rates else None
            rate_min = min(rel_rates) if rel_rates else None
            recent_spts = [(math.log1p(max(h, 0.0)), logit(r))
                         for h, r in share_pts
                         if r is not None
                         and SETTINGS["rate_floor"] <= r < 100.0
                         and h >= h_last - 24.0]
            if len(recent_spts) >= 2:
                _, b_share_recent, _, _ = ols([p[0] for p in recent_spts],
                                              [p[1] for p in recent_spts])
            else:
                b_share_recent = b_share
            use_share = False
            if med_sh is not None and b_share >= SETTINGS["share_slope_min"]:
                use_share = True  # (a) strong sustained share rise
            elif (med_sh is not None and rate_last is not None and rate_min
                  and rate_first is not None
                  and rate_last / rate_min >= SETTINGS["share_rise_min"]
                  and rate_last / rate_first >= SETTINGS["share_rise_min"]
                  and b_share_recent < SETTINGS["share_flat_slope"]):
                use_share = True  # (b) rose a lot from launch and from
                # its minimum, now flat: saturated
            elif (med_sh is not None and med_ep is not None
                  and rate_last is not None and rate_last > 0
                  and rate_first is not None
                  and med_ep / rate_last > SETTINGS["ep_rise_max"]
                  and b_share_recent <= SETTINGS["ep_flat_slope"]
                  and rate_last > rate_first):
                use_share = True  # (c) E/P implies a huge share rise but
                # the share is flat and above launch: earner growth was
                # P-driven, so it stops when P stops
            if use_share:
                med, sigma = med_sh, sigma_sh
            elif med_ep is not None:
                med, sigma = med_ep, sigma_ep
                # if the share fit exists and disagrees wildly, widen the
                # spread: the trophy's dynamics are genuinely ambiguous
                if med_sh is not None and SETTINGS["widen"]:
                    d = abs(logit(med_sh) - logit(med_ep))
                    sigma = sigma + SETTINGS["widen"] * d
            elif med_sh is not None:
                med, sigma = med_sh, sigma_sh
            else:
                med = rate_last if rate_last is not None else 0.01
                sigma = 1.0
            sigma = max(0.08, min(2.0, sigma * SETTINGS["sigma_scale"]))
        med = clamp_pct(med)
        lg = logit(med)
        qs = []
        for z in Z:
            q = inv_logit(lg + z * sigma)
            qs.append(round4(min(100.0, max(0.01, q))))
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
