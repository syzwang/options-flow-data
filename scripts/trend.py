#!/usr/bin/env python3
"""
Multi-day flow trend analysis.

Usage:
    python3 trend.py MSTR [--days 5]
    python3 trend.py TSLA --days 10
    python3 trend.py MSTR TSLA --days 5 --html

Loads daily snapshots and computes:
1. Delta-25 P/C IV spread time series + zero-crossing detection
2. ATM IV trend across days
3. OI day-over-day changes (position delta)
4. Expected Move accuracy (predicted vs actual)
5. Volume persistence by strike (multi-day conviction)

Saves: ~/options_portfolio/flow_snapshots/trend-TICKER.json
"""
import os, sys, json, glob, argparse, warnings, subprocess, re
from datetime import datetime, timedelta

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "numpy", "pandas"])
import numpy as np
import pandas as pd
import yfinance as yf
warnings.filterwarnings('ignore')

SNAPSHOT_DIR = os.environ.get('FLOW_SNAPSHOT_DIR') or os.path.expanduser("~/options_portfolio/flow_snapshots")
REPORT_DIR = os.environ.get('FLOW_REPORT_DIR') or os.path.expanduser("~/options_reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_snapshots(ticker, days=10):
    """Load the most recent N snapshots for a ticker (date-prefixed files only)."""
    import re
    pattern = os.path.join(SNAPSHOT_DIR, f"*-{ticker}.json")
    all_files = sorted(glob.glob(pattern))
    # Only keep files matching YYYY-MM-DD-TICKER.json pattern (exclude trend-TICKER.json etc)
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}-' + re.escape(ticker) + r'\.json$')
    files = [f for f in all_files if date_pattern.search(os.path.basename(f))]
    files = files[-days:]  # keep most recent N
    snapshots = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            snapshots.append(data)
    return snapshots


def pc_spread_timeseries(snapshots):
    """
    Build delta-25 Put/Call IV spread time series per expiration bucket.
    Returns dict of {bucket: [(date, spread_value), ...]}
    Buckets: 'front' (nearest), '2w', '1m', '2m' based on DTE ranges.
    """
    def dte_bucket(dte):
        if dte <= 7: return 'front'
        if dte <= 14: return '2w'
        if dte <= 35: return '1m'
        if dte <= 70: return '2m'
        return 'far'

    series = {}
    for snap in snapshots:
        date = snap['date']
        spot = snap['spot']
        for t in snap.get('term', []):
            spread = t.get('pc_iv_spread', None)
            if spread is None:
                continue
            bucket = dte_bucket(t['dte'])
            if bucket not in series:
                series[bucket] = []
            series[bucket].append({
                'date': date,
                'expiry': t['expiry'],
                'dte': t['dte'],
                'spread': spread,
                'd25_call_iv': t.get('d25_call_iv', 0),
                'd25_put_iv': t.get('d25_put_iv', 0),
                'atm_iv': t.get('atm_iv', 0),
                'spot': spot,
            })
    return series


def detect_zero_crossings(series):
    """
    Detect when P/C IV spread crosses below zero (call skew dominates).
    Returns list of crossing events with date, bucket, and context.
    Multiple expiries on the same date within a bucket are averaged into one
    point-per-date before comparing — otherwise cross-expiry jumps get mistaken
    for time-based crossings.
    """
    crossings = []
    for bucket, points in series.items():
        # Collapse to one (spread, spot) per date by averaging across expiries
        by_date = {}
        for p in points:
            by_date.setdefault(p['date'], []).append(p)
        daily = []
        for date in sorted(by_date.keys()):
            group = by_date[date]
            avg_spread = sum(g['spread'] for g in group) / len(group)
            daily.append({'date': date, 'spread': avg_spread, 'spot': group[0]['spot']})
        sorted_pts = daily
        for i in range(1, len(sorted_pts)):
            prev = sorted_pts[i-1]['spread']
            curr = sorted_pts[i]['spread']
            if prev >= 0 and curr < 0:
                crossings.append({
                    'type': 'bearish_signal',
                    'bucket': bucket,
                    'date': sorted_pts[i]['date'],
                    'prev_date': sorted_pts[i-1]['date'],
                    'spread_from': round(prev, 2),
                    'spread_to': round(curr, 2),
                    'spot': sorted_pts[i]['spot'],
                    'message': f"P/C spread crossed below zero in {bucket} bucket: "
                               f"{prev:+.1f} -> {curr:+.1f} pts. "
                               f"Historical pattern: call skew dominance often signals near-term top."
                })
            elif prev <= 0 and curr > 0:
                crossings.append({
                    'type': 'normalization',
                    'bucket': bucket,
                    'date': sorted_pts[i]['date'],
                    'prev_date': sorted_pts[i-1]['date'],
                    'spread_from': round(prev, 2),
                    'spread_to': round(curr, 2),
                    'spot': sorted_pts[i]['spot'],
                    'message': f"P/C spread normalized back above zero in {bucket} bucket: "
                               f"{prev:+.1f} -> {curr:+.1f} pts. Bullish euphoria fading."
                })
    return crossings


def skew_percentile_analysis(pc_series, bucket='front'):
    """
    Percentile rank of current front-bucket P/C spread within available history.
    Lower percentile = more negative = more extreme call skew.
    Small samples are flagged; interpretation notes adjust accordingly.
    """
    points = pc_series.get(bucket, [])
    if len(points) < 2:
        return None
    sorted_pts = sorted(points, key=lambda x: x['date'])
    values = [p['spread'] for p in sorted_pts]
    current = values[-1]
    # Percentile: fraction of history <= current value
    below_or_eq = sum(1 for v in values if v <= current)
    pct_rank = below_or_eq / len(values) * 100
    n = len(values)
    # Label historical context based on where current sits
    if pct_rank <= 5:
        regime = 'extreme_call_skew'
        label = 'Extreme call skew (bottom 5%)'
    elif pct_rank <= 20:
        regime = 'heavy_call_skew'
        label = 'Heavy call skew (bottom 20%)'
    elif pct_rank <= 80:
        regime = 'normal_range'
        label = 'Within normal range'
    elif pct_rank <= 95:
        regime = 'heavy_put_skew'
        label = 'Heavy put skew (top 20%)'
    else:
        regime = 'extreme_put_skew'
        label = 'Extreme put skew (top 5%)'
    return {
        'current': round(current, 2),
        'percentile': round(pct_rank, 1),
        'sample_size': n,
        'min': round(min(values), 2),
        'max': round(max(values), 2),
        'median': round(sorted(values)[len(values)//2], 2),
        'regime': regime,
        'label': label,
        'reliable': n >= 60,
    }


def dte_call_build_ratio(snapshots):
    """
    Aggregate call OI delta across ALL expirations and bucket by DTE:
      short (<=14d), mid (15-45d), long (>45d).
      Short-heavy call build = speculative / non-conviction.
      Balanced = structural bullishness.
    """
    if len(snapshots) < 2:
        return None
    prev, curr = snapshots[-2], snapshots[-1]
    prev_chains = prev.get('chains', {})
    curr_chains = curr.get('chains', {})
    common = set(prev_chains.keys()) & set(curr_chains.keys())
    if not common:
        return None
    # DTE is stored per-term; build expiry->DTE map from current term
    exp_dte = {t['expiry']: t['dte'] for t in curr.get('term', [])}
    buckets = {'short': 0, 'mid': 0, 'long': 0}
    put_buckets = {'short': 0, 'mid': 0, 'long': 0}
    for exp in common:
        dte = exp_dte.get(exp)
        if dte is None:
            continue
        if dte <= 14:
            key = 'short'
        elif dte <= 45:
            key = 'mid'
        else:
            key = 'long'
        prev_calls = {r['strike']: r.get('openInterest', 0) for r in prev_chains[exp].get('calls', [])}
        curr_calls = {r['strike']: r.get('openInterest', 0) for r in curr_chains[exp].get('calls', [])}
        prev_puts = {r['strike']: r.get('openInterest', 0) for r in prev_chains[exp].get('puts', [])}
        curr_puts = {r['strike']: r.get('openInterest', 0) for r in curr_chains[exp].get('puts', [])}
        call_delta = sum(curr_calls.get(k, 0) - prev_calls.get(k, 0) for k in set(prev_calls) | set(curr_calls))
        put_delta = sum(curr_puts.get(k, 0) - prev_puts.get(k, 0) for k in set(prev_puts) | set(curr_puts))
        buckets[key] += call_delta
        put_buckets[key] += put_delta
    total_call = sum(buckets.values())
    if total_call <= 0:
        return {'call_buckets': buckets, 'put_buckets': put_buckets, 'total_call': total_call,
                'short_pct': 0, 'regime': 'net_unwind',
                'prev_date': prev['date'], 'date': curr['date']}
    short_pct = buckets['short'] / total_call * 100
    long_pct = buckets['long'] / total_call * 100
    if short_pct >= 80:
        regime = 'speculative'
    elif short_pct >= 60:
        regime = 'near_term_lean'
    elif long_pct >= 30:
        regime = 'structural'
    else:
        regime = 'balanced'
    return {
        'call_buckets': buckets,
        'put_buckets': put_buckets,
        'total_call': total_call,
        'short_pct': round(short_pct, 1),
        'mid_pct': round(buckets['mid'] / total_call * 100, 1),
        'long_pct': round(long_pct, 1),
        'regime': regime,
        'prev_date': prev['date'],
        'date': curr['date'],
    }


def gex_walls(snapshot, top_n=5):
    """
    Dealer Gamma Exposure walls. Uses pre-computed 'gex' per row (signed:
    calls positive, puts negative). Aggregates |gex| across all expirations
    per strike and finds the largest concentrations above/below spot.

    Returns: dict with 'call_walls' (above spot), 'put_walls' (below spot),
    'net_gex' (total signed) and 'regime' classification.
    """
    spot = snapshot.get('spot', 0)
    if spot <= 0:
        return None
    call_by_strike = {}
    put_by_strike = {}
    total_call_gex = 0.0
    total_put_gex = 0.0
    for exp, chain in snapshot.get('chains', {}).items():
        for r in chain.get('calls', []):
            g = r.get('gex', 0) or 0
            k = r['strike']
            call_by_strike[k] = call_by_strike.get(k, 0) + g
            total_call_gex += g
        for r in chain.get('puts', []):
            g = r.get('gex', 0) or 0
            k = r['strike']
            put_by_strike[k] = put_by_strike.get(k, 0) + g
            total_put_gex += g

    # Call walls: strikes ABOVE spot, ranked by |call gex|. Exclude strikes >25% OTM (irrelevant to near-term dealer hedging)
    call_walls = sorted(
        [{'strike': k, 'gex': v, 'otm_pct': (k / spot - 1) * 100}
         for k, v in call_by_strike.items()
         if k > spot and (k / spot - 1) <= 0.25],
        key=lambda x: abs(x['gex']), reverse=True
    )[:top_n]
    # Put walls: strikes BELOW spot, ranked by |put gex| (puts are negative, abs for ranking)
    put_walls = sorted(
        [{'strike': k, 'gex': v, 'otm_pct': (1 - k / spot) * 100}
         for k, v in put_by_strike.items()
         if k < spot and (1 - k / spot) <= 0.25],
        key=lambda x: abs(x['gex']), reverse=True
    )[:top_n]

    net_gex = total_call_gex + total_put_gex
    # Regime: positive net = dealers long gamma → suppress moves (magnet); negative = amplify (squeeze risk)
    if net_gex > abs(total_put_gex) * 0.5:
        regime = 'long_gamma'
    elif net_gex < 0:
        regime = 'short_gamma'
    else:
        regime = 'neutral'
    return {
        'call_walls': call_walls,
        'put_walls': put_walls,
        'total_call_gex': total_call_gex,
        'total_put_gex': total_put_gex,
        'net_gex': net_gex,
        'regime': regime,
        'spot': spot,
    }


def btc_context(ticker, snapshots):
    """
    BTC price, 24h/7d moves, and rolling correlation with underlying.
    Only relevant for BTC-proxy names (MSTR primarily). Returns None on any error.
    """
    if ticker.upper() != 'MSTR':
        return None
    try:
        import yfinance as yf
        btc = yf.Ticker('BTC-USD').history(period='90d', interval='1d')
        if btc.empty or len(btc) < 30:
            return None
        btc_close = btc['Close']
        btc_now = float(btc_close.iloc[-1])
        btc_24h = (btc_now / float(btc_close.iloc[-2]) - 1) * 100 if len(btc_close) >= 2 else 0
        btc_7d = (btc_now / float(btc_close.iloc[-8]) - 1) * 100 if len(btc_close) >= 8 else 0
        # Rolling 30d correlation with MSTR daily returns
        correlation = None
        beta = None
        if len(snapshots) >= 5:
            import datetime as _dt
            mstr_series = {}
            for s in snapshots:
                try:
                    d = _dt.datetime.strptime(s['date'], '%Y-%m-%d').date()
                    mstr_series[d] = s['spot']
                except Exception:
                    continue
            # Align with BTC dates
            pairs = []
            dates_sorted = sorted(mstr_series.keys())
            for i in range(1, len(dates_sorted)):
                d_prev = dates_sorted[i-1]
                d_curr = dates_sorted[i]
                mstr_ret = (mstr_series[d_curr] / mstr_series[d_prev] - 1) * 100
                # Find BTC closes on same dates
                btc_prev = None
                btc_curr = None
                for idx, row in btc.iterrows():
                    idate = idx.date() if hasattr(idx, 'date') else idx
                    if idate == d_prev:
                        btc_prev = float(row['Close'])
                    if idate == d_curr:
                        btc_curr = float(row['Close'])
                if btc_prev and btc_curr:
                    btc_ret = (btc_curr / btc_prev - 1) * 100
                    pairs.append((mstr_ret, btc_ret))
            if len(pairs) >= 4:
                mx = [p[0] for p in pairs]
                bx = [p[1] for p in pairs]
                mean_m = sum(mx) / len(mx)
                mean_b = sum(bx) / len(bx)
                cov = sum((mx[i] - mean_m) * (bx[i] - mean_b) for i in range(len(pairs))) / len(pairs)
                var_m = sum((v - mean_m) ** 2 for v in mx) / len(mx)
                var_b = sum((v - mean_b) ** 2 for v in bx) / len(bx)
                if var_m > 0 and var_b > 0:
                    correlation = cov / (var_m ** 0.5 * var_b ** 0.5)
                    beta = cov / var_b  # MSTR beta to BTC
        # Divergence signal: today BTC flat but MSTR active, or vice versa
        mstr_spot = snapshots[-1]['spot'] if snapshots else 0
        mstr_24h = 0
        if len(snapshots) >= 2:
            prev_spot = snapshots[-2]['spot']
            if prev_spot > 0:
                mstr_24h = (mstr_spot / prev_spot - 1) * 100
        # Flag divergence if MSTR move > 1.5× (|beta| × BTC move) or BTC flat + MSTR non-flat
        divergence = None
        if beta and abs(btc_24h) > 0.1:
            expected_mstr = beta * btc_24h
            residual = mstr_24h - expected_mstr
            if abs(residual) > max(2.0, abs(expected_mstr) * 0.5):
                divergence = {
                    'expected': round(expected_mstr, 2),
                    'actual': round(mstr_24h, 2),
                    'residual': round(residual, 2),
                    'direction': 'premium' if residual > 0 else 'discount',
                }
        elif abs(btc_24h) < 0.5 and abs(mstr_24h) > 2.0:
            divergence = {
                'expected': 0.0,
                'actual': round(mstr_24h, 2),
                'residual': round(mstr_24h, 2),
                'direction': 'premium' if mstr_24h > 0 else 'discount',
            }
        return {
            'btc_price': round(btc_now, 0),
            'btc_24h': round(btc_24h, 2),
            'btc_7d': round(btc_7d, 2),
            'mstr_24h': round(mstr_24h, 2),
            'correlation': round(correlation, 2) if correlation is not None else None,
            'beta': round(beta, 2) if beta is not None else None,
            'divergence': divergence,
            'sample_n': len(snapshots),
        }
    except Exception:
        return None


def oi_delta(snapshots):
    """
    Compute day-over-day OI changes by strike.
    Returns list of (date, strike, call_oi_delta, put_oi_delta) for the most liquid expiration.
    """
    if len(snapshots) < 2:
        return []

    deltas = []
    for i in range(1, len(snapshots)):
        prev, curr = snapshots[i-1], snapshots[i]
        prev_chains = prev.get('chains', {})
        curr_chains = curr.get('chains', {})

        # Find overlapping expirations
        common_exps = set(prev_chains.keys()) & set(curr_chains.keys())
        if not common_exps:
            continue

        # Use nearest common expiration
        nearest_exp = min(common_exps)
        pc = prev_chains[nearest_exp]
        cc = curr_chains[nearest_exp]

        # Build OI maps
        prev_call_oi = {r['strike']: r.get('openInterest', 0) for r in pc.get('calls', [])}
        prev_put_oi = {r['strike']: r.get('openInterest', 0) for r in pc.get('puts', [])}
        curr_call_oi = {r['strike']: r.get('openInterest', 0) for r in cc.get('calls', [])}
        curr_put_oi = {r['strike']: r.get('openInterest', 0) for r in cc.get('puts', [])}

        all_strikes = sorted(set(list(prev_call_oi.keys()) + list(curr_call_oi.keys()) +
                                 list(prev_put_oi.keys()) + list(curr_put_oi.keys())))

        day_deltas = []
        for strike in all_strikes:
            c_delta = curr_call_oi.get(strike, 0) - prev_call_oi.get(strike, 0)
            p_delta = curr_put_oi.get(strike, 0) - prev_put_oi.get(strike, 0)
            if c_delta != 0 or p_delta != 0:
                day_deltas.append({
                    'strike': strike,
                    'call_oi_delta': c_delta,
                    'put_oi_delta': p_delta,
                    'net': c_delta + p_delta,
                })

        if day_deltas:
            deltas.append({
                'date': curr['date'],
                'prev_date': prev['date'],
                'expiry': nearest_exp,
                'spot': curr['spot'],
                'changes': sorted(day_deltas, key=lambda x: abs(x['call_oi_delta']) + abs(x['put_oi_delta']), reverse=True)[:30],
                'total_call_oi_delta': sum(d['call_oi_delta'] for d in day_deltas),
                'total_put_oi_delta': sum(d['put_oi_delta'] for d in day_deltas),
            })

    return deltas


def volume_persistence(snapshots):
    """
    Find strikes with consistent high volume across multiple days.
    Persistent volume = real conviction, not noise.
    """
    if len(snapshots) < 2:
        return {'call_persistent': [], 'put_persistent': []}

    # Collect volume by strike across all days
    call_vol_by_strike = {}
    put_vol_by_strike = {}

    for snap in snapshots:
        for exp, chain in snap.get('chains', {}).items():
            for r in chain.get('calls', []):
                k = r['strike']
                if k not in call_vol_by_strike:
                    call_vol_by_strike[k] = []
                call_vol_by_strike[k].append({'date': snap['date'], 'vol': r.get('volume', 0), 'expiry': exp})
            for r in chain.get('puts', []):
                k = r['strike']
                if k not in put_vol_by_strike:
                    put_vol_by_strike[k] = []
                put_vol_by_strike[k].append({'date': snap['date'], 'vol': r.get('volume', 0), 'expiry': exp})

    def find_persistent(vol_map, min_days=2, min_vol=100):
        results = []
        for strike, entries in vol_map.items():
            active_days = [e for e in entries if e['vol'] >= min_vol]
            if len(active_days) >= min_days:
                total_vol = sum(e['vol'] for e in active_days)
                results.append({
                    'strike': strike,
                    'active_days': len(active_days),
                    'total_days': len(entries),
                    'total_volume': total_vol,
                    'avg_daily_vol': total_vol / len(active_days),
                    'dates': [e['date'] for e in active_days],
                    'persistence_score': len(active_days) / len(snapshots),
                })
        return sorted(results, key=lambda x: x['total_volume'], reverse=True)[:15]

    return {
        'call_persistent': find_persistent(call_vol_by_strike),
        'put_persistent': find_persistent(put_vol_by_strike),
    }


def em_accuracy(snapshots):
    """
    Compare past Expected Move predictions with actual price movement.
    Scales EM by sqrt(calendar_days) so Fri→Mon gaps (weekend) compare fairly.
    """
    from datetime import datetime as _dt
    import math
    results = []
    for i in range(len(snapshots) - 1):
        snap = snapshots[i]
        next_snap = snapshots[i + 1]

        for t in snap.get('term', []):
            if t['dte'] <= 7:  # front-month EM
                em_upper_1d = t.get('em_upper', 0)
                em_lower_1d = t.get('em_lower', 0)
                if em_upper_1d == 0:
                    continue

                actual_price = next_snap['spot']
                predicted_spot = snap['spot']
                em_pct_1d = t.get('expected_move_pct', 0)

                # Calendar-day gap (Fri→Mon = 3 days)
                d1 = _dt.strptime(snap['date'], '%Y-%m-%d')
                d2 = _dt.strptime(next_snap['date'], '%Y-%m-%d')
                cal_days = max((d2 - d1).days, 1)
                scale = math.sqrt(cal_days)

                em_pct = em_pct_1d * scale
                em_width = predicted_spot * em_pct / 100
                em_upper = predicted_spot + em_width
                em_lower = predicted_spot - em_width

                within_range = em_lower <= actual_price <= em_upper
                actual_move_signed = (actual_price - predicted_spot) / predicted_spot * 100
                actual_move_pct = abs(actual_move_signed)

                results.append({
                    'date': snap['date'],
                    'next_date': next_snap['date'],
                    'cal_days': cal_days,
                    'predicted_spot': predicted_spot,
                    'actual_spot': actual_price,
                    'em_pct': round(em_pct, 2),
                    'em_pct_1d': round(em_pct_1d, 2),
                    'actual_move_pct': round(actual_move_pct, 2),
                    'actual_move_signed': round(actual_move_signed, 2),
                    'em_upper': round(em_upper, 2),
                    'em_lower': round(em_lower, 2),
                    'within_em': within_range,
                    'move_vs_em': round(actual_move_pct / em_pct, 2) if em_pct > 0 else 0,
                })
                break  # only front-month
    return results


def iv_trend(snapshots):
    """Track ATM IV and RV30 across days.

    Front IV: first expiry with DTE >= 5 — avoids 0/1-DTE collapse at expiration
    (e.g., Friday-close 0DTE has near-zero IV which is pathological, not signal).
    Falls back to first available term if no DTE>=5 entry exists.
    """
    points = []
    for snap in snapshots:
        terms = snap.get('term', [])
        front_term = next((t for t in terms if t.get('dte', 0) >= 5), terms[0] if terms else None)
        front_iv = front_term['atm_iv'] if front_term else 0
        points.append({
            'date': snap['date'],
            'spot': snap['spot'],
            'rv30': snap.get('rv30', 0),
            'iv_pct_2yr': snap.get('iv_pct_2yr', 0),
            'front_atm_iv': front_iv,
            'nvrp': round(front_iv / snap['rv30'], 2) if snap.get('rv30', 0) > 0 else 0,
        })
    return points


def skew_alert_level(spread_value):
    """Classify current P/C IV spread into alert levels."""
    if spread_value < -5:
        return 'EXTREME_CALL_SKEW'
    elif spread_value < 0:
        return 'CALL_SKEW'
    elif spread_value < 5:
        return 'MUTED'
    elif spread_value < 10:
        return 'NORMAL'
    elif spread_value < 25:
        return 'ELEVATED_PUT_SKEW'
    else:
        return 'PANIC'


def _detect_mode():
    """Auto-detect preopen/intraday/postclose from current ET time."""
    try:
        from zoneinfo import ZoneInfo
        et = datetime.now(ZoneInfo('America/New_York'))
    except Exception:
        et = datetime.now()
    if et.weekday() >= 5:
        return 'postclose'
    mins = et.hour * 60 + et.minute
    if mins < 570:       # before 9:30 ET
        return 'preopen'
    if mins < 960:       # before 16:00 ET
        return 'intraday'
    return 'postclose'


ZH_REPLACEMENTS = [
    # ===== Mode banners (order: longer strings first to avoid partial replace) =====
    ("PRE-OPEN MODE &mdash; build today's playbook", "盘前模式 — 制定今日剧本"),
    ("INTRADAY MODE &mdash; this dashboard is static", "盘中模式 — 本仪表板不更新"),
    ("POST-CLOSE MODE &mdash; review &amp; validate", "盘后模式 — 复盘与验证"),
    ("Read once, write plan, then stop reading &mdash; data is static until tomorrow's snapshot.",
     "看一次、写计划、关掉 — 数据到明天 snapshot 前都不会变。"),
    ("Data is a <b style=\"color:#e6edf3;\">pre-open snapshot</b> (yesterday's close OI + 24-day history) and does NOT update during the session.",
     "数据是<b style=\"color:#e6edf3;\">盘前 snapshot</b>（昨收 OI + 24 天历史），盘中<b>不会更新</b>。"),
    ("Use real-time options chain / time &amp; sales for live flow. Only come back here if price is testing a Wall or Persistent strike.",
     "实时流向请用期权链 / 分时成交。只有当价格触及 Wall 或持续放量行权价时才回来参考。"),
    ("Feed findings into your trade journal and tomorrow's plan.", "把发现写进交易日志和明日计划。"),
    ("Focus on:", "关注："),
    (" (price anchors &mdash; set limit orders near them)", "（价格锚点 — 在附近挂限价单）"),
    (" (multi-day S/R levels)", "（多日支撑/阻力位）"),
    (" (overnight smart-money positioning)", "（隔夜 smart-money 持仓）"),
    (" (premium-selling edge)", "（卖权溢价优势）"),
    (" (did today land inside the envelope?)", "（今日是否落在区间内?）"),
    (" (did new positions validate yesterday's signal?)", "（新开仓是否验证了昨日信号?）"),
    (" (tested / broken / held?)", "（测试 / 突破 / 守住?）"),
    # ===== Card titles =====
    (">Flow Trend Analysis</h1>", ">期权流向分析</h1>"),
    (">Dealer Gamma Exposure (GEX) Walls</h2>", ">做市商 Gamma 墙（GEX）</h2>"),
    (">BTC Context (MSTR = BTC-levered proxy)</h2>", ">BTC 背景（MSTR = BTC 杠杆代理）</h2>"),
    (">mNAV (MSTR Premium to BTC NAV)</h2>", ">mNAV（MSTR 相对 BTC 净值溢价）</h2>"),
    ("Premium MSTR trades at vs the underlying BTC stash. <b style=\"color:#e6edf3;\">Simple</b> = MarketCap / BTC reserve. <b style=\"color:#e6edf3;\">EV-based</b> = (MC + debt + pref − cash) / BTC reserve (matches strategy.com). Cheap &lt; 1.2x · Normal 1.2–1.8x · Expensive &gt; 1.8x.",
     "MSTR 相对其 BTC 储备的溢价。<b style=\"color:#e6edf3;\">简易</b> = 市值 / BTC 储备价值。<b style=\"color:#e6edf3;\">企业价值口径</b> = (市值 + 债务 + 优先股 − 现金) / BTC 储备价值（与 strategy.com 口径一致）。便宜 &lt; 1.2x · 正常 1.2–1.8x · 偏贵 &gt; 1.8x。"),
    ("· source: strategy.com", "· 数据来源：strategy.com"),
    (">Simple mNAV</div>", ">简易 mNAV</div>"),
    (">EV-based mNAV</div>", ">企业价值口径 mNAV</div>"),
    (">Verdict</div>", ">判定</div>"),
    (">BTC Held</div>", ">BTC 持仓</div>"),
    (">BTC Reserve</div>", ">BTC 储备价值</div>"),
    (">Market Cap</div>", ">市值</div>"),
    (">Cheap</div>", ">便宜</div>"),
    (">Normal</div>", ">正常</div>"),
    (">Expensive</div>", ">偏贵</div>"),
    ("Holdings as of ", "持仓数据日期 "),
    (">Zero-Crossing Alerts (P/C Spread)</h2>", ">零轴穿越预警（P/C 偏度）</h2>"),
    (">Delta-25 Put/Call IV Spread (by expiration bucket)</h2>", ">Delta-25 Put/Call IV 偏度（按到期分桶）</h2>"),
    (">ATM IV vs RV30 Trend</h2>", ">ATM IV vs RV30 趋势</h2>"),
    (">IV Term Structure (ATM IV by DTE)</h2>", ">IV 期限结构（ATM IV 按 DTE）</h2>"),
    ("Slope across the IV curve.", "IV 曲线斜率。"),
    ("(rising IV with DTE) = normal/complacent.", "（IV 随 DTE 上升）= 正常/松懈。"),
    ("(falling) = market pricing immediate event risk. Use Front−Far &gt;5 pts as stress threshold.",
     "（下降）= 市场在定价短期事件风险。Front−Far &gt;5 pts 视为压力阈值。"),
    ("<b>Backwardation</b>", "<b>倒挂（Backwardation）</b>"),
    ("<b>Mild backwardation</b>", "<b>轻度倒挂</b>"),
    ("<b>Steep contango</b>", "<b>陡峭正向（Contango）</b>"),
    ("<b>Normal contango</b>", "<b>正常正向</b>"),
    ("<b>Flat term structure</b>", "<b>平坦期限结构</b>"),
    ("Front (≤7d)", "前周 (≤7天)"),
    ("2W (8-14d)", "2周 (8-14天)"),
    ("1M (15-35d)", "1月 (15-35天)"),
    ("2M (36-70d)", "2月 (36-70天)"),
    ("Far (>70d)", "远期 (>70天)"),
    ("— Front IV (", "— 前周 IV ("),
    ("%) above Far (", "%) 高于远期 ("),
    ("%) well below Far (", "%) 远低于远期 ("),
    ("%) by ", "%)，差值 "),
    (" pts. Market pricing near-term event/risk; far-dated options relatively 'cheap'.",
     " pts。市场定价短期事件/风险；远期合约相对便宜。"),
    ("— Front IV slightly above Far (", "— 前周 IV 略高于远期 ("),
    (" pts). Some near-term anxiety; not yet stress regime.", " pts）。短期有些紧张，但未达压力区。"),
    (" pts. Market complacent now, pricing uncertainty over time. Front puts may be cheap insurance.",
     " pts。市场目前松懈，长期才定价不确定性。前周 puts 可能是便宜的保险。"),
    ("— Front below Far (", "— 前周低于远期 ("),
    (" pts). Healthy term structure; nothing imminent priced in.",
     " pts）。期限结构健康；无短期事件被定价。"),
    ("— IV roughly uniform across DTE (", "— IV 在各 DTE 大致均匀 ("),
    (" pts). Neither stress nor complacency.", " pts）。既无压力也无松懈。"),
    ("Position Changes (OI Delta)", "持仓变动（OI Delta）"),
    (">Persistent Volume Strikes — Call Wall / Put Wall Detection</h2>", ">持续放量行权价 — Call/Put Wall 识别</h2>"),
    (">Expected Move Accuracy (backtest)</h2>", ">Expected Move 准确性（回测）</h2>"),
    # ===== KPI labels =====
    (">Spot</div>", ">现价</div>"),
    (">Front P/C Spread</div>", ">前周 P/C 偏度</div>"),
    (">BTC 24h</div>", ">BTC 24小时</div>"),
    (">BTC 7d</div>", ">BTC 7日</div>"),
    (">MSTR 24h</div>", ">MSTR 24小时</div>"),
    (">Correlation (", ">相关性 ("),
    (">MSTR β to BTC</div>", ">MSTR 对 BTC β</div>"),
    (">IV %ile (2yr)</div>", ">IV 分位数 (2年)</div>"),
    # ===== Table headers =====
    ("<th>Strike</th>", "<th>行权价</th>"),
    ("<th>Magnitude</th>", "<th>幅度</th>"),
    ("<th style=\"width:45%;\">Magnitude</th>", "<th style=\"width:45%;\">幅度</th>"),
    ("<th>Call OI Δ</th>", "<th>Call OI 变动</th>"),
    ("<th>Put OI Δ</th>", "<th>Put OI 变动</th>"),
    ("<th style=\"text-align:left;\">Strike</th>", "<th style=\"text-align:left;\">行权价</th>"),
    ("<th style=\"text-align:left;\">Type</th>", "<th style=\"text-align:left;\">类型</th>"),
    ("<th>Active Days</th>", "<th>活跃天数</th>"),
    ("<th>Total Vol</th>", "<th>总成交</th>"),
    ("<th>Avg/Day</th>", "<th>日均</th>"),
    ("<th style=\"text-align:right;\">Put OI Δ</th>", "<th style=\"text-align:right;\">Put OI 变动</th>"),
    ("<th style=\"text-align:left;\">Call OI Δ</th>", "<th style=\"text-align:left;\">Call OI 变动</th>"),
    ("<th style=\"text-align:center;\">Strike</th>", "<th style=\"text-align:center;\">行权价</th>"),
    ("▲ ABOVE SPOT", "▲ 高于现价"),
    ("▼ BELOW SPOT", "▼ 低于现价"),
    ("Avg Vol · Active", "日均 · 活跃"),
    ("No persistent strikes above spot", "现价上方无持续放量行权价"),
    ("No persistent strikes below spot", "现价下方无持续放量行权价"),
    (">Spot $", ">现价 $"),
    ("PUT Avg Vol · Active", "PUT 日均 · 活跃"),
    ("CALL Avg Vol · Active", "CALL 日均 · 活跃"),
    ("Strike · OTM", "行权价 · OTM"),
    ("No strikes above spot", "现价上方无数据"),
    ("No strikes below spot", "现价下方无数据"),
    ("Spot</span>", "现价</span>"),
    ("<th style=\"text-align:left;\">Date</th>", "<th style=\"text-align:left;\">日期</th>"),
    ("<th>Next Spot</th>", "<th>次日价</th>"),
    ("<th>EM Range</th>", "<th>EM 区间</th>"),
    ("<th>Actual Move</th>", "<th>实际波动</th>"),
    ("<th>Within EM?</th>", "<th>落在 EM 内？</th>"),
    ("<th style=\"text-align:left;\">Move</th>", "<th style=\"text-align:left;\">移动</th>"),
    ("<th style=\"text-align:left;\">Range</th>", "<th style=\"text-align:left;\">区间</th>"),
    ("<th style=\"text-align:left;\">Within EM?</th>", "<th style=\"text-align:left;\">落在 EM 内？</th>"),
    # ===== Skew regime labels =====
    ("EXTREME_CALL_SKEW", "极端看涨偏度"),
    ("ELEVATED_PUT_SKEW", "看跌偏度升高"),
    ("CALL_SKEW", "看涨偏度"),
    ("PUT_SKEW", "看跌偏度"),
    # ===== GEX wall card (long static blocks first) =====
    ("Strikes ranked by <b style=\"color:#e6edf3;\">Σ (gamma × OI × 100)</b> across all expirations — measures where <b style=\"color:#e6edf3;\">dealer hedging flow</b> concentrates. More accurate than volume-weighted walls for identifying actual resistance/support.",
     "按 <b style=\"color:#e6edf3;\">Σ (gamma × OI × 100)</b> 汇总所有到期的行权价 — 衡量<b style=\"color:#e6edf3;\">做市商对冲流</b>集中处。相比成交量墙更能反映实际支撑/阻力。"),
    ("Net GEX:", "净 GEX:"),
    ("Call walls (resistance above ", "Call 墙（阻力位，高于 "),
    ("Put walls (support below ", "Put 墙（支撑位，低于 "),
    ("Strike · OTM", "行权价 · OTM"),
    ("▲ CALL WALLS (resistance)", "▲ CALL 墙（阻力位）"),
    ("▼ PUT WALLS (support)", "▼ PUT 墙（支撑位）"),
    (">Spot<", ">现价<"),
    ("No call walls within 25% OTM", "25% OTM 内无 Call 墙"),
    ("No put walls within 25% OTM", "25% OTM 内无 Put 墙"),
    ("No call walls within 25% OTM", "25% OTM 内无 Call 墙"),
    ("No put walls within 25% OTM", "25% OTM 内无 Put 墙"),
    ("How GEX walls differ from volume walls (click)", "GEX 墙 vs 成交量墙（点击）"),
    ("<b style=\"color:#e6edf3;\">Volume walls</b> (below, Persistent Volume Strikes) show where retail/institutional <b>flow concentrates</b> — backward-looking conviction.",
     "<b style=\"color:#e6edf3;\">成交量墙</b>（下方持续放量行权价）显示散户/机构<b>流向集中点</b> — 后视镜信号。"),
    ("<b style=\"color:#e6edf3;\">GEX walls</b> (this card) show where <b>dealer hedging pressure</b> is largest — forward-looking, drives actual intraday price behavior.",
     "<b style=\"color:#e6edf3;\">GEX 墙</b>（本卡）显示<b>做市商对冲压力</b>最大处 — 前瞻性，驱动真实盘中价格行为。"),
    ("A strike can be a volume wall without being a GEX wall (e.g. far-dated OI has low gamma). The <b>GEX wall is what actually acts as resistance/support intraday</b> because dealers must trade stock to hedge. When they agree, the wall is robust. When they disagree, trust GEX.",
     "成交量墙不一定是 GEX 墙（例如远期 OI gamma 低）。<b>盘中真正起支撑/阻力作用的是 GEX 墙</b>，因为做市商必须交易股票来对冲。两者一致 = 墙稳固；不一致 = 信 GEX。"),
    # ===== GEX regime banner =====
    ("<b>Dealers net long gamma</b> — price acts as <b>magnet</b>. Moves get suppressed; intraday chop inside the wall band is likely. Favorable for premium selling (CC/CSP) — positions pin toward high-gamma strikes.",
     "<b>做市商净多 gamma</b> — 价格<b>磁吸</b>。波动被压制，盘中在墙之间震荡概率大。利于卖权（CC/CSP）— 仓位被钉向高 gamma 行权价。"),
    ("<b>Dealers net short gamma</b> — moves get <b>amplified</b>. Breakouts become squeezes, breakdowns become crashes. <b>Dangerous for premium selling</b>; consider straddles/long-vol. Spot crossing a major wall triggers mechanical hedging (chase rally / panic dump).",
     "<b>做市商净空 gamma</b> — 波动被<b>放大</b>。突破变 squeeze，跌破变 crash。<b>卖权危险</b>，考虑跨式/做多波动率。价格穿过关键墙触发机械对冲（追涨 / 恐慌抛售）。"),
    ("<b>Mixed gamma regime</b> — calls and puts roughly balanced. Spot behavior neither magnetic nor explosive; walls are soft.",
     "<b>混合 gamma 环境</b> — call 与 put 大致平衡。价格既不磁吸也不爆发，墙偏软。"),
    # ===== BTC context card =====
    ("MSTR's ~", "MSTR 约 "),
    (" beta to BTC means you must evaluate MSTR flow <b style=\"color:#e6edf3;\">relative to BTC</b>. Bullish MSTR options on a flat-BTC day = premium (historically fades). BTC ripping but MSTR calm = discount (historically catches up).",
     " 的 BTC beta 意味着 MSTR 流向必须<b style=\"color:#e6edf3;\">相对 BTC</b>看。BTC 平而 MSTR 看涨期权火 = 溢价（历史上会回落）；BTC 飙而 MSTR 平 = 折价（历史上会追上）。"),
    ("<b>MSTR trading at premium vs BTC-implied</b>", "<b>MSTR 相对 BTC 隐含价溢价</b>"),
    (" — expected ~", " — 按 β 预期约 "),
    (" based on β, actual ", "，实际 "),
    (" (residual ", "（残差 "),
    ("Short-term premium tends to mean-revert; if BTC stays flat, MSTR likely gives back the excess. Consider fading extreme premiums or avoiding fresh longs.",
     "短期溢价倾向均值回归；若 BTC 持平，MSTR 大概率吐出超额。考虑反向极端溢价或避免新建多头。"),
    ("<b>MSTR trading at discount vs BTC-implied</b>", "<b>MSTR 相对 BTC 隐含价折价</b>"),
    ("Relative weakness; could be tax-loss selling, dilution, or idiosyncratic. Mean-reversion bias bullish if BTC holds.",
     "相对弱势；可能是 tax-loss 抛售、稀释或公司特殊因素。BTC 企稳则均值回归偏多。"),
    ("<b>No MSTR/BTC divergence</b> — MSTR move is consistent with BTC move × beta. Options flow can be taken at face value without premium/discount overlay.",
     "<b>MSTR/BTC 无背离</b> — MSTR 走势与 BTC × β 一致。期权流向可直接采信，无需溢价/折价修正。"),
    # ===== Zero-crossing card intro =====
    ("Historical pattern: when delta-25 P/C IV spread breaks below zero, stock often near a local top within 1-3 weeks.",
     "历史规律：delta-25 P/C IV spread 跌破 0 时，股票往往在 1-3 周内见到阶段顶。"),
    # ===== P/C spread chart legend =====
    ("<b style=\"color:#e6edf3;\">Spread = Put IV − Call IV</b> at delta-25 strikes. Positive = puts more expensive (normal hedging demand). Negative = calls more expensive (speculative FOMO).",
     "<b style=\"color:#e6edf3;\">偏度 = Put IV − Call IV</b>（delta-25 行权价）。正 = put 贵（正常对冲需求）；负 = call 贵（投机 FOMO）。"),
    ("<b>Safe zone</b> (spread &gt; 0) — puts cost more than calls, market pricing in downside risk = healthy",
     "<b>安全区</b>（偏度 > 0）— put 贵于 call，市场定价下行风险 = 健康"),
    ("<b>Danger zone</b> (spread &lt; 0) — calls cost more than puts, investors chasing upside = top signal",
     "<b>危险区</b>（偏度 < 0）— call 贵于 put，投资者追涨 = 顶部信号"),
    ("Zero line (threshold)", "零轴（临界线）"),
    ("Spot price (right axis)", "现价（右轴）"),
    # ===== NVRP insight =====
    ("<b style=\"color:#e6edf3;\">NVRP = ATM IV / RV30.</b> The 1.3 threshold exists because sellers need ~30% cushion to cover bid-ask spread, gamma risk, and hedging error. &lt;1.0 = long-vol edge; 1.0-1.3 = marginal; &gt;1.3 = short-vol edge; &gt;1.5 = strong edge.",
     "<b style=\"color:#e6edf3;\">NVRP = ATM IV / RV30</b>。1.3 门槛：卖方需 ~30% 缓冲来覆盖买卖价差、gamma 风险和对冲误差。<1.0 = 做多波动率占优；1.0-1.3 = 边际；>1.3 = 做空波动率占优；>1.5 = 强优势。"),
    ("<b>Strong premium-selling edge</b>", "<b>卖权溢价优势明显</b>"),
    ("x (IV ", "x（IV "),
    ("%, +", "%，IV 高 "),
    ("%). Clear edge for CC/CSP/wheel after covering friction costs.",
     "%）。扣除摩擦成本后，CC/CSP/wheel 明显占优。"),
    # Legacy key — still referenced by other insight chains
    ("Clear edge for CC/CSP/wheel after covering friction costs.",
     "扣除摩擦成本后，CC/CSP/wheel 明显占优。"),
    ("<b>Premium-selling edge present</b>", "<b>卖权有边际优势</b>"),
    ("above 1.3 threshold. IV (", "超过 1.3 门槛。IV ("),
    (") exceeds RV (", ") 超过实际波动率 ("),
    (") enough to cover typical friction costs.", ") 足以覆盖典型摩擦成本。"),
    ("<b>Marginal / no edge</b>", "<b>优势边际 / 无优势</b>"),
    ("between 1.0-1.3. IV (", "在 1.0-1.3 区间。IV ("),
    (") only slightly above RV (", ") 仅略高于实际波动率 ("),
    ("); premium may not cover bid-ask + gamma risk. Avoid aggressive short vol.",
     "），溢价可能不足以覆盖买卖价差 + gamma 风险。避免激进做空波动率。"),
    ("<b>Long vol edge</b>", "<b>做多波动率占优</b>"),
    ("below 1.0. IV (", "低于 1.0。IV ("),
    (") cheaper than RV (", ") 比实际波动率 ("),
    (") — options underpriced. Consider BUYING vol (straddles/calls/puts) not selling.",
     ") 便宜 — 期权定价不足。考虑买波动率（跨式/calls/puts）而非卖。"),
    (" IV %ile (2yr):", " IV 分位数 (2年):"),
    # ===== Skew percentile banner =====
    ("<b>Extreme call skew — bottom ", "<b>极端看涨偏度 — 底部 "),
    ("<b>Heavy call skew — ", "<b>看涨偏度偏重 — "),
    ("<b>Heavy put skew — ", "<b>看跌偏度偏重 — "),
    ("<b>Extreme put skew — top ", "<b>极端看跌偏度 — 顶部 "),
    ("<b>Skew within normal band</b>", "<b>偏度在正常区间</b>"),
    ("th %ile</b> of ", "分位数</b>（"),
    ("-day history (current ", "天历史，当前 "),
    (" vs median ", " vs 中位数 "),
    (", low ", "，最低 "),
    ("). When skew is this compressed, rally often near exhaustion — consider fading or hedging with cheap OTM puts.",
     "）。偏度压到这种程度，涨势往往接近衰竭 — 考虑反向或用便宜的 OTM put 对冲。"),
    ("). Bullish flow building; watch for deterioration toward extreme.",
     "）。多头流正在建仓；留意继续恶化至极端。"),
    ("). No extreme positioning signal.", "）。无极端仓位信号。"),
    ("). Fear bid on puts; historically a lagging bottom signal.",
     "）。对 put 的恐慌买盘；历史上是滞后底部信号。"),
    ("-day history. Capitulation / panic bid on puts; historically near local bottoms.",
     "天历史。恐慌性 put 买盘；历史上接近阶段底。"),
    (" only ", " 仅 "),
    (" days of history — percentile unreliable until n≥60", " 天历史 — 分位数在 n≥60 前不可靠"),
    # ===== Call DTE regime banner =====
    ("<b>Speculative call build — ", "<b>投机性 Call 建仓 — "),
    ("% short-dated (≤14d)</b>", "% 短期 (≤14天)</b>"),
    (". Call OI flooding near-term expirations (+", "。Call OI 涌入近端到期（+"),
    (") while mid/long stay cold (+", "），中/长端冷清（+"),
    (" / +", " / +"),
    ("). Classic FOMO / catalyst-chase pattern — rally often unsustainable without long-dated conviction.",
     "）。典型 FOMO / 追催化剂模式 — 缺长期定投的涨幅通常不可持续。"),
    ("<b>Near-term lean — ", "<b>近端倾斜 — "),
    ("% short-dated</b>", "% 短期</b>"),
    (" (short +", "（短 +"),
    (" / mid +", " / 中 +"),
    (" / long +", " / 长 +"),
    ("). Short-dated bias but some mid-term participation. Watch whether long-dated follows.",
     "）。偏近端但中端有参与。观察长端是否跟进。"),
    ("<b>Structural bullish build — ", "<b>结构性多头建仓 — "),
    ("% long-dated (&gt;45d)</b>", "% 长期 (>45天)</b>"),
    ("). Conviction across the term structure — more sustainable than a short-dated-only squeeze.",
     "）。整条期限结构都有定投 — 比纯近端挤压更可持续。"),
    ("<b>Balanced call build</b>", "<b>Call 建仓均衡</b>"),
    ("). No strong DTE tilt either way.", "）。DTE 无明显偏向。"),
    ("<b>Net call OI unwind</b>", "<b>Call OI 净平仓</b>"),
    (" total). Positions being closed rather than opened.", " 合计）。仓位在平仓而非开仓。"),
    # ===== Spread insight headlines =====
    ("<b>EXTREME CALL SKEW</b> — all ", "<b>极端看涨偏度</b> — 全部 "),
    ("<b>Call skew across all buckets</b> — all ", "<b>所有到期桶都 call 偏</b> — 全部 "),
    ("<b>Partial call skew</b> — ", "<b>部分 call 偏</b> — "),
    ("<b>Mixed</b> — ", "<b>混合</b> — "),
    ("<b>Healthy skew</b> — all ", "<b>偏度健康</b> — 全部 "),
    (" buckets in danger zone (avg ", " 个到期桶在危险区（均值 "),
    (" pts, low ", " pts，最低 "),
    ("). Speculative FOMO across the full term structure. Historical pattern: local top within 1-3 weeks. Consider buying OTM puts as hedge while they're relatively cheap.",
     "）。整条期限结构都是投机 FOMO。历史规律：1-3 周内见阶段顶。考虑趁 OTM put 相对便宜时买入对冲。"),
    (" expirations negative (avg ", " 个到期为负（均值 "),
    (" pts). Bullish euphoria building; watch for further deterioration toward extreme levels.",
     " pts）。多头情绪升温；留意继续恶化至极端。"),
    (" buckets negative (low ", " 个到期桶为负（最低 "),
    ("). Speculative flow emerging in some expirations. Monitor if it spreads to all buckets.",
     "）。部分到期出现投机流，观察是否蔓延到所有桶。"),
    ("). Call skew localized to shorter dates; longer-dated expirations still in safe zone.",
     "）。Call 偏仅在近端；远端仍在安全区。"),
    (" buckets positive (avg ", " 个到期为正（均值 "),
    (" pts). Normal put-protection demand; no FOMO signal.",
     " pts）。正常 put 对冲需求，无 FOMO 信号。"),
    # ===== OI Delta insight + depth badges =====
    ("<b>Smart-money defensive posture</b> — ", "<b>Smart-money 防御姿态</b> — "),
    ("% of call OI growth is <b>ITM</b> (covered-call writers locking gains, not bullish bets), while ",
     "% 的 call OI 增长在 <b>ITM</b>（covered-call 锁利，非多头押注），同时 "),
    ("% of put OI growth is <b>deep OTM</b> (&lt;-15% spot = real tail hedges). Headline numbers hide this — true directional flow is <b>OTM calls ",
     "% 的 put OI 增长在<b>深度 OTM</b>（<-15% 现价 = 真正的尾部对冲）。表面总数会骗人 — 真实方向流是 <b>OTM calls "),
    (" vs OTM puts ", " vs OTM puts "),
    ("</b>. Institutions sell calls + buy puts = late-bull 'seatbelt-on' signal.",
     "</b>。机构卖 call + 买 put = 晚期牛市「系安全带」信号。"),
    ("<b>ITM call writing dominates</b> — ", "<b>ITM call 写卖占主导</b> — "),
    ("% of call OI growth is ITM. Likely covered-call writers taking profit, not bullish accumulation. Strip ITM to read real flow: OTM calls ",
     "% 的 call OI 增长在 ITM。大概率是 covered-call 获利而非多头建仓。剥离 ITM 看真实流向：OTM calls "),
    ("<b>Heavy tail-hedge buying</b> — ", "<b>重仓尾部对冲买入</b> — "),
    ("% of put OI growth is deep OTM (&lt;-15% spot, strikes ", "% 的 put OI 增长在深度 OTM（<-15% 现价，行权价 "),
    ("). Systematic insurance demand = real downside fear.", "）。系统性保险需求 = 真实下行恐慌。"),
    ("<b>Clean bullish accumulation</b> — OTM call adds ", "<b>纯多头建仓</b> — OTM call 新增 "),
    (" dominate; OTM put flow ", " 占主导；OTM put 流 "),
    (" is minor. Directional upside bets, minimal hedging.",
     " 较小。方向性上行押注，对冲很少。"),
    ("<b>Directional put buying</b> — OTM put adds ", "<b>方向性 put 买入</b> — OTM put 新增 "),
    (" dwarf call additions ", " 远超 call 新增 "),
    (". Clear bearish positioning.", "。明确空头仓位。"),
    ("<b>Mixed flow</b> — OTM calls ", "<b>混合流</b> — OTM calls "),
    (", OTM puts ", "，OTM puts "),
    (", ITM calls ", "，ITM calls "),
    (". No clear directional signal.", "。无明确方向信号。"),
    # OI depth badges
    ("OTM Call (&gt;spot)", "OTM Call (>现价)"),
    ("ITM Call (&lt;spot)", "ITM Call (<现价)"),
    ("OTM Put (&lt;spot)", "OTM Put (<现价)"),
    ("Deep OTM Put (&lt;-15%)", "深度 OTM Put (<-15%)"),
    ("upside bets", "上行押注"),
    ("often covered-call writing", "常为 covered-call 写卖"),
    ("downside / hedges", "下行 / 对冲"),
    ("tail insurance", "尾部保险"),
    # ===== OI Delta depth help =====
    ("How to read OI Delta by depth (click to expand)", "如何按深度读 OI Delta（点击展开）"),
    ("OI Delta = <b>net new contracts created</b> between snapshots (buyers + sellers both open). The headline total is misleading — <b>where the OI lands matters more than the total</b>.",
     "OI Delta = 两次 snapshot 之间<b>净新建合约</b>（买卖双方都开仓）。表面总数有误导性 — <b>OI 落在哪里比总量更重要</b>。"),
    ("<b style=\"color:#3fb950;\">OTM Call (above spot)</b> = directional upside bets. Retail FOMO or institutional call buying.",
     "<b style=\"color:#3fb950;\">OTM Call（现价之上）</b> = 方向性上行押注。散户 FOMO 或机构买 call。"),
    ("<b style=\"color:#d29922;\">ITM Call (below spot)</b> = usually <b>covered-call writers</b> locking gains on existing stock. Increases OI but is <b>not bullish</b> — it's profit-taking.",
     "<b style=\"color:#d29922;\">ITM Call（现价之下）</b> = 通常是 <b>covered-call 写卖方</b>对已持股锁利。OI 增加但<b>并非看涨</b> — 是获利了结。"),
    ("<b style=\"color:#f85149;\">OTM Put (below spot)</b> = downside bets or portfolio hedges.",
     "<b style=\"color:#f85149;\">OTM Put（现价之下）</b> = 下行押注或组合对冲。"),
    ("<b style=\"color:#f85149;\">Deep OTM Put (&lt;-15% spot)</b> = systematic tail insurance. Only bought by institutions/macro funds. When this surges, <b>smart money is nervous</b>.",
     "<b style=\"color:#f85149;\">深度 OTM Put（<-15% 现价）</b> = 系统性尾部保险。仅机构/宏观基金购买。激增 = <b>smart money 紧张</b>。"),
    ("<b>Classic patterns:</b>", "<b>经典模式：</b>"),
    ("<b>Pure bullish:</b> OTM calls dominate, puts quiet → clean upside conviction",
     "<b>纯多头：</b>OTM calls 占主导，puts 安静 → 纯上行定投"),
    ("<b>Pure bearish:</b> OTM puts dominate, calls quiet → clean downside conviction",
     "<b>纯空头：</b>OTM puts 占主导，calls 安静 → 纯下行定投"),
    ("<b>\"Seatbelt on\":</b> ITM calls + deep OTM puts both swell → institutions taking profit + hedging = late-bull warning",
     "<b>「系安全带」：</b>ITM calls + 深度 OTM puts 同时膨胀 → 机构获利 + 对冲 = 晚期牛市警告"),
    ("<b>Rotation:</b> ITM calls closing (OI drops) + OTM calls opening (OI rises) → long positions being rolled up after a rally",
     "<b>轮动：</b>ITM calls 平仓（OI 下降）+ OTM calls 开仓（OI 上升）→ 多头仓位在涨后向上 roll"),
    # ===== Volume persistence card =====
    ("Strikes with heavy volume across <b>multiple days</b> = structural positioning (dealers守在这里), not one-off flow.",
     "在<b>多日</b>保持高成交的行权价 = 结构性仓位（做市商守在这里），非一次性流。"),
    ("How walls are identified (click to expand)", "墙如何识别（点击展开）"),
    ("<b style=\"color:#3fb950;\">Call Wall</b> — OTM call strike (3–15% above spot) with highest <code>persistence × avg daily volume</code>. Dealers are short these calls → <b>short gamma</b> → hedging buy-pressure exhausts near this strike → price stalls (ceiling / resistance).",
     "<b style=\"color:#3fb950;\">Call 墙</b> — OTM call 行权价（现价上方 3–15%），<code>持续性 × 日均成交</code> 最高。做市商 short 这些 call → <b>short gamma</b> → 接近此位时对冲买压耗尽 → 价格停滞（天花板/阻力）。"),
    ("<b style=\"color:#f85149;\">Put Wall</b> — OTM put strike (3–15% below spot) with highest score. Dealers short these puts → auto-buy on dips near strike → support.",
     "<b style=\"color:#f85149;\">Put 墙</b> — OTM put 行权价（现价下方 3–15%），得分最高。做市商 short 这些 put → 接近此位自动买入 → 支撑。"),
    ("<b>Sweet spot: 3–10% OTM</b>", "<b>最佳区间：3–10% OTM</b>"),
    (". Too close = magnet (gets crossed). Too far = lottery / breakout target, not active resistance.",
     "。太近 = 磁吸（会被穿过）；太远 = 彩票 / 突破目标，非真实阻力。"),
    ("<b>Strength:</b> 85%+ persistence = ", "<b>强度：</b>持续性 85%+ = "),
    ("strong (structural)", "强（结构性）"),
    (", 65–85% = ", "，65–85% = "),
    ("moderate", "中等"),
    (", 50–65% = ", "，50–65% = "),
    ("weak", "弱"),
    ("<b>Trade rules:</b> Sell CC at call wall · Sell CSP at put wall · Don't buy lottos above call wall · Watch wall migration (up = bullish re-rating, down = top signal, thinning = resistance weakening).",
     "<b>交易规则：</b>在 call 墙卖 CC · 在 put 墙卖 CSP · 别买 call 墙上方的彩票 · 观察墙的迁移（向上 = 看涨重定价，向下 = 顶部信号，变稀 = 阻力减弱）。"),
    ("<b>⚠️ Asymmetry warning:</b> Dense call ladder + thin put side = late-bull euphoria (no hedging demand, market complacent).",
     "<b>⚠️ 不对称警告：</b>密集 call 阶梯 + 稀薄 put 端 = 晚期牛市狂热（无对冲需求，市场自满）。"),
    # ===== Wall insight headlines =====
    ("<b>Asymmetric: call-heavy, put-thin</b>", "<b>不对称：call 重、put 稀</b>"),
    (" — call side total volume is ", " — call 端总成交量是 "),
    ("× the put side</b>", "× put 端</b>"),
    (". Dense call ladder + thin put support = <b>late-bull euphoria</b> (market complacent, no structural hedging). Expect upside to stall near call wall; downside has little cushion if sentiment flips.",
     "。密集 call 阶梯 + 稀薄 put 支撑 = <b>晚期牛市狂热</b>（市场自满、无结构对冲）。上行到 call 墙大概率停滞；情绪翻转时下行缓冲很少。"),
    ("<b>Balanced two-sided positioning</b>", "<b>双向持仓均衡</b>"),
    (" — both walls structural ", " — 两道墙都结构性 "),
    (" (call ", "（call "),
    (", put ", "，put "),
    ("). Expect range-bound trade between the walls. Good regime for selling strangles / iron condors.",
     "）。预期在墙之间区间震荡。适合卖勒式 / 铁鹰。"),
    ("<b>Put-heavy / call-light</b>", "<b>Put 重 / Call 轻</b>"),
    (" — heavy put structure but no call resistance. Often seen after a washout — upside path open if market stabilizes.",
     " — put 结构重但无 call 阻力。常见于洗盘之后 — 市场企稳则上行通道打开。"),
    ("<b>No structural walls detected</b>", "<b>未检测到结构性墙</b>"),
    (" — no strike has persistence ≥50% in the valid zone. Could be early in the data window, or positioning is rotating fast. Read with caution.",
     " — 有效区间内无行权价持续性 ≥50%。可能是数据窗口尚早，或仓位快速轮动。谨慎解读。"),
    ("<b>Mixed positioning</b>", "<b>混合持仓</b>"),
    (", call/put vol ratio ", "，call/put 成交比 "),
    ("×. No dominant structural theme.", "×。无主导结构性主题。"),
    ("Spot within ", "现价距"),
    ("% of call wall → imminent resistance test", "% 的 call 墙 → 即将测试阻力"),
    ("% of put support → support test", "% 的 put 支撑 → 测试支撑"),
    ("Put support $", "Put 支撑 $"),
    ("Call resistance $", "Call 阻力 $"),
    ("Implied range:", "隐含区间:"),
    ("% downside · ", "% 下行 · "),
    ("% upside)", "% 上行)"),
    # ===== EM accuracy card =====
    ("Did the front-month Expected Move correctly contain the next day's price? EM scaled by √(calendar days) for weekend gaps. <b>Actual/EM</b> ratio: 1.0 = perfect, &lt;1 = EM over-predicts, &gt;1 = under-predicts.",
     "前月 Expected Move 是否框住次日价格？EM 按 √(日历天数) 缩放以处理周末跳空。<b>Actual/EM</b> 比率：1.0 = 完美；<1 = EM 高估；>1 = EM 低估。"),
    ("Hit rate", "命中率"),
    ("vs 68% expected", "vs 预期 68%"),
    ("Avg |move|", "平均 |波动|"),
    ("Bias (median)", "偏差（中位数）"),
    ("calibrated", "已校准"),
    ("over-predicts", "高估"),
    ("under-predicts", "低估"),
    (" outlier", " 异常值"),
    ("s (&gt;2× EM)", "（>2× EM）"),
    (" (&gt;2× EM)", "（>2× EM）"),
    ("Max miss", "最大偏离"),
    ("Miss direction", "偏离方向"),
    ("Sample size n=", "样本数 n="),
    (" &lt; 5 — stats are unreliable. Need more daily snapshots for meaningful hit rate.",
     " < 5 — 统计不可靠。需要更多每日 snapshot 才能得到有意义的命中率。"),
    ("<b>Too few samples to judge</b>", "<b>样本太少无法判断</b>"),
    (" prediction", " 次预测"),
    (" observed. Need 5+ for a reliable read. Interpret trends qualitatively below.",
     " 已观察。需要 5+ 才可靠。以下趋势按定性解读。"),
    ("<b>Tail-event regime</b>", "<b>尾部事件环境</b>"),
    (" moves exceeded 2× EM. Vol regime has broken out of normal range; EM is not containing shocks. Size positions smaller, widen stops.",
     " 次波动超过 2× EM。波动率环境已突破正常区间；EM 框不住冲击。仓位减小，止损放宽。"),
    ("<b>EM is conservative (too wide)</b>", "<b>EM 偏保守（太宽）</b>"),
    (" — hit rate ", " — 命中率 "),
    ("% exceeds expected 68%. Option IV priced at premium to realized; premium-selling has edge.",
     "% 超过预期 68%。期权 IV 相对实际贵；卖权有优势。"),
    ("<b>EM too narrow</b>", "<b>EM 太窄</b>"),
    ("% below 68% benchmark.", "% 低于 68% 基准。"),
    (" Most misses to ", " 多数偏离在"),
    ("upside", "上行"),
    ("downside", "下行"),
    ("both sides", "两侧"),
    (" Realized vol running higher than implied; buying options has edge, sell premium with caution.",
     " 实际波动高于隐含；买期权有优势，卖权需谨慎。"),
    ("<b>Well-calibrated</b>", "<b>校准良好</b>"),
    (" — median actual/EM ratio ", " — 中位数 actual/EM 比 "),
    ("x within [0.8, 1.2]. EM predicting real moves accurately; use it as a trading range estimator.",
     "x 在 [0.8, 1.2]。EM 预测准确，可作区间估算。"),
    (", median bias ", "，中位数偏差 "),
    ("x. No strong regime signal.", "x。无强烈环境信号。"),
    ("✓ Yes", "✓ 是"),
    ("✗ No", "✗ 否"),
    ("OUTLIER", "异常"),
    # ===== Footer =====
    ("Options flow trend analysis — for educational/research purposes only. Not financial advice.",
     "期权流向趋势分析 — 仅供教育/研究。不构成投资建议。"),
    ("Data from yfinance. OI may be stale. Volume-based signals are more reliable.",
     "数据来自 yfinance。OI 可能滞后。基于成交量的信号更可靠。"),
    ("snapshots |", "个 snapshot |"),
    # ===== Zero-crossing alert messages =====
    ("P/C spread crossed below zero in ", "P/C 偏度跌破 0（"),
    ("P/C spread normalized back above zero in ", "P/C 偏度回升至 0 之上（"),
    (" bucket: ", " 到期）："),
    (" pts. Historical pattern: call skew dominance often signals near-term top.",
     " pts。历史规律：call 偏度主导常预示近期顶。"),
    (" pts. Bullish euphoria fading.", " pts。看涨狂热消退。"),
    (": P/C spread ", "：P/C 偏度 "),
    ("No zero-crossings detected in this period.", "本期间未检测到零轴穿越。"),
    ("Current spread by bucket:", "各到期桶当前 spread："),
    (">now ", ">现 "),
    ("no crossings in window", "本窗口无穿越"),
    # ===== OI Delta fallbacks =====
    ("Net OI Change: Call ", "OI 净变动：Call "),
    (" | Put ", " | Put "),
    (" | Spot $", " | 现价 $"),
    ("Need 2+ snapshots to compute OI changes.", "需要 2+ 个 snapshot 才能计算 OI 变动。"),
    ("Need 2+ snapshots", "需要 2+ 个 snapshot"),
]


def apply_zh(html):
    """Best-effort English → Chinese post-processing on rendered dashboard HTML."""
    for en, zh in ZH_REPLACEMENTS:
        html = html.replace(en, zh)
    return html


def _mode_banner(mode):
    """Top-of-dashboard banner explaining what to focus on in the current session."""
    if mode == 'preopen':
        return """
<div style="background:linear-gradient(90deg, rgba(88,166,255,0.15), rgba(88,166,255,0.03));
            border:1px solid #58a6ff; border-left:4px solid #58a6ff;
            border-radius:8px; padding:14px 18px; margin:12px 0 20px;">
  <div style="color:#58a6ff; font-weight:600; font-size:15px; margin-bottom:6px;">
    PRE-OPEN MODE &mdash; build today's playbook
  </div>
  <div style="color:#c9d1d9; font-size:13px; line-height:1.6;">
    Focus on: <b style="color:#e6edf3;">Call/Put Walls</b> (price anchors &mdash; set limit orders near them),
    <b style="color:#e6edf3;">Persistent Volume</b> (multi-day S/R levels),
    <b style="color:#e6edf3;">OI Delta</b> (overnight smart-money positioning),
    <b style="color:#e6edf3;">NVRP</b> (premium-selling edge).
    Read once, write plan, then stop reading &mdash; data is static until tomorrow's snapshot.
  </div>
</div>"""
    if mode == 'intraday':
        return """
<div style="background:linear-gradient(90deg, rgba(240,136,62,0.15), rgba(240,136,62,0.03));
            border:1px solid #f0883e; border-left:4px solid #f0883e;
            border-radius:8px; padding:14px 18px; margin:12px 0 20px;">
  <div style="color:#f0883e; font-weight:600; font-size:15px; margin-bottom:6px;">
    INTRADAY MODE &mdash; this dashboard is static
  </div>
  <div style="color:#c9d1d9; font-size:13px; line-height:1.6;">
    Data is a <b style="color:#e6edf3;">pre-open snapshot</b> (yesterday's close OI + 24-day history) and does NOT update during the session.
    Use real-time options chain / time &amp; sales for live flow. Only come back here if price is testing a Wall or Persistent strike.
  </div>
</div>"""
    if mode == 'postclose':
        return """
<div style="background:linear-gradient(90deg, rgba(63,185,80,0.15), rgba(63,185,80,0.03));
            border:1px solid #3fb950; border-left:4px solid #3fb950;
            border-radius:8px; padding:14px 18px; margin:12px 0 20px;">
  <div style="color:#3fb950; font-weight:600; font-size:15px; margin-bottom:6px;">
    POST-CLOSE MODE &mdash; review &amp; validate
  </div>
  <div style="color:#c9d1d9; font-size:13px; line-height:1.6;">
    Focus on: <b style="color:#e6edf3;">EM Accuracy</b> (did today land inside the envelope?),
    <b style="color:#e6edf3;">OI Delta</b> (did new positions validate yesterday's signal?),
    <b style="color:#e6edf3;">Walls</b> (tested / broken / held?).
    Feed findings into your trade journal and tomorrow's plan.
  </div>
</div>"""
    return ""


def generate_html(ticker, trend_data, output_path, mode='auto', lang='en'):
    """Generate an interactive trend dashboard."""
    # Nav: derive date from output filename, build cross-links
    fname = os.path.basename(output_path)
    m = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
    report_date = m.group(1) if m else datetime.now().strftime('%Y-%m-%d')
    lang_suffix = '_zh' if lang == 'zh' else ''
    if lang == 'zh':
        nav_lbl_kpi, nav_lbl_skew, nav_lbl_term, nav_lbl_walls = '概览', '偏度', '期限', '墙'
    else:
        nav_lbl_kpi, nav_lbl_skew, nav_lbl_term, nav_lbl_walls = 'KPIs', 'Skew', 'Term', 'Walls'

    pc = trend_data['pc_spread_series']
    crossings = trend_data['zero_crossings']
    iv = trend_data['iv_trend']
    oi = trend_data['oi_deltas']
    persist = trend_data['volume_persistence']
    em = trend_data['em_accuracy']
    current = trend_data['current_state']
    skew_pct = trend_data.get('skew_percentile')
    call_dte = trend_data.get('call_dte_ratio')
    gex_data = trend_data.get('gex_walls')
    btc_data = trend_data.get('btc_context')

    # Load MSTR BTC holdings (only used for mNAV card; missing file is non-fatal)
    holdings = None
    if ticker == 'MSTR':
        holdings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'mstr_holdings.json')
        try:
            with open(holdings_path) as _hf:
                holdings = json.load(_hf)
        except (FileNotFoundError, json.JSONDecodeError):
            holdings = None

    if mode == 'auto':
        mode = _detect_mode()
    mode_banner_html = _mode_banner(mode)

    # Build chart data for P/C spread time series
    # Group by bucket, average same-date multi-expiry points, use dates as x-axis
    spread_datasets = []
    colors = {'front': '#3fb950', '2w': '#f85149', '1m': '#d29922', '2m': '#58a6ff', 'far': '#a371f7'}
    for bucket in ['front', '2w', '1m', '2m', 'far']:
        if bucket not in pc:
            continue
        by_date = {}
        for p in pc[bucket]:
            by_date.setdefault(p['date'], []).append(p['spread'])
        dates = sorted(by_date.keys())
        values = [round(sum(by_date[d]) / len(by_date[d]), 2) for d in dates]
        spread_datasets.append({
            'label': bucket.upper(),
            'data': values,
            'dates': dates,
            'borderColor': colors.get(bucket, '#8b949e'),
        })

    # IV trend chart data
    iv_dates = [p['date'] for p in iv]
    iv_values = [round(p['front_atm_iv'], 1) for p in iv]
    rv_values = [round(p['rv30'], 1) for p in iv]
    nvrp_values = [p['nvrp'] for p in iv]
    spot_values = [round(p['spot'], 2) for p in iv]

    # OI delta data (latest day)
    oi_latest = oi[-1] if oi else None

    # Crossing alerts HTML — grouped by bucket, compact rows, current-state summary strip
    alert_html = ""
    if crossings:
        BUCKET_ORDER = ['front', '2w', '1m', '2m', 'far']
        by_bucket_cross = {}
        for c in crossings:
            by_bucket_cross.setdefault(c['bucket'], []).append(c)

        # Latest spread per bucket for the summary strip — avg across expiries on the latest date
        latest_state = {}
        for bucket, points in pc.items():
            if not points:
                continue
            latest_date = max(p['date'] for p in points)
            same_day = [p['spread'] for p in points if p['date'] == latest_date]
            latest_state[bucket] = sum(same_day) / len(same_day)

        pills = []
        for b in BUCKET_ORDER:
            if b not in latest_state:
                continue
            val = latest_state[b]
            col = '#f85149' if val < 0 else '#3fb950'
            pills.append(
                f'<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;'
                f'border-radius:999px;background:{col}18;border:1px solid {col}55;font-size:12px;">'
                f'<span style="color:#8b949e;font-weight:500;letter-spacing:0.3px;">{b.upper()}</span>'
                f'<span style="color:{col};font-weight:600;font-family:ui-monospace,SFMono-Regular,monospace;">{val:+.1f}</span>'
                f'</span>'
            )
        summary_html = ""
        if pills:
            summary_html = (
                '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:6px 0 14px;">'
                '<span style="color:#8b949e;font-size:11px;margin-right:2px;">Current spread by bucket:</span>'
                + ''.join(pills) + '</div>'
            )

        bucket_blocks = []
        for b in BUCKET_ORDER:
            if b not in latest_state:
                continue
            events = sorted(by_bucket_cross.get(b, []), key=lambda x: x['date'], reverse=True)
            cur = latest_state.get(b)
            cur_col = '#8b949e' if cur is None else ('#f85149' if cur < 0 else '#3fb950')
            cur_pill = ''
            if cur is not None:
                cur_pill = (
                    f'<span style="margin-left:auto;font-size:11px;color:{cur_col};'
                    f'font-family:ui-monospace,SFMono-Regular,monospace;font-weight:600;">'
                    f'now {cur:+.1f}</span>'
                )
            rows = []
            for c in events:
                color = '#f85149' if c['type'] == 'bearish_signal' else '#3fb950'
                icon = "&#9888;" if c['type'] == 'bearish_signal' else "&#10003;"
                rows.append(
                    f'<div style="display:flex;align-items:center;gap:8px;'
                    f'padding:5px 9px;border-left:3px solid {color};background:{color}0d;border-radius:4px;margin:3px 0;font-size:12px;">'
                    f'<span style="color:{color};flex:none;">{icon}</span>'
                    f'<span style="color:#8b949e;font-family:ui-monospace,SFMono-Regular,monospace;flex:none;">{c["date"][5:]}</span>'
                    f'<span style="color:#e6edf3;font-family:ui-monospace,SFMono-Regular,monospace;margin-left:auto;">'
                    f'{c["spread_from"]:+.1f} &rarr; <b style="color:{color};">{c["spread_to"]:+.1f}</b>'
                    f'</span></div>'
                )
            body = ''.join(rows) if rows else '<div style="color:#6e7681;font-size:11px;padding:6px 0 2px;font-style:italic;">no crossings in window</div>'
            bucket_blocks.append(
                '<div style="background:#0f1117;border:1px solid #21262d;border-radius:8px;padding:10px 12px;">'
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                f'<span style="color:#e6edf3;font-size:12px;font-weight:600;letter-spacing:0.5px;">{b.upper()}</span>'
                f'{cur_pill}</div>'
                + body + '</div>'
            )
        grid_html = (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;">'
            + ''.join(bucket_blocks) + '</div>'
        )
        alert_html = summary_html + grid_html
    else:
        alert_html = '<div style="color:#8b949e;padding:8px;">No zero-crossings detected in this period.</div>'

    # Current skew state
    current_front_spread = current.get('front_pc_spread', 0)
    skew_level = skew_alert_level(current_front_spread)
    skew_color_map = {
        'EXTREME_CALL_SKEW': '#f85149', 'CALL_SKEW': '#f0883e', 'MUTED': '#d29922',
        'NORMAL': '#3fb950', 'ELEVATED_PUT_SKEW': '#58a6ff', 'PANIC': '#f85149'
    }
    skew_color = skew_color_map.get(skew_level, '#8b949e')

    # Persistent volume table — merge calls+puts, sort by strike high-to-low, add bars
    persist_rows = ""
    all_persist = []
    for item in persist.get('call_persistent', [])[:12]:
        all_persist.append({**item, 'type': 'CALL'})
    for item in persist.get('put_persistent', [])[:12]:
        all_persist.append({**item, 'type': 'PUT'})
    all_persist.sort(key=lambda x: x['strike'], reverse=True)
    spot_val = current.get('spot', 0)
    max_vol = max((p['total_volume'] for p in all_persist), default=1) or 1

    # --- Identify Call Wall / Put Wall from persistent strikes ---
    def _find_wall(items, side, spot):
        cands = []
        for p in items:
            if p['type'] != side:
                continue
            strike = p['strike']
            if spot <= 0:
                continue
            if side == 'CALL' and strike <= spot:
                continue
            if side == 'PUT' and strike >= spot:
                continue
            pct = p['active_days'] / p['total_days'] if p['total_days'] > 0 else 0
            if pct < 0.5:
                continue
            otm = abs(strike - spot) / spot
            if otm > 0.15:
                continue
            avg_vol = p['total_volume'] / p['active_days'] if p['active_days'] > 0 else 0
            raw_score = pct * avg_vol
            score = raw_score
            magnet_penalty = False
            if 0.03 <= otm <= 0.10:
                score *= 1.5
            elif otm < 0.03:
                score *= 0.8
                magnet_penalty = True
            if strike % 10 == 0:
                score *= 1.15
            elif strike % 5 == 0:
                score *= 1.05
            cands.append({'score': score, 'raw_score': raw_score, 'strike': strike,
                          'pct': pct, 'avg_vol': avg_vol, 'otm': otm,
                          'active': p['active_days'], 'total': p['total_days'],
                          'magnet': magnet_penalty})
        if not cands:
            return None
        cands.sort(key=lambda x: x['score'], reverse=True)
        top = cands[0]
        # Band detection: if #2 is within 30% of #1 AND not a magnet, treat as band
        if len(cands) >= 2:
            runner = cands[1]
            if runner['score'] / top['score'] > 0.70 and not runner['magnet'] and not top['magnet']:
                lo_strike = min(top['strike'], runner['strike'])
                hi_strike = max(top['strike'], runner['strike'])
                avg_pct = (top['pct'] + runner['pct']) / 2
                combined_vol = top['avg_vol'] + runner['avg_vol']
                lo_otm = min(top['otm'], runner['otm'])
                hi_otm = max(top['otm'], runner['otm'])
                return {'is_band': True, 'lo': lo_strike, 'hi': hi_strike,
                        'pct': avg_pct, 'avg_vol': combined_vol,
                        'lo_otm': lo_otm, 'hi_otm': hi_otm,
                        'active': f"{top['active']}+{runner['active']}",
                        'total': top['total']}
        return {'is_band': False, **top}

    call_wall = _find_wall(all_persist, 'CALL', spot_val)
    put_wall = _find_wall(all_persist, 'PUT', spot_val)

    def _wall_badge(w, label, color):
        if not w:
            return f'<span style="padding:4px 10px;border-radius:6px;background:#21262d;color:#8b949e;font-size:12px;">{label}: n/a</span>'
        strength = 'strong' if w['pct'] >= 0.85 else 'moderate' if w['pct'] >= 0.65 else 'weak'
        strength_color = '#3fb950' if strength == 'strong' else '#d29922' if strength == 'moderate' else '#8b949e'
        sign = '+' if label == 'Call Wall' else '−'
        if w.get('is_band'):
            return (f'<span style="padding:4px 10px;border-radius:6px;background:{color}22;'
                    f'border:1px solid {color}66;color:{color};font-size:12px;font-weight:500;">'
                    f'{label} Band: <b>${w["lo"]:.0f}–${w["hi"]:.0f}</b> '
                    f'<span style="color:#e6edf3;font-weight:400;">'
                    f'(avg {int(w["pct"]*100)}% persist · combined {int(w["avg_vol"]):,}/d · {sign}{w["lo_otm"]*100:.1f} to {sign}{w["hi_otm"]*100:.1f}%)</span> '
                    f'<span style="color:{strength_color};">· {strength}</span></span>')
        return (f'<span style="padding:4px 10px;border-radius:6px;background:{color}22;'
                f'border:1px solid {color}66;color:{color};font-size:12px;font-weight:500;">'
                f'{label}: <b>${w["strike"]:.0f}</b> '
                f'<span style="color:#e6edf3;font-weight:400;">'
                f'({w["active"]}/{w["total"]}d · {int(w["avg_vol"]):,}/d · {sign}{w["otm"]*100:.1f}%)</span> '
                f'<span style="color:{strength_color};">· {strength}</span></span>')

    wall_badges = _wall_badge(call_wall, 'Call Wall', '#3fb950') + ' ' + _wall_badge(put_wall, 'Put Wall', '#f85149')

    # --- Wall structure auto-insight + range visualizer ---
    wall_insight = ""
    range_viz = ""
    if spot_val > 0:
        total_call_vol = sum(p['total_volume'] for p in all_persist if p['type'] == 'CALL')
        total_put_vol = sum(p['total_volume'] for p in all_persist if p['type'] == 'PUT')
        cp_ratio = total_call_vol / total_put_vol if total_put_vol > 0 else 99.0

        def _w_strength(w):
            if not w:
                return 'none'
            p = w['pct']
            return 'strong' if p >= 0.85 else 'moderate' if p >= 0.65 else 'weak'

        c_str = _w_strength(call_wall)
        p_str = _w_strength(put_wall)

        # Derive wall price bounds (use lo/hi for band, strike for single)
        def _w_lo(w):
            if not w:
                return None
            return w['lo'] if w.get('is_band') else w['strike']
        def _w_hi(w):
            if not w:
                return None
            return w['hi'] if w.get('is_band') else w['strike']

        call_floor = _w_lo(call_wall)  # lower bound of call resistance
        put_ceil = _w_hi(put_wall)     # upper bound of put support

        # Distance from spot to nearest wall
        call_dist_pct = ((call_floor - spot_val) / spot_val * 100) if call_floor else None
        put_dist_pct = ((spot_val - put_ceil) / spot_val * 100) if put_ceil else None

        # Tiered interpretation
        if cp_ratio > 4 and c_str in ('strong', 'moderate') and p_str in ('weak', 'none'):
            wi_color, wi_text, wi_icon = '#d29922', '#d29922', '⚠️'
            wi_msg = (f"<b>Asymmetric: call-heavy, put-thin</b> — call side total volume is "
                      f"<b>{cp_ratio:.1f}× the put side</b>. Dense call ladder + thin put support = "
                      f"<b>late-bull euphoria</b> (market complacent, no structural hedging). "
                      f"Expect upside to stall near call wall; downside has little cushion if sentiment flips.")
        elif cp_ratio > 2 and c_str == 'strong' and p_str in ('strong', 'moderate'):
            wi_color, wi_text, wi_icon = '#58a6ff', '#58a6ff', 'ℹ️'
            wi_msg = (f"<b>Balanced two-sided positioning</b> — both walls structural "
                      f"(call {c_str}, put {p_str}). Expect range-bound trade between the walls. "
                      f"Good regime for selling strangles / iron condors.")
        elif c_str == 'none' and p_str in ('strong', 'moderate'):
            wi_color, wi_text, wi_icon = '#3fb950', '#a8e6b8', '✓'
            wi_msg = (f"<b>Put-heavy / call-light</b> — heavy put structure but no call resistance. "
                      f"Often seen after a washout — upside path open if market stabilizes.")
        elif c_str == 'none' and p_str == 'none':
            wi_color, wi_text, wi_icon = '#8b949e', '#8b949e', 'ℹ️'
            wi_msg = (f"<b>No structural walls detected</b> — no strike has persistence ≥50% in the valid zone. "
                      f"Could be early in the data window, or positioning is rotating fast. Read with caution.")
        else:
            wi_color, wi_text, wi_icon = '#58a6ff', '#58a6ff', 'ℹ️'
            wi_msg = (f"<b>Mixed positioning</b> — call {c_str}, put {p_str}, call/put vol ratio {cp_ratio:.1f}×. "
                      f"No dominant structural theme.")

        # Proximity alerts appended to the main message
        prox_notes = []
        if call_dist_pct is not None and call_dist_pct < 3:
            prox_notes.append(f"<span style='color:#f85149;'>⚠ Spot within {call_dist_pct:.1f}% of call wall → imminent resistance test</span>")
        if put_dist_pct is not None and put_dist_pct < 3:
            prox_notes.append(f"<span style='color:#f85149;'>⚠ Spot within {put_dist_pct:.1f}% of put support → support test</span>")
        prox_html = ('<br>' + ' · '.join(prox_notes)) if prox_notes else ''

        wall_insight = f"""
    <div style="margin:10px 0;padding:10px 12px;border-left:3px solid {wi_color};background:{wi_color}11;border-radius:4px;font-size:12px;line-height:1.5;">
      <span style="margin-right:6px;">{wi_icon}</span><span style="color:{wi_text};">{wi_msg}</span>{prox_html}
    </div>"""

        # Range visualizer: spot position between walls
        if call_floor and put_ceil:
            span = call_floor - put_ceil
            if span > 0:
                spot_pos = max(0, min(100, (spot_val - put_ceil) / span * 100))
                implied_range_pct = span / spot_val * 100
                range_viz = f"""
    <div style="margin:8px 0 12px;padding:10px 12px 18px;background:#0f1117;border-radius:6px;font-size:11px;">
      <div style="display:flex;justify-content:space-between;color:#8b949e;margin-bottom:8px;">
        <span><b style="color:#f85149;">Put support ${put_ceil:.0f}</b></span>
        <span>Implied range: <b style="color:#e6edf3;">{implied_range_pct:.1f}%</b> ({put_dist_pct:.1f}% downside · {call_dist_pct:.1f}% upside)</span>
        <span><b style="color:#3fb950;">Call resistance ${call_floor:.0f}</b></span>
      </div>
      <div style="position:relative;height:10px;background:linear-gradient(to right, #f8514933 0%, #21262d 30%, #21262d 70%, #3fb95033 100%);border-radius:5px;margin-bottom:16px;">
        <div style="position:absolute;left:{spot_pos:.1f}%;top:-3px;width:2px;height:16px;background:#d29922;"></div>
        <div style="position:absolute;left:{spot_pos:.1f}%;top:16px;transform:translateX(-50%);color:#d29922;font-weight:500;white-space:nowrap;">▲ Spot ${spot_val:.2f}</div>
      </div>
    </div>"""
    # --- Unified strike axis for walls + OI delta sync ---
    persist_by_strike = {}
    for p in all_persist:
        persist_by_strike.setdefault(p['strike'], {})[p['type']] = p
    oi_by_strike = {}
    if oi_latest:
        for ch in oi_latest.get('changes', []):
            oi_by_strike[ch['strike']] = ch

    candidate_strikes = set(persist_by_strike) | set(oi_by_strike)
    if spot_val > 0:
        candidate_strikes = {s for s in candidate_strikes if abs(s - spot_val) / spot_val <= 0.25}
    if len(candidate_strikes) > 25 and spot_val > 0:
        candidate_strikes = set(sorted(candidate_strikes, key=lambda s: abs(s - spot_val))[:25])
    unified_strikes = sorted(candidate_strikes, reverse=True)

    # Wall table: use same <table> structure as OI Delta so row heights match exactly
    max_avg_vol = max(
        (p['total_volume'] / max(p['active_days'], 1) for p in all_persist),
        default=1,
    ) or 1

    def _wall_side_cell(item, side):
        if not item:
            dash = "<span style='color:#30363d;'>—</span>"
            if side == 'left':
                return f"<div style='display:flex;justify-content:flex-end;align-items:center;padding-right:4px;'>{dash}</div>"
            return f"<div style='display:flex;justify-content:flex-start;align-items:center;padding-left:4px;'>{dash}</div>"
        color = '#3fb950' if item['type'] == 'CALL' else '#f85149'
        pct = item['active_days'] / item['total_days'] if item['total_days'] > 0 else 0
        avg_vol = int(item['total_volume'] / item['active_days']) if item['active_days'] > 0 else 0
        opacity = 0.35 + 0.65 * pct
        bar_w = max(4, int(avg_vol / max_avg_vol * 80))
        bar = f"<div style='height:10px;width:{bar_w}px;background:{color};opacity:{opacity:.2f};border-radius:2px;'></div>"
        metric = f"<span style='color:#e6edf3;font-size:11px;white-space:nowrap;'>{avg_vol:,}/d <span style='color:#8b949e;'>· {item['active_days']}/{item['total_days']}</span></span>"
        if side == 'left':
            return f"<div style='display:flex;justify-content:flex-end;align-items:center;gap:6px;'>{metric}{bar}</div>"
        return f"<div style='display:flex;justify-content:flex-start;align-items:center;gap:6px;'>{bar}{metric}</div>"

    wall_rows_html = ""
    wall_spot_divider_tr = (
        f"<tr><td colspan='3' style='border-top:1.5px dashed #e6edf3;border-bottom:1.5px dashed #e6edf3;"
        f"background:#e6edf308;text-align:center;padding:4px 0;color:#e6edf3;font-weight:600;font-size:11px;'>"
        f"Spot ${spot_val:.2f}</td></tr>"
    )
    prev_above_wall = None
    for strike in unified_strikes:
        info = persist_by_strike.get(strike, {})
        above = strike > spot_val
        if prev_above_wall is True and not above:
            wall_rows_html += wall_spot_divider_tr
        prev_above_wall = above
        otm_pct = abs(strike - spot_val) / spot_val * 100 if spot_val > 0 else 0
        sign = '+' if above else '−' if strike < spot_val else '·'
        otm_color = '#3fb950' if above else '#f85149' if strike < spot_val else '#8b949e'
        strike_cell = (
            f"<span style='color:#e6edf3;font-weight:500;'>${strike:.0f}</span> "
            f"<span style='color:{otm_color};font-size:10px;opacity:0.8;'>{sign}{otm_pct:.1f}%</span>"
        )
        wall_rows_html += (
            f"<tr>"
            f"<td style='width:45%;'>{_wall_side_cell(info.get('PUT'), 'left')}</td>"
            f"<td style='text-align:center;width:10%;white-space:nowrap;'>{strike_cell}</td>"
            f"<td style='width:45%;'>{_wall_side_cell(info.get('CALL'), 'right')}</td>"
            f"</tr>"
        )

    # OI delta table — same unified strike axis, diverging bars (put left, strike center, call right)
    oi_rows = ""
    if oi_latest:
        spot = oi_latest.get('spot', 0)
        max_abs = max(
            (max(abs(ch['call_oi_delta']), abs(ch['put_oi_delta'])) for ch in oi_by_strike.values()),
            default=1,
        ) or 1

        def _oi_cell(delta, side, missing=False):
            if missing:
                dash = "<span style='color:#30363d;'>—</span>"
                if side == 'left':
                    return f"<div style='display:flex;justify-content:flex-end;align-items:center;padding-right:4px;'>{dash}</div>"
                return f"<div style='display:flex;justify-content:flex-start;align-items:center;padding-left:4px;'>{dash}</div>"
            color = '#3fb950' if delta > 0 else '#f85149' if delta < 0 else '#8b949e'
            if delta == 0:
                num = "<span style='color:#8b949e;'>0</span>"
                if side == 'left':
                    return f"<div style='display:flex;justify-content:flex-end;align-items:center;padding-right:4px;'>{num}</div>"
                return f"<div style='display:flex;justify-content:flex-start;align-items:center;padding-left:4px;'>{num}</div>"
            bar_w = max(2, int(abs(delta) / max_abs * 80))
            bar = f"<div style='height:10px;width:{bar_w}px;background:{color};border-radius:2px;'></div>"
            num = f"<span style='color:{color};font-weight:500;'>{delta:+,d}</span>"
            if side == 'left':
                return f"<div style='display:flex;justify-content:flex-end;align-items:center;gap:6px;'>{num}{bar}</div>"
            return f"<div style='display:flex;justify-content:flex-start;align-items:center;gap:6px;'>{bar}{num}</div>"

        prev_above = None
        spot_divider_tr = (
            f"<tr><td colspan='3' style='border-top:1.5px dashed #e6edf3;border-bottom:1.5px dashed #e6edf3;"
            f"background:#e6edf308;text-align:center;padding:4px 0;color:#e6edf3;font-weight:600;font-size:11px;'>"
            f"Spot ${spot:.2f}</td></tr>"
        )
        for strike_val in unified_strikes:
            ch = oi_by_strike.get(strike_val)
            above = strike_val > spot
            if prev_above is True and not above:
                oi_rows += spot_divider_tr
            prev_above = above
            if ch:
                c_delta = int(ch['call_oi_delta'])
                p_delta = int(ch['put_oi_delta'])
                put_cell = _oi_cell(p_delta, 'left')
                call_cell = _oi_cell(c_delta, 'right')
            else:
                put_cell = _oi_cell(0, 'left', missing=True)
                call_cell = _oi_cell(0, 'right', missing=True)
            otm_pct = abs(strike_val - spot) / spot * 100 if spot > 0 else 0
            sign = '+' if above else '−' if strike_val < spot else '·'
            otm_color = '#3fb950' if above else '#f85149' if strike_val < spot else '#8b949e'
            strike_label = (
                f"<span style='color:#e6edf3;font-weight:500;'>${strike_val:.0f}</span> "
                f"<span style='color:{otm_color};font-size:10px;opacity:0.8;'>{sign}{otm_pct:.1f}%</span>"
            )
            oi_rows += (
                f"<tr>"
                f"<td style='width:45%;'>{put_cell}</td>"
                f"<td style='text-align:center;width:10%;white-space:nowrap;'>{strike_label}</td>"
                f"<td style='width:45%;'>{call_cell}</td>"
                f"</tr>"
            )

    # --- OI Delta depth analysis + auto-insight ---
    oi_insight = ""
    oi_depth_html = ""
    if oi_latest and oi_latest.get('spot', 0) > 0:
        _spot = oi_latest['spot']
        otm_call = int(sum(ch['call_oi_delta'] for ch in oi_latest['changes'] if ch['strike'] > _spot))
        itm_call = int(sum(ch['call_oi_delta'] for ch in oi_latest['changes'] if ch['strike'] < _spot))
        otm_put = int(sum(ch['put_oi_delta'] for ch in oi_latest['changes'] if ch['strike'] < _spot))
        itm_put = int(sum(ch['put_oi_delta'] for ch in oi_latest['changes'] if ch['strike'] > _spot))
        deep_otm_put = int(sum(ch['put_oi_delta'] for ch in oi_latest['changes'] if ch['strike'] < _spot * 0.85))

        total_call = max(otm_call + itm_call, 1)
        total_put = max(otm_put + itm_put, 1)
        itm_call_ratio = itm_call / total_call if total_call > 0 else 0
        deep_put_ratio = deep_otm_put / total_put if total_put > 0 else 0

        # Tier logic
        if itm_call_ratio > 0.5 and deep_put_ratio > 0.3:
            oi_color, oi_text, oi_icon = '#d29922', '#d29922', '⚠️'
            oi_msg = (f"<b>Smart-money defensive posture</b> — {itm_call_ratio*100:.0f}% of call OI growth is "
                      f"<b>ITM</b> (covered-call writers locking gains, not bullish bets), while "
                      f"{deep_put_ratio*100:.0f}% of put OI growth is <b>deep OTM</b> (&lt;-15% spot = real tail hedges). "
                      f"Headline numbers hide this — true directional flow is <b>OTM calls {otm_call:+,d} "
                      f"vs OTM puts {otm_put:+,d}</b>. Institutions sell calls + buy puts = late-bull 'seatbelt-on' signal.")
        elif itm_call_ratio > 0.4:
            oi_color, oi_text, oi_icon = '#d29922', '#d29922', '⚠️'
            oi_msg = (f"<b>ITM call writing dominates</b> — {itm_call_ratio*100:.0f}% of call OI growth is ITM. "
                      f"Likely covered-call writers taking profit, not bullish accumulation. "
                      f"Strip ITM to read real flow: OTM calls {otm_call:+,d} vs OTM puts {otm_put:+,d}.")
        elif deep_put_ratio > 0.4:
            oi_color, oi_text, oi_icon = '#f85149', '#f85149', '🔥'
            oi_msg = (f"<b>Heavy tail-hedge buying</b> — {deep_put_ratio*100:.0f}% of put OI growth is deep OTM "
                      f"(&lt;-15% spot, strikes {deep_otm_put:+,d}). Systematic insurance demand = real downside fear.")
        elif otm_call > abs(otm_put) * 2 and otm_call > 5000:
            oi_color, oi_text, oi_icon = '#3fb950', '#a8e6b8', '✓'
            oi_msg = (f"<b>Clean bullish accumulation</b> — OTM call adds {otm_call:+,d} dominate; "
                      f"OTM put flow {otm_put:+,d} is minor. Directional upside bets, minimal hedging.")
        elif abs(otm_put) > otm_call * 2 and abs(otm_put) > 5000:
            oi_color, oi_text, oi_icon = '#f85149', '#f85149', '🔥'
            oi_msg = (f"<b>Directional put buying</b> — OTM put adds {otm_put:+,d} dwarf call additions "
                      f"{otm_call:+,d}. Clear bearish positioning.")
        else:
            oi_color, oi_text, oi_icon = '#58a6ff', '#58a6ff', 'ℹ️'
            oi_msg = (f"<b>Mixed flow</b> — OTM calls {otm_call:+,d}, OTM puts {otm_put:+,d}, ITM calls {itm_call:+,d}. "
                      f"No clear directional signal.")

        oi_insight = f"""
    <div style="margin:10px 0;padding:10px 12px;border-left:3px solid {oi_color};background:{oi_color}11;border-radius:4px;font-size:12px;line-height:1.5;">
      <span style="margin-right:6px;">{oi_icon}</span><span style="color:{oi_text};">{oi_msg}</span>
    </div>"""

        oi_depth_html = f"""
    <div style="display:flex;flex-wrap:wrap;gap:6px;margin:8px 0;font-size:11px;">
      <span style="padding:4px 9px;background:#3fb95022;border:1px solid #3fb95055;border-radius:6px;color:#e6edf3;">
        OTM Call (&gt;spot) <b style="color:#3fb950;">{otm_call:+,d}</b> <span style="color:#8b949e;">upside bets</span>
      </span>
      <span style="padding:4px 9px;background:#d2992222;border:1px solid #d2992255;border-radius:6px;color:#e6edf3;">
        ITM Call (&lt;spot) <b style="color:#d29922;">{itm_call:+,d}</b> <span style="color:#8b949e;">often covered-call writing</span>
      </span>
      <span style="padding:4px 9px;background:#f8514922;border:1px solid #f8514955;border-radius:6px;color:#e6edf3;">
        OTM Put (&lt;spot) <b style="color:#f85149;">{otm_put:+,d}</b> <span style="color:#8b949e;">downside / hedges</span>
      </span>
      <span style="padding:4px 9px;background:#21262d;border:1px solid #30363d;border-radius:6px;color:#e6edf3;">
        Deep OTM Put (&lt;-15%) <b style="color:#f85149;">{deep_otm_put:+,d}</b> <span style="color:#8b949e;">tail insurance</span>
      </span>
    </div>"""

    # EM accuracy table + insight
    em_rows = ""
    em_summary = ""
    em_insight = ""
    if em:
        n = len(em)
        hit = sum(1 for e in em if e['within_em'])
        hit_pct = hit / n * 100
        avg_err = sum(e['actual_move_pct'] for e in em) / n
        # Use median for bias (robust to outliers)
        mves_sorted = sorted(e['move_vs_em'] for e in em)
        median_mve = mves_sorted[n // 2] if n % 2 else (mves_sorted[n // 2 - 1] + mves_sorted[n // 2]) / 2
        max_mve = max(e['move_vs_em'] for e in em)
        outliers = sum(1 for e in em if e['move_vs_em'] > 2.0)
        up_misses = sum(1 for e in em if not e['within_em'] and e['actual_move_signed'] > 0)
        down_misses = sum(1 for e in em if not e['within_em'] and e['actual_move_signed'] < 0)

        # Benchmark: 1σ EM should contain ~68% of moves
        BENCHMARK = 68.0
        insufficient = n < 5
        hit_color = '#8b949e' if insufficient else ('#3fb950' if hit_pct >= BENCHMARK else '#d29922' if hit_pct >= 50 else '#f85149')
        bias_color = '#8b949e' if insufficient else ('#3fb950' if 0.8 <= median_mve <= 1.2 else '#d29922' if median_mve < 0.8 else '#f85149')
        bias_label = 'calibrated' if 0.8 <= median_mve <= 1.2 else ('over-predicts' if median_mve < 0.8 else 'under-predicts')

        outlier_badge = ''
        if outliers > 0:
            outlier_badge = f'<span style="margin-left:6px;padding:2px 6px;border-radius:4px;background:#f8514922;color:#f85149;font-size:10px;">⚠ {outliers} outlier{"s" if outliers > 1 else ""} (&gt;2× EM)</span>'

        sample_warn = ''
        if insufficient:
            sample_warn = f'<div style="margin-top:6px;padding:6px 10px;background:#d2992211;border-left:2px solid #d29922;border-radius:4px;font-size:11px;color:#d29922;">⚠ Sample size n={n} &lt; 5 — stats are unreliable. Need more daily snapshots for meaningful hit rate.</div>'

        em_summary = f"""
  <div style="display:flex;gap:20px;margin-bottom:8px;padding:12px 14px;background:#0d1117;border-radius:6px;border:1px solid #21262d;flex-wrap:wrap;">
    <div><span style="color:#8b949e;font-size:11px;">Hit rate</span><br><span style="font-size:20px;color:{hit_color};font-weight:600;">{hit_pct:.0f}%</span> <span style="color:#8b949e;font-size:11px;">({hit}/{n}) <span title="1σ EM should contain ~68%">vs 68% expected</span></span></div>
    <div><span style="color:#8b949e;font-size:11px;">Avg |move|</span><br><span style="font-size:20px;font-weight:600;">{avg_err:.1f}%</span></div>
    <div><span style="color:#8b949e;font-size:11px;">Bias (median)</span><br><span style="font-size:20px;color:{bias_color};font-weight:600;">{median_mve:.2f}x</span> <span style="color:#8b949e;font-size:11px;">{bias_label}</span>{outlier_badge}</div>
    <div><span style="color:#8b949e;font-size:11px;">Max miss</span><br><span style="font-size:20px;font-weight:600;color:{'#f85149' if max_mve > 2 else '#e6edf3'};">{max_mve:.2f}x</span></div>
    <div><span style="color:#8b949e;font-size:11px;">Miss direction</span><br><span style="font-size:14px;font-weight:500;">↑{up_misses} · ↓{down_misses}</span></div>
  </div>
  {sample_warn}"""

        # Auto-interpretation banner
        if insufficient:
            ei_color, ei_text, ei_icon = '#8b949e', '#8b949e', 'ℹ️'
            ei_msg = f"<b>Too few samples to judge</b> — {n} prediction{'s' if n > 1 else ''} observed. Need 5+ for a reliable read. Interpret trends qualitatively below."
        elif outliers > 0 and outliers / n >= 0.2:
            ei_color, ei_text, ei_icon = '#f85149', '#f85149', '🔥'
            ei_msg = f"<b>Tail-event regime</b> — {outliers}/{n} moves exceeded 2× EM. Vol regime has broken out of normal range; EM is not containing shocks. Size positions smaller, widen stops."
        elif hit_pct >= 80:
            ei_color, ei_text, ei_icon = '#3fb950', '#a8e6b8', '✓'
            ei_msg = f"<b>EM is conservative (too wide)</b> — hit rate {hit_pct:.0f}% exceeds expected 68%. Option IV priced at premium to realized; premium-selling has edge."
        elif hit_pct < 50:
            ei_color, ei_text, ei_icon = '#d29922', '#d29922', '⚠️'
            skew_note = f" Most misses to {'upside' if up_misses > down_misses else 'downside' if down_misses > up_misses else 'both sides'}." if (up_misses + down_misses) > 0 else ""
            ei_msg = f"<b>EM too narrow</b> — hit rate {hit_pct:.0f}% below 68% benchmark.{skew_note} Realized vol running higher than implied; buying options has edge, sell premium with caution."
        elif 0.8 <= median_mve <= 1.2:
            ei_color, ei_text, ei_icon = '#3fb950', '#a8e6b8', '✓'
            ei_msg = f"<b>Well-calibrated</b> — median actual/EM ratio {median_mve:.2f}x within [0.8, 1.2]. EM predicting real moves accurately; use it as a trading range estimator."
        else:
            ei_color, ei_text, ei_icon = '#58a6ff', '#58a6ff', 'ℹ️'
            ei_msg = f"<b>Mixed</b> — hit rate {hit_pct:.0f}%, median bias {median_mve:.2f}x. No strong regime signal."

        em_insight = f"""
  <div style="margin:8px 0 12px;padding:10px 12px;border-left:3px solid {ei_color};background:{ei_color}11;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">{ei_icon}</span><span style="color:{ei_text};">{ei_msg}</span>
  </div>"""

    for e in reversed(em):  # newest first
        within = e['within_em']
        color = '#3fb950' if within else '#f85149'
        date_label = e['date']
        if e.get('cal_days', 1) > 1:
            date_label += f" <span style='font-size:10px;color:#d29922;'>→{e['next_date']} ({e['cal_days']}d)</span>"

        # Move trajectory: $spot → $next_spot, with signed %
        spot = e['predicted_spot']
        actual = e['actual_spot']
        em_lo = e['em_lower']
        em_hi = e['em_upper']
        move_signed = e['actual_move_signed']
        direction = '↑' if move_signed > 0 else '↓' if move_signed < 0 else '·'
        dir_color = '#3fb950' if move_signed > 0 else '#f85149' if move_signed < 0 else '#8b949e'
        move_html = (
            f"<div style='font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;line-height:1.4;'>"
            f"<span style='color:#8b949e;'>${spot:.2f}</span>"
            f" <span style='color:#8b949e;'>→</span> "
            f"<span style='color:{dir_color};font-weight:500;'>${actual:.2f}</span>"
            f"</div>"
        )

        # Visual bar: axis covers [min(em_lo, actual), max(em_hi, actual)] + 0.5% margin
        x_min = min(em_lo, actual) - spot * 0.005
        x_max = max(em_hi, actual) + spot * 0.005
        span = max(x_max - x_min, 1e-9)
        def _pct(val): return max(0.0, min(100.0, (val - x_min) / span * 100))
        em_lo_pct = _pct(em_lo)
        em_hi_pct = _pct(em_hi)
        spot_pct = _pct(spot)
        actual_pct = _pct(actual)
        ratio_color = color if e['move_vs_em'] > 1.0 else '#8b949e'
        signed_pct = (move_signed / spot * 100) if spot > 0 else 0
        move_pct_str = f"{signed_pct:+.1f}%"
        # Label anchored to nearest edge: left half → left side, right half → right side
        if actual_pct < 50:
            label_pos_style = "left:5px;"
        else:
            label_pos_style = "right:5px;"
        bar_html = (
            "<div>"
            # Label row above bar, pinned to nearest edge
            f"<div style='position:relative;height:12px;'>"
            f"<div style='position:absolute;{label_pos_style}"
            f"font-size:10px;color:{color};font-weight:600;font-family:ui-monospace,SFMono-Regular,monospace;white-space:nowrap;line-height:12px;'>{move_pct_str}</div>"
            "</div>"
            # Bar itself
            "<div style='position:relative;height:20px;background:#161b22;border-radius:3px;overflow:hidden;'>"
            f"<div style='position:absolute;left:{em_lo_pct:.1f}%;width:{em_hi_pct-em_lo_pct:.1f}%;"
            f"top:0;bottom:0;background:#3fb95022;border-left:1.5px solid #3fb95088;border-right:1.5px solid #3fb95088;'></div>"
            f"<div style='position:absolute;left:{spot_pct:.1f}%;top:1px;bottom:1px;width:1px;"
            f"background:repeating-linear-gradient(to bottom,#8b949e 0 2px,transparent 2px 4px);'></div>"
            f"<div style='position:absolute;left:{actual_pct:.1f}%;top:3px;width:4px;height:14px;"
            f"background:{color};border-radius:2px;transform:translateX(-50%);box-shadow:0 0 0 1.5px #0d1117;'></div>"
            f"<div style='position:absolute;left:5px;top:3px;font-size:10px;color:#8b949e;font-family:ui-monospace,SFMono-Regular,monospace;pointer-events:none;'>±{e['em_pct']:.1f}%</div>"
            f"<div style='position:absolute;right:5px;top:3px;font-size:10px;color:{ratio_color};font-weight:500;font-family:ui-monospace,SFMono-Regular,monospace;pointer-events:none;'>{e['move_vs_em']:.2f}x</div>"
            "</div>"
            "</div>"
        )

        # Outcome: within + outlier flag
        outlier_flag = ' <span style="padding:1px 5px;border-radius:3px;background:#f8514933;color:#f85149;font-size:9px;font-weight:600;">OUTLIER</span>' if e['move_vs_em'] > 2.0 else ''
        outcome_html = f"<span style='color:{color};font-weight:500;'>{'✓ Yes' if within else '✗ No'}</span>{outlier_flag}"

        em_rows += (
            f"<tr><td>{date_label}</td>"
            f"<td>{move_html}</td>"
            f"<td style='min-width:220px;'>{bar_html}</td>"
            f"<td>{outcome_html}</td></tr>"
        )

    # Spread datasets JSON for Chart.js
    spread_js_datasets = ""
    all_dates_set = set()
    for ds in spread_datasets:
        all_dates_set.update(ds['dates'])
    all_dates = sorted(all_dates_set)

    chart_datasets = []
    for ds in spread_datasets:
        date_val_map = dict(zip(ds['dates'], ds['data']))
        aligned = [date_val_map.get(d, 'null') for d in all_dates]
        chart_datasets.append({
            'label': ds['label'],
            'data': aligned,
            'borderColor': ds['borderColor'],
            'borderWidth': 2,
            'pointRadius': 3,
            'tension': 0,
            'spanGaps': True,
        })

    num_snapshots = len(trend_data.get('iv_trend', []))

    # Build spot data aligned to chart dates for overlay
    spot_by_date = {p['date']: p['spot'] for p in iv}
    spot_aligned = [spot_by_date.get(d, 'null') for d in all_dates]

    # Build latest values for legend suffix
    latest_spread_vals = {}
    for ds in spread_datasets:
        if ds['data']:
            last_val = ds['data'][-1]
            if last_val != 'null':
                latest_spread_vals[ds['label']] = last_val

    # Auto-generated NVRP insight
    cur_iv = current.get('front_atm_iv', 0)
    cur_rv = current.get('rv30', 0)
    cur_nvrp = current.get('nvrp', 0)
    cur_iv_pct = current.get('iv_pct_2yr', 0)
    nvrp_insight = ""
    if cur_nvrp > 0:
        if cur_nvrp >= 1.5:
            nv_color, nv_text, nv_icon, nv_msg = '#3fb950', '#a8e6b8', '✓', f"<b>Strong premium-selling edge</b> — NVRP {cur_nvrp:.2f}x (IV {cur_iv:.1f}% vs RV {cur_rv:.1f}%, +{(cur_nvrp-1)*100:.0f}%). Clear edge for CC/CSP/wheel after covering friction costs."
        elif cur_nvrp >= 1.3:
            nv_color, nv_text, nv_icon, nv_msg = '#7ee787', '#a8e6b8', '✓', f"<b>Premium-selling edge present</b> — NVRP {cur_nvrp:.2f}x above 1.3 threshold. IV ({cur_iv:.1f}%) exceeds RV ({cur_rv:.1f}%) enough to cover typical friction costs."
        elif cur_nvrp >= 1.0:
            nv_color, nv_text, nv_icon, nv_msg = '#d29922', '#d29922', '⚠️', f"<b>Marginal / no edge</b> — NVRP {cur_nvrp:.2f}x between 1.0-1.3. IV ({cur_iv:.1f}%) only slightly above RV ({cur_rv:.1f}%); premium may not cover bid-ask + gamma risk. Avoid aggressive short vol."
        else:
            nv_color, nv_text, nv_icon, nv_msg = '#f85149', '#f85149', '🔥', f"<b>Long vol edge</b> — NVRP {cur_nvrp:.2f}x below 1.0. IV ({cur_iv:.1f}%) cheaper than RV ({cur_rv:.1f}%) — options underpriced. Consider BUYING vol (straddles/calls/puts) not selling."
        pct_str = f' &nbsp;|&nbsp; IV %ile (2yr): <b>{cur_iv_pct:.0f}</b>' if cur_iv_pct > 0 else ''
        nvrp_insight = f"""
  <div style="margin:10px 0;padding:10px 12px;border-left:3px solid {nv_color};background:{nv_color}11;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">{nv_icon}</span><span style="color:{nv_text};">{nv_msg}</span>{pct_str}
  </div>"""

    # --- IV Term Structure: latest ATM IV per bucket ---
    term_html = ""
    term_struct = {}
    if pc:
        ts_dates = {p['date'] for bucket_data in pc.values() for p in bucket_data}
        if ts_dates:
            latest_term_date = max(ts_dates)
            for bucket in ['front', '2w', '1m', '2m', 'far']:
                if bucket not in pc:
                    continue
                # Exclude 0/1-DTE (pathological IV at expiration); keep DTE>=2
                ivs = [p['atm_iv'] for p in pc[bucket]
                       if p['date'] == latest_term_date and p.get('atm_iv', 0) > 0
                       and p.get('dte', 0) >= 2]
                if ivs:
                    term_struct[bucket] = sum(ivs) / len(ivs)

    if len(term_struct) >= 2:
        bucket_labels = {'front': 'Front (≤7d)', '2w': '2W (8-14d)', '1m': '1M (15-35d)', '2m': '2M (36-70d)', 'far': 'Far (>70d)'}
        max_iv = max(term_struct.values())
        front_iv_ts = term_struct.get('front', 0)
        far_iv_ts = (term_struct.get('2m') or term_struct.get('far')
                     or term_struct.get('1m') or term_struct.get('2w') or front_iv_ts)
        slope = front_iv_ts - far_iv_ts  # >0 backwardation, <0 contango

        if slope > 5:
            ts_color, ts_icon, ts_msg = '#f85149', '🔥', f"<b>Backwardation</b> — Front IV ({front_iv_ts:.1f}%) above Far ({far_iv_ts:.1f}%) by {slope:+.1f} pts. Market pricing near-term event/risk; far-dated options relatively 'cheap'."
        elif slope > 1:
            ts_color, ts_icon, ts_msg = '#d29922', '⚠️', f"<b>Mild backwardation</b> — Front IV slightly above Far ({slope:+.1f} pts). Some near-term anxiety; not yet stress regime."
        elif slope < -5:
            ts_color, ts_icon, ts_msg = '#3fb950', '✓', f"<b>Steep contango</b> — Front IV ({front_iv_ts:.1f}%) well below Far ({far_iv_ts:.1f}%) by {slope:+.1f} pts. Market complacent now, pricing uncertainty over time. Front puts may be cheap insurance."
        elif slope < -1:
            ts_color, ts_icon, ts_msg = '#7ee787', '✓', f"<b>Normal contango</b> — Front below Far ({slope:+.1f} pts). Healthy term structure; nothing imminent priced in."
        else:
            ts_color, ts_icon, ts_msg = '#8b949e', '·', f"<b>Flat term structure</b> — IV roughly uniform across DTE ({slope:+.1f} pts). Neither stress nor complacency."

        bar_rows = ""
        for bucket in ['front', '2w', '1m', '2m', 'far']:
            if bucket not in term_struct:
                continue
            iv_val = term_struct[bucket]
            bar_w = int(iv_val / max_iv * 100) if max_iv > 0 else 0
            bar_rows += (
                f"<div style='display:grid;grid-template-columns:120px 70px 1fr;gap:10px;align-items:center;padding:3px 0;font-size:12px;'>"
                f"<div style='color:#e6edf3;font-weight:500;'>{bucket_labels[bucket]}</div>"
                f"<div style='font-family:ui-monospace,SFMono-Regular,monospace;color:#e6edf3;text-align:right;'>{iv_val:.1f}%</div>"
                f"<div><div style='height:14px;width:{bar_w}%;background:linear-gradient(90deg,#58a6ff,#a371f7);border-radius:2px;min-width:4px;'></div></div>"
                f"</div>"
            )

        term_html = f"""
<div class="card" id="term">
  <h2>IV Term Structure (ATM IV by DTE)</h2>
  <div style="font-size:11px;color:#8b949e;margin-bottom:8px;line-height:1.6;">
    Slope across the IV curve. <b style="color:#3fb950;">Contango</b> (rising IV with DTE) = normal/complacent. <b style="color:#f85149;">Backwardation</b> (falling) = market pricing immediate event risk. Use Front−Far &gt;5 pts as stress threshold.
  </div>
  <div style="margin:8px 0;padding:10px 12px;border-left:3px solid {ts_color};background:{ts_color}11;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">{ts_icon}</span><span style="color:{ts_color};">{ts_msg}</span>
  </div>
  <div style="margin-top:10px;">{bar_rows}</div>
</div>"""

    # Auto-generated current-state insight
    spread_insight = ""
    if latest_spread_vals:
        vals = list(latest_spread_vals.values())
        neg_count = sum(1 for v in vals if v < 0)
        total = len(vals)
        extreme_neg = min(vals)
        avg_val = sum(vals) / len(vals)
        if neg_count == total and extreme_neg < -10:
            color, text_color, icon, msg = '#f85149', '#f85149', '🔥', f'<b>EXTREME CALL SKEW</b> — all {total} buckets in danger zone (avg {avg_val:+.1f} pts, low {extreme_neg:+.1f}). Speculative FOMO across the full term structure. Historical pattern: local top within 1-3 weeks. Consider buying OTM puts as hedge while they\'re relatively cheap.'
        elif neg_count == total:
            color, text_color, icon, msg = '#d29922', '#d29922', '⚠️', f'<b>Call skew across all buckets</b> — all {total} expirations negative (avg {avg_val:+.1f} pts). Bullish euphoria building; watch for further deterioration toward extreme levels.'
        elif neg_count >= total / 2:
            color, text_color, icon, msg = '#d29922', '#d29922', '⚠️', f'<b>Partial call skew</b> — {neg_count}/{total} buckets negative (low {extreme_neg:+.1f}). Speculative flow emerging in some expirations. Monitor if it spreads to all buckets.'
        elif neg_count > 0:
            color, text_color, icon, msg = '#58a6ff', '#58a6ff', 'ℹ️', f'<b>Mixed</b> — {neg_count}/{total} buckets negative (low {extreme_neg:+.1f}). Call skew localized to shorter dates; longer-dated expirations still in safe zone.'
        else:
            color, text_color, icon, msg = '#3fb950', '#a8e6b8', '✓', f'<b>Healthy skew</b> — all {total} buckets positive (avg {avg_val:+.1f} pts). Normal put-protection demand; no FOMO signal.'
        spread_insight = f"""
  <div style="margin:10px 0;padding:10px 12px;border-left:3px solid {color};background:{color}11;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">{icon}</span><span style="color:{text_color};">{msg}</span>
  </div>"""

    # --- Historical skew percentile banner ---
    skew_pct_html = ""
    if skew_pct:
        regime_map = {
            'extreme_call_skew': ('#f85149', '#f85149', '🔥',
                f"<b>Extreme call skew — bottom {skew_pct['percentile']:.0f}th %ile</b> of {skew_pct['sample_size']}-day history (current {skew_pct['current']:+.1f} vs median {skew_pct['median']:+.1f}, low {skew_pct['min']:+.1f}). When skew is this compressed, rally often near exhaustion — consider fading or hedging with cheap OTM puts."),
            'heavy_call_skew': ('#d29922', '#d29922', '⚠️',
                f"<b>Heavy call skew — {skew_pct['percentile']:.0f}th %ile</b> of {skew_pct['sample_size']}-day history (current {skew_pct['current']:+.1f} vs median {skew_pct['median']:+.1f}). Bullish flow building; watch for deterioration toward extreme."),
            'normal_range': ('#3fb950', '#a8e6b8', '✓',
                f"<b>Skew within normal band</b> — {skew_pct['percentile']:.0f}th %ile of {skew_pct['sample_size']}-day history (current {skew_pct['current']:+.1f} vs median {skew_pct['median']:+.1f}). No extreme positioning signal."),
            'heavy_put_skew': ('#58a6ff', '#58a6ff', 'ℹ️',
                f"<b>Heavy put skew — {skew_pct['percentile']:.0f}th %ile</b> of {skew_pct['sample_size']}-day history (current {skew_pct['current']:+.1f}). Fear bid on puts; historically a lagging bottom signal."),
            'extreme_put_skew': ('#3fb950', '#a8e6b8', '✓',
                f"<b>Extreme put skew — top {skew_pct['percentile']:.0f}th %ile</b> of {skew_pct['sample_size']}-day history. Capitulation / panic bid on puts; historically near local bottoms."),
        }
        sc, st, si, smsg = regime_map.get(skew_pct['regime'], ('#8b949e', '#8b949e', 'ℹ️', ''))
        warn = ''
        if not skew_pct['reliable']:
            warn = f' <span style="color:#8b949e;font-size:11px;">· ⚠ only {skew_pct["sample_size"]} days of history — percentile unreliable until n≥60</span>'
        skew_pct_html = f"""
  <div style="margin:10px 0;padding:10px 12px;border-left:3px solid {sc};background:{sc}11;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">{si}</span><span style="color:{st};">{smsg}</span>{warn}
  </div>"""

    # --- Call OI build by DTE bucket banner ---
    call_dte_html = ""
    if call_dte and call_dte['total_call'] > 0:
        b = call_dte['call_buckets']
        regime_map = {
            'speculative': ('#f85149', '#f85149', '🔥',
                f"<b>Speculative call build — {call_dte['short_pct']:.0f}% short-dated (≤14d)</b>. Call OI flooding near-term expirations (+{b['short']:,}) while mid/long stay cold (+{b['mid']:,} / +{b['long']:,}). Classic FOMO / catalyst-chase pattern — rally often unsustainable without long-dated conviction."),
            'near_term_lean': ('#d29922', '#d29922', '⚠️',
                f"<b>Near-term lean — {call_dte['short_pct']:.0f}% short-dated</b> (short +{b['short']:,} / mid +{b['mid']:,} / long +{b['long']:,}). Short-dated bias but some mid-term participation. Watch whether long-dated follows."),
            'structural': ('#3fb950', '#a8e6b8', '✓',
                f"<b>Structural bullish build — {call_dte['long_pct']:.0f}% long-dated (&gt;45d)</b> (short +{b['short']:,} / mid +{b['mid']:,} / long +{b['long']:,}). Conviction across the term structure — more sustainable than a short-dated-only squeeze."),
            'balanced': ('#58a6ff', '#58a6ff', 'ℹ️',
                f"<b>Balanced call build</b> (short +{b['short']:,} / mid +{b['mid']:,} / long +{b['long']:,}). No strong DTE tilt either way."),
            'net_unwind': ('#8b949e', '#8b949e', 'ℹ️',
                f"<b>Net call OI unwind</b> ({call_dte['total_call']:+,} total). Positions being closed rather than opened."),
        }
        cc, ct, ci, cmsg = regime_map.get(call_dte['regime'], ('#8b949e', '#8b949e', 'ℹ️', ''))
        call_dte_html = f"""
  <div style="margin:10px 0;padding:10px 12px;border-left:3px solid {cc};background:{cc}11;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">{ci}</span><span style="color:{ct};">{cmsg}</span>
  </div>"""

    # --- GEX (Dealer Gamma Exposure) Walls card ---
    gex_card_html = ""
    if gex_data and (gex_data['call_walls'] or gex_data['put_walls']):
        spot = gex_data['spot']
        regime_map = {
            'long_gamma': ('#3fb950', '#a8e6b8', '✓',
                f"<b>Dealers net long gamma</b> — price acts as <b>magnet</b>. Moves get suppressed; intraday chop inside the wall band is likely. Favorable for premium selling (CC/CSP) — positions pin toward high-gamma strikes."),
            'short_gamma': ('#f85149', '#f85149', '🔥',
                f"<b>Dealers net short gamma</b> — moves get <b>amplified</b>. Breakouts become squeezes, breakdowns become crashes. <b>Dangerous for premium selling</b>; consider straddles/long-vol. Spot crossing a major wall triggers mechanical hedging (chase rally / panic dump)."),
            'neutral': ('#d29922', '#d29922', 'ℹ️',
                f"<b>Mixed gamma regime</b> — calls and puts roughly balanced. Spot behavior neither magnetic nor explosive; walls are soft."),
        }
        gc, gt, gi, gmsg = regime_map.get(gex_data['regime'], ('#8b949e', '#8b949e', 'ℹ️', ''))
        # Scale GEX for readable display
        def _fmt_gex(g):
            ag = abs(g)
            if ag >= 1e9:
                return f"${g/1e9:+.2f}B"
            if ag >= 1e6:
                return f"${g/1e6:+.1f}M"
            return f"${g/1e3:+.0f}K"
        # Price ladder: strikes sorted high→low, spot divider in middle
        calls = list(gex_data['call_walls'])[:5]
        puts = list(gex_data['put_walls'])[:5]
        calls_by_strike = sorted(calls, key=lambda w: w['strike'], reverse=True)
        puts_by_strike = sorted(puts, key=lambda w: w['strike'], reverse=True)
        # Shared magnitude scale so call and put bar lengths are comparable
        global_max = max(
            (abs(w['gex']) for w in calls + puts),
            default=1,
        ) or 1

        def _ladder_row(w, color):
            bar_pct = abs(w['gex']) / global_max * 100
            bar_bg = color + '33'
            return (
                '<div style="display:grid;grid-template-columns:68px 56px 1fr 92px;align-items:center;gap:10px;'
                'padding:5px 0;font-size:12px;font-family:ui-monospace,SFMono-Regular,monospace;">'
                f'<span style="color:#e6edf3;font-weight:500;">${w["strike"]:.0f}</span>'
                f'<span style="color:#8b949e;font-size:11px;">{"+" if color == "#3fb950" else "−"}{w["otm_pct"]:.1f}%</span>'
                f'<div style="background:#21262d;height:10px;border-radius:3px;overflow:hidden;">'
                f'<div style="background:{bar_bg};border-left:2px solid {color};height:100%;width:{bar_pct:.1f}%;"></div>'
                f'</div>'
                f'<span style="color:{color};text-align:right;font-weight:500;">{_fmt_gex(w["gex"])}</span>'
                '</div>'
            )

        call_ladder = ''.join(_ladder_row(w, '#3fb950') for w in calls_by_strike)
        put_ladder = ''.join(_ladder_row(w, '#f85149') for w in puts_by_strike)
        if not call_ladder:
            call_ladder = '<div style="color:#8b949e;font-size:11px;padding:4px 0;">No call walls within 25% OTM</div>'
        if not put_ladder:
            put_ladder = '<div style="color:#8b949e;font-size:11px;padding:4px 0;">No put walls within 25% OTM</div>'

        spot_divider = (
            '<div style="display:grid;grid-template-columns:68px 56px 1fr 92px;align-items:center;gap:10px;'
            'margin:6px 0;padding:4px 0;border-top:1.5px dashed #e6edf3;border-bottom:1.5px dashed #e6edf3;'
            'background:#e6edf308;">'
            f'<span style="color:#e6edf3;font-weight:600;font-size:12px;">${spot:.2f}</span>'
            '<span style="color:#8b949e;font-size:10px;">Spot</span>'
            '<div></div><div></div></div>'
        )

        gex_card_html = f"""
<div class="card" id="gex">
  <h2>Dealer Gamma Exposure (GEX) Walls</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:6px;line-height:1.6;">
    Strikes ranked by <b style="color:#e6edf3;">Σ (gamma × OI × 100)</b> across all expirations — measures where <b style="color:#e6edf3;">dealer hedging flow</b> concentrates. More accurate than volume-weighted walls for identifying actual resistance/support.
    Net GEX: <b style="color:{gc};">{_fmt_gex(gex_data['net_gex'])}</b> (call {_fmt_gex(gex_data['total_call_gex'])} + put {_fmt_gex(gex_data['total_put_gex'])}).
  </div>
  <div style="margin:10px 0;padding:10px 12px;border-left:3px solid {gc};background:{gc}11;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">{gi}</span><span style="color:{gt};">{gmsg}</span>
  </div>
  <div style="margin-top:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;font-size:10px;letter-spacing:0.5px;color:#8b949e;margin-bottom:4px;padding:0 2px;">
      <span>Strike · OTM</span><span style="color:#3fb950;">▲ CALL WALLS (resistance)</span><span>GEX</span>
    </div>
    {call_ladder}
    {spot_divider}
    {put_ladder}
    <div style="display:flex;justify-content:space-between;align-items:center;font-size:10px;letter-spacing:0.5px;color:#8b949e;margin-top:4px;padding:0 2px;">
      <span>&nbsp;</span><span style="color:#f85149;">▼ PUT WALLS (support)</span><span>&nbsp;</span>
    </div>
  </div>
  <details style="font-size:11px;color:#8b949e;margin-top:10px;">
    <summary style="cursor:pointer;color:#58a6ff;">How GEX walls differ from volume walls (click)</summary>
    <div style="margin-top:8px;line-height:1.7;padding:8px 10px;background:#0f1117;border-radius:6px;">
      <b style="color:#e6edf3;">Volume walls</b> (below, Persistent Volume Strikes) show where retail/institutional <b>flow concentrates</b> — backward-looking conviction.
      <br><b style="color:#e6edf3;">GEX walls</b> (this card) show where <b>dealer hedging pressure</b> is largest — forward-looking, drives actual intraday price behavior.
      <br><br>A strike can be a volume wall without being a GEX wall (e.g. far-dated OI has low gamma). The <b>GEX wall is what actually acts as resistance/support intraday</b> because dealers must trade stock to hedge. When they agree, the wall is robust. When they disagree, trust GEX.
    </div>
  </details>
</div>"""

    # --- BTC Context card (MSTR only) ---
    btc_card_html = ""
    if btc_data:
        btc_color = '#3fb950' if btc_data['btc_24h'] >= 0 else '#f85149'
        btc_arrow = '▲' if btc_data['btc_24h'] >= 0 else '▼'
        mstr_color = '#3fb950' if btc_data['mstr_24h'] >= 0 else '#f85149'
        mstr_arrow = '▲' if btc_data['mstr_24h'] >= 0 else '▼'
        corr_str = f"{btc_data['correlation']:+.2f}" if btc_data['correlation'] is not None else 'n/a'
        beta_str = f"{btc_data['beta']:+.2f}" if btc_data['beta'] is not None else 'n/a'
        # Divergence banner
        div_html = ""
        if btc_data['divergence']:
            d = btc_data['divergence']
            if d['direction'] == 'premium':
                dc, dt, di, dmsg = '#d29922', '#d29922', '⚠️', (
                    f"<b>MSTR trading at premium vs BTC-implied</b> — expected ~{d['expected']:+.1f}% based on β, actual {d['actual']:+.2f}% (residual {d['residual']:+.2f}%). "
                    f"Short-term premium tends to mean-revert; if BTC stays flat, MSTR likely gives back the excess. Consider fading extreme premiums or avoiding fresh longs.")
            else:
                dc, dt, di, dmsg = '#58a6ff', '#58a6ff', 'ℹ️', (
                    f"<b>MSTR trading at discount vs BTC-implied</b> — expected ~{d['expected']:+.1f}% based on β, actual {d['actual']:+.2f}% (residual {d['residual']:+.2f}%). "
                    f"Relative weakness; could be tax-loss selling, dilution, or idiosyncratic. Mean-reversion bias bullish if BTC holds.")
            div_html = f"""
  <div style="margin:10px 0;padding:10px 12px;border-left:3px solid {dc};background:{dc}11;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">{di}</span><span style="color:{dt};">{dmsg}</span>
  </div>"""
        else:
            div_html = f"""
  <div style="margin:10px 0;padding:10px 12px;border-left:3px solid #3fb950;background:#3fb95011;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">✓</span><span style="color:#a8e6b8;"><b>No MSTR/BTC divergence</b> — MSTR move is consistent with BTC move × beta. Options flow can be taken at face value without premium/discount overlay.</span>
  </div>"""

        btc_card_html = f"""
<div class="card" id="btc">
  <h2>BTC Context (MSTR = BTC-levered proxy)</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:6px;line-height:1.6;">
    MSTR's ~{beta_str} beta to BTC means you must evaluate MSTR flow <b style="color:#e6edf3;">relative to BTC</b>. Bullish MSTR options on a flat-BTC day = premium (historically fades). BTC ripping but MSTR calm = discount (historically catches up).
  </div>
  <div class="kpi-row" style="grid-template-columns:repeat(auto-fit, minmax(140px,1fr));">
    <div class="kpi">
      <div class="kpi-label">BTC-USD</div>
      <div class="kpi-value">${btc_data['btc_price']:,.0f}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">BTC 24h</div>
      <div class="kpi-value" style="color:{btc_color};">{btc_arrow} {btc_data['btc_24h']:+.2f}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">BTC 7d</div>
      <div class="kpi-value" style="color:{'#3fb950' if btc_data['btc_7d']>=0 else '#f85149'};">{btc_data['btc_7d']:+.2f}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">MSTR 24h</div>
      <div class="kpi-value" style="color:{mstr_color};">{mstr_arrow} {btc_data['mstr_24h']:+.2f}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Correlation ({btc_data['sample_n']}d)</div>
      <div class="kpi-value">{corr_str}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">MSTR β to BTC</div>
      <div class="kpi-value">{beta_str}</div>
    </div>
  </div>
  {div_html}
</div>"""

    # --- mNAV card (MSTR only) ---
    mnav_card_html = ""
    if ticker == 'MSTR' and holdings and btc_data:
        spot = current.get('spot') or 0
        btc_price = btc_data.get('btc_price') or 0
        shares = holdings.get('basic_shares_outstanding') or 0
        btc_held = holdings.get('btc_holdings') or 0
        debt = holdings.get('debt') or 0
        pref = holdings.get('pref') or 0
        cash = holdings.get('cash') or 0
        as_of = holdings.get('as_of_date', '')
        mc = spot * shares
        btc_reserve = btc_held * btc_price
        if btc_reserve > 0:
            simple_mnav = mc / btc_reserve
            ev = mc + debt + pref - cash
            ev_mnav = ev / btc_reserve

            def _verdict(x):
                if x < 1.2:
                    return ('cheap', '#3fb950', 'Cheap')
                if x <= 1.8:
                    return ('normal', '#d29922', 'Normal')
                return ('expensive', '#f85149', 'Expensive')
            tag, color, label = _verdict(simple_mnav)
            mnav_card_html = f"""
<div class="card" id="mnav">
  <h2>mNAV (MSTR Premium to BTC NAV)</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:10px;line-height:1.6;">
    Premium MSTR trades at vs the underlying BTC stash. <b style="color:#e6edf3;">Simple</b> = MarketCap / BTC reserve. <b style="color:#e6edf3;">EV-based</b> = (MC + debt + pref − cash) / BTC reserve (matches strategy.com). Cheap &lt; 1.2x · Normal 1.2–1.8x · Expensive &gt; 1.8x.
  </div>
  <div class="kpi-row" style="grid-template-columns:repeat(auto-fit, minmax(140px,1fr));">
    <div class="kpi">
      <div class="kpi-label">Simple mNAV</div>
      <div class="kpi-value" style="color:{color};">{simple_mnav:.2f}x</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">EV-based mNAV</div>
      <div class="kpi-value">{ev_mnav:.2f}x</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Verdict</div>
      <div class="kpi-value" style="color:{color};">{label}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">BTC Held</div>
      <div class="kpi-value">{btc_held:,}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">BTC Reserve</div>
      <div class="kpi-value">${btc_reserve/1e9:.1f}B</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Market Cap</div>
      <div class="kpi-value">${mc/1e9:.1f}B</div>
    </div>
  </div>
  <div style="font-size:11px;color:#6e7681;margin-top:6px;">Holdings as of {as_of} · source: strategy.com</div>
</div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{ticker} Flow Trend</title>
<style>
body {{ background:#000; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display","Inter",system-ui,sans-serif; font-feature-settings:"tnum" 1,"cv11" 1; margin:0; padding:20px; }}
.header {{ display:flex; align-items:center; gap:16px; margin-bottom:20px; }}
.header h1 {{ margin:0; font-size:24px; font-weight:500; letter-spacing:-0.01em; }}
.badge {{ padding:4px 12px; border-radius:12px; font-size:13px; font-weight:500; }}
.card {{ background:#1c1c1e; border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:20px; margin-bottom:16px; box-shadow:0 1px 0 rgba(255,255,255,0.04) inset; }}
.card h2 {{ margin:0 0 12px; font-size:16px; font-weight:500; color:#8b949e; letter-spacing:-0.005em; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; grid-template-rows:auto auto; gap:16px; }}
.grid > .card {{ display:grid; grid-row:span 2; grid-template-rows:subgrid; gap:0; }}
.kpi-row {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(140px,1fr)); gap:12px; margin-bottom:16px; }}
.kpi {{ background:#1c1c1e; border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:12px 16px; text-align:center; }}
.kpi-label {{ font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; }}
.kpi-value {{ font-size:22px; font-weight:500; margin-top:4px; font-variant-numeric:tabular-nums; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
th {{ color:#8b949e; font-weight:500; text-align:right; padding:6px 10px; border-bottom:1px solid #30363d; }}
td {{ padding:6px 10px; text-align:right; border-bottom:1px solid rgba(255,255,255,0.04); }}
tbody tr:nth-child(even) td {{ background:rgba(255,255,255,0.015); }}
th:first-child, td:first-child {{ text-align:left; }}
.chart-wrap {{ position:relative; height:280px; }}
.chart-wrap-tall {{ position:relative; height:420px; }}
.card-head {{ min-height:270px; }}
.tbl-wrap {{ max-height:400px; overflow-y:auto; }}
.sync-tbl td {{ height:30px; box-sizing:border-box; }}
.sync-tbl tr td[colspan="3"] {{ height:auto; }}
.zero-line {{ color:#f85149; font-size:11px; }}
.topnav {{ position:sticky; top:0; z-index:100; background:rgba(0,0,0,0.78); backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px); display:flex; gap:14px; padding:10px 20px; margin:-20px -20px 16px; border-bottom:1px solid rgba(255,255,255,0.06); font-size:13px; align-items:center; }}
.topnav a {{ color:#8b949e; text-decoration:none; padding:4px 10px; border-radius:6px; transition:background 0.12s,color 0.12s; }}
.topnav a:hover {{ color:#e6edf3; background:rgba(255,255,255,0.06); }}
.topnav a.active {{ color:#e6edf3; background:rgba(88,166,255,0.18); }}
.nav-center {{ display:flex; gap:2px; flex:1; flex-wrap:wrap; justify-content:center; }}
.nav-left, .nav-right {{ display:flex; gap:4px; align-items:center; }}
.nav-divider {{ color:#30363d; }}
.card {{ scroll-margin-top:64px; }}
.kpi-row {{ scroll-margin-top:64px; }}
</style>
</head><body>

<nav class="topnav">
  <div class="nav-left">
    <a href="trend_MSTR_{report_date}{lang_suffix}.html" class="{'active' if ticker == 'MSTR' else ''}">MSTR</a>
    <a href="trend_TSLA_{report_date}{lang_suffix}.html" class="{'active' if ticker == 'TSLA' else ''}">TSLA</a>
  </div>
  <div class="nav-center">
    <a href="#kpis">{nav_lbl_kpi}</a>
    <a href="#btc">BTC</a>
    {('<a href="#mnav">mNAV</a>' if ticker == 'MSTR' else '')}
    <a href="#spread">{nav_lbl_skew}</a>
    <a href="#iv">IV</a>
    <a href="#term">{nav_lbl_term}</a>
    <a href="#gex">GEX</a>
    <a href="#oi">OI</a>
    <a href="#walls">{nav_lbl_walls}</a>
    <a href="#em">EM</a>
  </div>
  <div class="nav-right">
    <a href="trend_{ticker}_{report_date}.html" class="{'active' if lang == 'en' else ''}">EN</a>
    <a href="trend_{ticker}_{report_date}_zh.html" class="{'active' if lang == 'zh' else ''}">中文</a>
  </div>
</nav>

<div class="header">
  <h1>{ticker} Flow Trend Analysis</h1>
  <span class="badge" style="background:{skew_color}22;color:{skew_color};border:1px solid {skew_color};">{skew_level.replace('_',' ')}</span>
  <span style="color:#8b949e;font-size:13px;">{num_snapshots} snapshots | {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
</div>

<!-- KPI Row -->
<div class="kpi-row" id="kpis">
  <div class="kpi">
    <div class="kpi-label">Spot</div>
    <div class="kpi-value">${current.get('spot', 0):.2f}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Front P/C Spread</div>
    <div class="kpi-value" style="color:{skew_color};">{current_front_spread:+.1f} pts</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">ATM IV</div>
    <div class="kpi-value">{current.get('front_atm_iv', 0):.1f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">RV30</div>
    <div class="kpi-value">{current.get('rv30', 0):.1f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">NVRP</div>
    <div class="kpi-value">{current.get('nvrp', 0):.2f}x</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">IV %ile (2yr)</div>
    <div class="kpi-value">{current.get('iv_pct_2yr', 0):.0f}</div>
  </div>
</div>

{mode_banner_html}

{btc_card_html}

{mnav_card_html}

<!-- Zero-Crossing Alerts -->
<div class="card">
  <h2>Zero-Crossing Alerts (P/C Spread)</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:8px;">
    Historical pattern: when delta-25 P/C IV spread breaks below zero, stock often near a local top within 1-3 weeks.
  </div>
  {alert_html}
</div>

<!-- P/C Spread Time Series (full width) -->
<div class="card" id="spread">
  <h2>Delta-25 Put/Call IV Spread (by expiration bucket)</h2>
  <div style="font-size:11px;color:#8b949e;margin-bottom:6px;line-height:1.6;">
    <b style="color:#e6edf3;">Spread = Put IV − Call IV</b> at delta-25 strikes. Positive = puts more expensive (normal hedging demand). Negative = calls more expensive (speculative FOMO).
  </div>
  <div style="font-size:11px;margin-bottom:6px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;">
    <span style="color:#a8e6b8;"><span style="display:inline-block;width:14px;height:10px;background:#3fb95022;border:1px solid #3fb95055;vertical-align:middle;margin-right:4px;"></span><b>Safe zone</b> (spread &gt; 0) — puts cost more than calls, market pricing in downside risk = healthy</span>
    <span style="color:#f85149;"><span style="display:inline-block;width:14px;height:10px;background:#f8514922;border:1px solid #f8514955;vertical-align:middle;margin-right:4px;"></span><b>Danger zone</b> (spread &lt; 0) — calls cost more than puts, investors chasing upside = top signal</span>
    <span><span style="display:inline-block;width:16px;border-top:1.5px dashed #e6edf3;vertical-align:middle;margin-right:4px;"></span>Zero line (threshold)</span>
    <span><span style="display:inline-block;width:16px;border-top:1.5px dashed #8b949e;vertical-align:middle;margin-right:4px;"></span>Spot price (right axis)</span>
  </div>
  {spread_insight}
  {skew_pct_html}
  <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px;" id="spreadLegend"></div>
  <div class="chart-wrap-tall"><canvas id="spreadChart"></canvas></div>
</div>

<!-- IV + NVRP Trend -->
<div class="card" id="iv">
  <h2>ATM IV vs RV30 Trend</h2>
  <div style="font-size:11px;color:#8b949e;margin-bottom:6px;line-height:1.6;">
    <b style="color:#e6edf3;">NVRP = ATM IV / RV30.</b> The 1.3 threshold exists because sellers need ~30% cushion to cover bid-ask spread, gamma risk, and hedging error. &lt;1.0 = long-vol edge; 1.0-1.3 = marginal; &gt;1.3 = short-vol edge; &gt;1.5 = strong edge.
  </div>
  {nvrp_insight}
  <div class="chart-wrap"><canvas id="ivChart"></canvas></div>
</div>

{term_html}

{gex_card_html}

<div class="grid">
  <!-- OI Delta -->
  <div class="card" id="oi">
    <div class="card-head">
    <h2>Position Changes (OI Delta){' — ' + oi_latest['prev_date'] + ' → ' + oi_latest['date'] if oi_latest else ''}</h2>
    <div style="font-size:12px;color:#e6edf3;margin-bottom:6px;">
      {('Net OI Change: Call ' + ('+' if oi_latest['total_call_oi_delta'] >= 0 else '') + f"{int(oi_latest['total_call_oi_delta']):,d}" + ' | Put ' + ('+' if oi_latest['total_put_oi_delta'] >= 0 else '') + f"{int(oi_latest['total_put_oi_delta']):,d}" + f" | Spot ${oi_latest.get('spot', 0):.2f}") if oi_latest else 'Need 2+ snapshots to compute OI changes.'}
    </div>
    {oi_depth_html}
    {oi_insight}
    {call_dte_html}
    <details style="font-size:11px;color:#8b949e;margin:6px 0 10px;">
      <summary style="cursor:pointer;color:#58a6ff;">How to read OI Delta by depth (click to expand)</summary>
      <div style="margin-top:8px;line-height:1.7;padding:8px 10px;background:#0f1117;border-radius:6px;">
        OI Delta = <b>net new contracts created</b> between snapshots (buyers + sellers both open). The headline total is misleading — <b>where the OI lands matters more than the total</b>.
        <br><br>
        <b style="color:#3fb950;">OTM Call (above spot)</b> = directional upside bets. Retail FOMO or institutional call buying.
        <br><b style="color:#d29922;">ITM Call (below spot)</b> = usually <b>covered-call writers</b> locking gains on existing stock. Increases OI but is <b>not bullish</b> — it's profit-taking.
        <br><b style="color:#f85149;">OTM Put (below spot)</b> = downside bets or portfolio hedges.
        <br><b style="color:#f85149;">Deep OTM Put (&lt;-15% spot)</b> = systematic tail insurance. Only bought by institutions/macro funds. When this surges, <b>smart money is nervous</b>.
        <br><br>
        <b>Classic patterns:</b>
        <br>• <b>Pure bullish:</b> OTM calls dominate, puts quiet → clean upside conviction
        <br>• <b>Pure bearish:</b> OTM puts dominate, calls quiet → clean downside conviction
        <br>• <b>"Seatbelt on":</b> ITM calls + deep OTM puts both swell → institutions taking profit + hedging = late-bull warning
        <br>• <b>Rotation:</b> ITM calls closing (OI drops) + OTM calls opening (OI rises) → long positions being rolled up after a rally
      </div>
    </details>
    </div>
    <div class="tbl-wrap">
    <table class="sync-tbl">
      <tr><th style="text-align:right;">Put OI Δ</th><th style="text-align:center;">Strike · OTM</th><th style="text-align:left;">Call OI Δ</th></tr>
      {oi_rows if oi_rows else '<tr><td colspan="3" style="color:#8b949e;text-align:center;">Need 2+ snapshots</td></tr>'}
    </table>
    </div>
  </div>

  <!-- Volume Persistence -->
  <div class="card" id="walls">
    <div class="card-head">
    <h2>Persistent Volume Strikes — Call Wall / Put Wall Detection</h2>
    <div style="font-size:12px;color:#e6edf3;line-height:1.6;margin-bottom:6px;">
      Strikes with heavy volume across <b>multiple days</b> = structural positioning (dealers守在这里), not one-off flow.
      <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;">{wall_badges}</div>
    </div>
    {range_viz}
    {wall_insight}
    <details style="font-size:11px;color:#8b949e;margin-bottom:10px;">
      <summary style="cursor:pointer;color:#58a6ff;">How walls are identified (click to expand)</summary>
      <div style="margin-top:8px;line-height:1.7;padding:8px 10px;background:#0f1117;border-radius:6px;">
        <b style="color:#3fb950;">Call Wall</b> — OTM call strike (3–15% above spot) with highest <code>persistence × avg daily volume</code>.
        Dealers are short these calls → <b>short gamma</b> → hedging buy-pressure exhausts near this strike → price stalls (ceiling / resistance).
        <br><b style="color:#f85149;">Put Wall</b> — OTM put strike (3–15% below spot) with highest score. Dealers short these puts → auto-buy on dips near strike → support.
        <br><br><b>Sweet spot: 3–10% OTM</b>. Too close = magnet (gets crossed). Too far = lottery / breakout target, not active resistance.
        <br><b>Strength:</b> 85%+ persistence = <span style="color:#3fb950;">strong (structural)</span>, 65–85% = <span style="color:#d29922;">moderate</span>, 50–65% = <span style="color:#8b949e;">weak</span>.
        <br><b>Trade rules:</b> Sell CC at call wall · Sell CSP at put wall · Don't buy lottos above call wall · Watch wall migration (up = bullish re-rating, down = top signal, thinning = resistance weakening).
        <br><b>⚠️ Asymmetry warning:</b> Dense call ladder + thin put side = late-bull euphoria (no hedging demand, market complacent).
      </div>
    </details>
    </div>
    <div class="tbl-wrap">
    <table class="sync-tbl">
      <tr><th style="text-align:right;color:#f85149;">PUT Avg Vol · Active</th><th style="text-align:center;">Strike · OTM</th><th style="text-align:left;color:#3fb950;">CALL Avg Vol · Active</th></tr>
      {wall_rows_html if all_persist and wall_rows_html else '<tr><td colspan="3" style="color:#8b949e;text-align:center;">Need 2+ snapshots</td></tr>'}
    </table>
    </div>
  </div>
</div>

<!-- EM Accuracy -->
<div class="card" id="em">
  <h2>Expected Move Accuracy (backtest)</h2>
  <div style="font-size:11px;color:#8b949e;margin-bottom:8px;">
    Did the front-month Expected Move correctly contain the next day's price? EM scaled by √(calendar days) for weekend gaps. <b>Actual/EM</b> ratio: 1.0 = perfect, &lt;1 = EM over-predicts, &gt;1 = under-predicts.
  </div>
  {em_summary}
  <table>
    <tr><th style="text-align:left;">Date</th><th style="text-align:left;">Move</th><th style="text-align:left;">Range</th><th style="text-align:left;">Within EM?</th></tr>
    {em_rows if em_rows else '<tr><td colspan="4" style="color:#8b949e;text-align:center;">Need 2+ snapshots</td></tr>'}
  </table>
</div>

<div style="color:#8b949e;font-size:11px;text-align:center;margin-top:20px;padding:10px;">
  Options flow trend analysis — for educational/research purposes only. Not financial advice.
  <br>Data from yfinance. OI may be stale. Volume-based signals are more reliable.
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js" onload="initCharts()"></script>
<script>
function initCharts() {{
  // P/C Spread Chart (enhanced)
  const spreadDates = {json.dumps(all_dates)};
  const spotData = {json.dumps(spot_aligned, default=str)};
  const spreadDatasets = {json.dumps(chart_datasets, default=str)};

  // Add spot price as gray dashed line on right axis
  spreadDatasets.push({{
    label: 'Spot',
    data: spotData,
    borderColor: '#8b949e',
    borderWidth: 1.5,
    borderDash: [6, 3],
    pointRadius: 0,
    tension: 0.3,
    spanGaps: true,
    yAxisID: 'ySpot',
    order: 10
  }});

  // Build custom legend with latest values
  const latestVals = {json.dumps(latest_spread_vals)};
  const legendEl = document.getElementById('spreadLegend');
  spreadDatasets.forEach(ds => {{
    if (ds.label === 'Spot') return;
    const val = latestVals[ds.label];
    const valStr = val !== undefined ? ` ${{val > 0 ? '+' : ''}}${{val.toFixed(1)}}` : '';
    const dot = document.createElement('span');
    dot.style.cssText = 'display:inline-flex;align-items:center;gap:4px;font-size:12px;color:#e6edf3;';
    dot.innerHTML = `<span style="width:10px;height:10px;border-radius:50%;background:${{ds.borderColor}};display:inline-block;"></span>${{ds.label}}<span style="color:${{val < 0 ? '#f85149' : '#3fb950'}};font-weight:500;">${{valStr}}</span>`;
    legendEl.appendChild(dot);
  }});

  const spreadChart = new Chart(document.getElementById('spreadChart'), {{
    type: 'line',
    data: {{ labels: spreadDates, datasets: spreadDatasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      layout: {{ padding: {{ right: 44 }} }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#161b22',
          borderColor: '#30363d',
          borderWidth: 1,
          titleColor: '#e6edf3',
          bodyColor: '#e6edf3',
          padding: 10,
          callbacks: {{
            label: function(ctx) {{
              if (ctx.dataset.label === 'Spot') return 'Spot: $' + ctx.parsed.y.toFixed(2);
              const v = ctx.parsed.y;
              return ctx.dataset.label + ': ' + (v > 0 ? '+' : '') + v.toFixed(1) + ' pts';
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', font: {{ size: 11 }}, maxRotation: 0 }}, grid: {{ display: false }} }},
        y: {{
          position: 'left',
          ticks: {{ color: '#8b949e', font: {{ size: 11 }}, callback: v => (v > 0 ? '+' : '') + v }},
          grid: {{ color: 'rgba(255,255,255,0.05)' }},
          title: {{ display: true, text: 'P/C IV Spread (pts)', color: '#8b949e', font: {{ size: 12 }} }}
        }},
        ySpot: {{
          position: 'right',
          ticks: {{ color: '#8b949e', font: {{ size: 10 }}, callback: v => '$' + v }},
          grid: {{ display: false }},
          title: {{ display: true, text: 'Spot Price', color: '#8b949e', font: {{ size: 11 }} }}
        }}
      }}
    }},
    plugins: [{{
      // Zone shading: red below zero, green above
      beforeDraw(chart) {{
        const yScale = chart.scales.y;
        const ctx = chart.ctx;
        const area = chart.chartArea;
        const yZero = yScale.getPixelForValue(0);
        if (yZero < area.top || yZero > area.bottom) return;
        ctx.save();
        // Green zone above zero
        ctx.fillStyle = 'rgba(63, 185, 80, 0.12)';
        ctx.fillRect(area.left, area.top, area.right - area.left, yZero - area.top);
        // Red zone below zero
        ctx.fillStyle = 'rgba(248, 81, 73, 0.18)';
        ctx.fillRect(area.left, yZero, area.right - area.left, area.bottom - yZero);
        // Zero line
        ctx.strokeStyle = '#e6edf3';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(area.left, yZero);
        ctx.lineTo(area.right, yZero);
        ctx.stroke();
        ctx.fillStyle = '#e6edf3';
        ctx.font = '500 11px -apple-system,sans-serif';
        ctx.fillText('ZERO — call skew below this line', area.left + 6, yZero - 6);
        ctx.restore();
      }},
      // Data labels: end-anchored, with anti-overlap (sort by y, push labels apart)
      afterDatasetsDraw(chart) {{
        const ctx = chart.ctx;
        const area = chart.chartArea;
        const labels = [];
        chart.data.datasets.forEach((ds, i) => {{
          if (ds.label === 'Spot') return;
          const meta = chart.getDatasetMeta(i);
          let lastIdx = -1;
          for (let k = ds.data.length - 1; k >= 0; k--) {{
            const v = ds.data[k];
            if (v !== null && v !== 'null' && v !== undefined) {{ lastIdx = k; break; }}
          }}
          if (lastIdx < 0) return;
          const pt = meta.data[lastIdx];
          const val = ds.data[lastIdx];
          labels.push({{ x: pt.x + 4, y: pt.y, color: ds.borderColor, text: (val > 0 ? '+' : '') + parseFloat(val).toFixed(1) }});
        }});
        // Anti-collision: sort by y, enforce min spacing
        labels.sort((a, b) => a.y - b.y);
        const minGap = 14;
        for (let i = 1; i < labels.length; i++) {{
          if (labels[i].y - labels[i-1].y < minGap) labels[i].y = labels[i-1].y + minGap;
        }}
        // Clamp to chart area
        for (const l of labels) l.y = Math.max(area.top + 6, Math.min(area.bottom - 6, l.y));
        ctx.save();
        ctx.font = '600 11px -apple-system,sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        for (const l of labels) {{
          ctx.fillStyle = l.color;
          ctx.fillText(l.text, l.x, l.y);
        }}
        ctx.restore();
      }}
    }}]
  }});

  // IV + NVRP Chart
  const ivDates = {json.dumps(iv_dates)};
  new Chart(document.getElementById('ivChart'), {{
    type: 'line',
    data: {{
      labels: ivDates,
      datasets: [
        {{ label: 'ATM IV', data: {json.dumps(iv_values)}, borderColor: '#3fb950', borderWidth: 2, pointRadius: 3, tension: 0, yAxisID: 'y' }},
        {{ label: 'RV30', data: {json.dumps(rv_values)}, borderColor: '#58a6ff', borderWidth: 2, pointRadius: 3, tension: 0, yAxisID: 'y' }},
        {{ label: 'NVRP', data: {json.dumps(nvrp_values)}, borderColor: '#d29922', borderWidth: 2, pointRadius: 3, tension: 0, borderDash: [4,4], yAxisID: 'y1' }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      layout: {{ padding: {{ right: 44 }} }},
      plugins: {{ legend: {{ position: 'top', labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ display: false }} }},
        y: {{ position: 'left', ticks: {{ color: '#8b949e' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, title: {{ display: true, text: 'IV / RV (%)', color: '#8b949e' }} }},
        y1: {{ position: 'right', ticks: {{ color: '#d29922' }}, grid: {{ display: false }}, title: {{ display: true, text: 'NVRP', color: '#d29922' }} }}
      }}
    }},
    plugins: [{{
      // NVRP threshold reference lines (1.3 edge, 1.0 break-even)
      beforeDatasetsDraw(chart) {{
        const y1 = chart.scales.y1;
        if (!y1) return;
        const ctx = chart.ctx;
        const area = chart.chartArea;
        ctx.save();
        [[1.3, '#d29922', 'edge 1.3'], [1.0, '#8b949e', 'breakeven 1.0']].forEach(([v, col, lbl]) => {{
          const y = y1.getPixelForValue(v);
          if (y < area.top || y > area.bottom) return;
          ctx.strokeStyle = col;
          ctx.globalAlpha = 0.35;
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(area.left, y);
          ctx.lineTo(area.right, y);
          ctx.stroke();
          ctx.globalAlpha = 0.7;
          ctx.setLineDash([]);
          ctx.fillStyle = col;
          ctx.font = '500 10px -apple-system,sans-serif';
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';
          ctx.fillText(lbl, area.right - 4, y - 8);
        }});
        ctx.restore();
      }},
      // End-anchored labels with anti-overlap
      afterDatasetsDraw(chart) {{
        const ctx = chart.ctx;
        const area = chart.chartArea;
        const labels = [];
        chart.data.datasets.forEach((ds, i) => {{
          const meta = chart.getDatasetMeta(i);
          let lastIdx = -1;
          for (let k = ds.data.length - 1; k >= 0; k--) {{
            const v = ds.data[k];
            if (v !== null && v !== undefined) {{ lastIdx = k; break; }}
          }}
          if (lastIdx < 0) return;
          const pt = meta.data[lastIdx];
          const val = ds.data[lastIdx];
          const fmt = ds.label === 'NVRP' ? val.toFixed(2) : val.toFixed(1) + '%';
          labels.push({{ x: pt.x + 4, y: pt.y, color: ds.borderColor, text: fmt }});
        }});
        labels.sort((a, b) => a.y - b.y);
        const minGap = 14;
        for (let i = 1; i < labels.length; i++) {{
          if (labels[i].y - labels[i-1].y < minGap) labels[i].y = labels[i-1].y + minGap;
        }}
        for (const l of labels) l.y = Math.max(area.top + 6, Math.min(area.bottom - 6, l.y));
        ctx.save();
        ctx.font = '600 11px -apple-system,sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        for (const l of labels) {{
          ctx.fillStyle = l.color;
          ctx.fillText(l.text, l.x, l.y);
        }}
        ctx.restore();
      }}
    }}]
  }});
}}
if (window.Chart) initCharts();
</script>
</body></html>"""
    if lang == 'zh':
        html = apply_zh(html)
    with open(output_path, 'w') as f:
        f.write(html)


def analyze(ticker, days=10, html=False, mode='auto', lang='en'):
    """Run full trend analysis for a ticker."""
    snapshots = load_snapshots(ticker, days)
    if not snapshots:
        print(f"  No snapshots found for {ticker}")
        return None

    print(f"  Loaded {len(snapshots)} snapshots ({snapshots[0]['date']} to {snapshots[-1]['date']})")

    # Build all analyses
    pc_series = pc_spread_timeseries(snapshots)
    crossings = detect_zero_crossings(pc_series)
    oi_deltas = oi_delta(snapshots)
    persist = volume_persistence(snapshots)
    em = em_accuracy(snapshots)
    iv = iv_trend(snapshots)
    skew_pct = skew_percentile_analysis(pc_series, 'front')
    call_dte = dte_call_build_ratio(snapshots)
    gex_data = gex_walls(snapshots[-1]) if snapshots else None
    btc_data = btc_context(ticker, snapshots)

    # Current state summary
    latest = snapshots[-1]
    latest_terms = latest.get('term', [])
    front_term = next((t for t in latest_terms if t.get('dte', 0) >= 5),
                      latest_terms[0] if latest_terms else {})
    current = {
        'spot': latest['spot'],
        'rv30': latest.get('rv30', 0),
        'iv_pct_2yr': latest.get('iv_pct_2yr', 0),
        'front_atm_iv': front_term.get('atm_iv', 0),
        'front_pc_spread': front_term.get('pc_iv_spread', 0),
        'nvrp': round(front_term.get('atm_iv', 0) / latest['rv30'], 2) if latest.get('rv30', 0) > 0 else 0,
        'skew_level': skew_alert_level(front_term.get('pc_iv_spread', 0)),
    }

    # Check all expirations for negative spread
    negative_spreads = []
    for t in latest.get('term', []):
        if t.get('pc_iv_spread', 999) < 0:
            negative_spreads.append(f"{t['expiry']} (DTE {t['dte']}): {t['pc_iv_spread']:+.1f} pts")

    trend_data = {
        'ticker': ticker,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'num_snapshots': len(snapshots),
        'date_range': [snapshots[0]['date'], snapshots[-1]['date']],
        'current_state': current,
        'pc_spread_series': pc_series,
        'zero_crossings': crossings,
        'oi_deltas': oi_deltas,
        'volume_persistence': persist,
        'em_accuracy': em,
        'iv_trend': iv,
        'negative_spreads': negative_spreads,
        'skew_percentile': skew_pct,
        'call_dte_ratio': call_dte,
        'gex_walls': gex_data,
        'btc_context': btc_data,
    }

    # Save trend JSON
    trend_path = os.path.join(SNAPSHOT_DIR, f"trend-{ticker}.json")
    with open(trend_path, 'w') as f:
        json.dump(trend_data, f, default=str, indent=2)
    print(f"  Trend saved to {trend_path}")

    # Console summary
    print(f"\n  === {ticker} TREND SUMMARY ===")
    print(f"  Spot: ${current['spot']:.2f}  |  ATM IV: {current['front_atm_iv']:.1f}%  |  RV30: {current['rv30']:.1f}%  |  NVRP: {current['nvrp']:.2f}x")
    print(f"  Front P/C Spread: {current['front_pc_spread']:+.1f} pts  →  {current['skew_level']}")

    if negative_spreads:
        print(f"\n  *** CALL SKEW ALERT — P/C spread negative at: ***")
        for ns in negative_spreads:
            print(f"    {ns}")

    if crossings:
        print(f"\n  Zero-crossings detected ({len(crossings)}):")
        for c in crossings:
            print(f"    [{c['type']}] {c['bucket']} {c['date']}: {c['spread_from']:+.1f} → {c['spread_to']:+.1f}")
    else:
        print(f"\n  No zero-crossings in this period.")

    if persist['call_persistent']:
        print(f"\n  Top persistent CALL strikes:")
        for p in persist['call_persistent'][:3]:
            print(f"    ${p['strike']:.0f}  ({p['active_days']}/{p['total_days']} days, {p['total_volume']:,.0f} total vol)")

    if persist['put_persistent']:
        print(f"\n  Top persistent PUT strikes:")
        for p in persist['put_persistent'][:3]:
            print(f"    ${p['strike']:.0f}  ({p['active_days']}/{p['total_days']} days, {p['total_volume']:,.0f} total vol)")

    if em:
        within = sum(1 for e in em if e['within_em'])
        print(f"\n  EM accuracy: {within}/{len(em)} predictions within range ({within/len(em)*100:.0f}%)")

    # Generate HTML if requested
    if html:
        suffix = f"_{lang}" if lang != 'en' else ""
        html_path = os.path.join(REPORT_DIR, f"trend_{ticker}_{datetime.now().strftime('%Y-%m-%d')}{suffix}.html")
        generate_html(ticker, trend_data, html_path, mode=mode, lang=lang)
        print(f"\n  HTML report: {html_path}")
        return trend_data, html_path

    return trend_data


def main():
    parser = argparse.ArgumentParser(description='Multi-day flow trend analysis')
    parser.add_argument('tickers', nargs='*', default=['MSTR', 'TSLA'])
    parser.add_argument('--days', type=int, default=10)
    parser.add_argument('--html', action='store_true')
    parser.add_argument('--mode', choices=['auto', 'preopen', 'intraday', 'postclose'], default='auto',
                        help="Session mode (auto-detects from ET time if omitted)")
    parser.add_argument('--lang', choices=['en', 'zh'], default='en',
                        help="Dashboard language. zh writes to trend_TICKER_DATE_zh.html")
    args = parser.parse_args()

    for ticker in args.tickers:
        print(f"\n{'='*50}")
        print(f"Trend analysis: {ticker} (last {args.days} days)")
        print(f"{'='*50}")
        analyze(ticker, args.days, args.html, mode=args.mode, lang=args.lang)


if __name__ == '__main__':
    main()
