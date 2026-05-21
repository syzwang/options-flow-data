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
    if days:
        files = files[-days:]  # keep most recent N
    snapshots = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            snapshots.append(data)
    return snapshots


def front_term(snap):
    """Use first expiry with DTE >= 5 to avoid pathological 0/1-DTE IV."""
    terms = snap.get('term', [])
    return next((t for t in terms if t.get('dte', 0) >= 5), terms[0] if terms else {})


def constant_maturity_iv(snap, target_dte=14):
    """
    Variance-interpolated constant-maturity ATM/50-delta IV.

    Snapshot data stores ATM-band IV by listed expiry. To approximate a
    constant 14D 50-delta IV, interpolate total variance (IV^2 * T) between
    the expiries bracketing the target DTE, then convert back to IV.
    """
    terms = [
        {
            'dte': float(t.get('dte') or 0),
            'atm_iv': float(t.get('atm_iv') or 0),
            'expiry': t.get('expiry', ''),
        }
        for t in snap.get('term', [])
        if (t.get('dte') or 0) > 0 and (t.get('atm_iv') or 0) > 0
    ]
    terms = sorted(terms, key=lambda t: t['dte'])
    if not terms:
        return {'iv': 0, 'method': 'missing', 'target_dte': target_dte}

    target_t = target_dte / 365.0
    lower = next((t for t in reversed(terms) if t['dte'] <= target_dte), None)
    upper = next((t for t in terms if t['dte'] >= target_dte), None)

    if lower and upper and lower['dte'] != upper['dte']:
        t1 = lower['dte'] / 365.0
        t2 = upper['dte'] / 365.0
        v1 = (lower['atm_iv'] / 100.0) ** 2 * t1
        v2 = (upper['atm_iv'] / 100.0) ** 2 * t2
        w = (target_t - t1) / (t2 - t1)
        variance = v1 + w * (v2 - v1)
        iv = (variance / target_t) ** 0.5 * 100.0 if target_t > 0 and variance > 0 else 0
        return {
            'iv': iv,
            'method': 'variance_interpolated',
            'target_dte': target_dte,
            'lower_dte': lower['dte'],
            'upper_dte': upper['dte'],
            'lower_expiry': lower['expiry'],
            'upper_expiry': upper['expiry'],
        }

    nearest = min(terms, key=lambda t: abs(t['dte'] - target_dte))
    return {
        'iv': nearest['atm_iv'],
        'method': 'nearest_expiry',
        'target_dte': target_dte,
        'nearest_dte': nearest['dte'],
        'nearest_expiry': nearest['expiry'],
    }


def iv_percentile_history(snapshots):
    """
    Local ATM IV percentile from accumulated option snapshots.
    yfinance does not provide reliable 2-year historical option IV, so this
    ranks each snapshot's 14D variance-interpolated ATM IV against saved history.
    """
    values = []
    out = {}
    for snap in sorted(snapshots, key=lambda s: s.get('date', '')):
        iv = constant_maturity_iv(snap, 14).get('iv', 0) or 0
        if iv <= 0:
            continue
        values.append(iv)
        pct = sum(v <= iv for v in values) / len(values) * 100
        out[snap.get('date')] = {
            'iv_percentile': round(pct, 1),
            'iv_percentile_n': len(values),
            'iv_percentile_reliable': len(values) >= 60,
            'iv_percentile_min': round(min(values), 1),
            'iv_percentile_max': round(max(values), 1),
            'iv_percentile_median': round(sorted(values)[len(values)//2], 1),
        }
    return out


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
                               f"Historical pattern: call skew dominance flags elevated upside chase/squeeze risk."
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


def skew_index_history():
    """CBOE SKEW Index 2y daily history from yfinance.
    Macro-wide tail-hedge demand for S&P 500 — used as regime context, not stock-specific.
    """
    try:
        h = yf.Ticker('^SKEW').history(period='2y', interval='1d')
        if h.empty:
            return None
        closes = h['Close'].dropna()
        if len(closes) < 100:
            return None
        dates = [d.strftime('%Y-%m-%d') for d in closes.index]
        values = [round(float(v), 2) for v in closes.values]
        current = values[-1]
        sorted_vals = sorted(values)
        rank = sum(1 for v in sorted_vals if v <= current) / len(sorted_vals) * 100
        return {
            'dates': dates,
            'values': values,
            'current': current,
            'min': min(values),
            'max': max(values),
            'mean': round(sum(values) / len(values), 2),
            'pct_rank': round(rank, 1),
            'window_days': len(values),
        }
    except Exception:
        return None


def oi_delta_distribution(snapshots):
    """Per-strike, per-expiry OI delta from yesterday's → today's snapshot.
    Returns: { expiry: {strikes, call_delta, put_delta, spot, dte, prev_date, date} }
    """
    if len(snapshots) < 2:
        return {}
    prev, curr = snapshots[-2], snapshots[-1]
    prev_chains = prev.get('chains', {})
    curr_chains = curr.get('chains', {})
    out = {}
    for expiry in sorted(set(prev_chains.keys()) & set(curr_chains.keys())):
        pc = {r.get('strike'): int(r.get('openInterest') or 0) for r in prev_chains[expiry].get('calls', []) if r.get('strike') is not None}
        pp = {r.get('strike'): int(r.get('openInterest') or 0) for r in prev_chains[expiry].get('puts', [])  if r.get('strike') is not None}
        cc = {r.get('strike'): int(r.get('openInterest') or 0) for r in curr_chains[expiry].get('calls', []) if r.get('strike') is not None}
        cp = {r.get('strike'): int(r.get('openInterest') or 0) for r in curr_chains[expiry].get('puts', [])  if r.get('strike') is not None}
        strikes = sorted(set(pc) | set(pp) | set(cc) | set(cp))
        cd = [cc.get(k, 0) - pc.get(k, 0) for k in strikes]
        pd_ = [cp.get(k, 0) - pp.get(k, 0) for k in strikes]
        if not any(cd) and not any(pd_):
            continue
        try:
            dte = max(0, (datetime.strptime(expiry, '%Y-%m-%d').date() - datetime.strptime(curr['date'], '%Y-%m-%d').date()).days)
        except Exception:
            dte = None
        out[expiry] = {
            'strikes': strikes,
            'call_delta': cd,
            'put_delta': pd_,
            'spot': curr.get('spot', 0),
            'dte': dte,
            'prev_date': prev['date'],
            'date': curr['date'],
        }
    return out


def oi_distribution(snapshot):
    """Per-expiry OI histogram + max pain from the latest snapshot.

    Max pain = strike S that minimizes total option-holder ITM value at expiry:
        sum( call_OI(K) * max(S - K, 0) ) + sum( put_OI(K) * max(K - S, 0) )
    """
    chains = snapshot.get('chains', {})
    spot = snapshot.get('spot', 0)
    out = {}
    for expiry in sorted(chains.keys()):
        leg = chains[expiry]
        calls = leg.get('calls', [])
        puts = leg.get('puts', [])
        by_strike = {}
        for c in calls:
            k = c.get('strike')
            if k is None:
                continue
            by_strike.setdefault(k, [0, 0])[0] = int(c.get('openInterest') or 0)
        for p in puts:
            k = p.get('strike')
            if k is None:
                continue
            by_strike.setdefault(k, [0, 0])[1] = int(p.get('openInterest') or 0)
        if not by_strike:
            continue
        strikes = sorted(by_strike.keys())
        # Max pain
        best_strike, best_pain = strikes[0], float('inf')
        for S in strikes:
            pain = 0.0
            for K, (coi, poi) in by_strike.items():
                if S > K:
                    pain += (S - K) * coi
                elif K > S:
                    pain += (K - S) * poi
            if pain < best_pain:
                best_pain = pain
                best_strike = S
        # DTE
        try:
            dte = max(0, (datetime.strptime(expiry, '%Y-%m-%d').date() - datetime.strptime(snapshot['date'], '%Y-%m-%d').date()).days)
        except Exception:
            dte = None
        out[expiry] = {
            'strikes': strikes,
            'call_oi': [by_strike[k][0] for k in strikes],
            'put_oi': [by_strike[k][1] for k in strikes],
            'max_pain': best_strike,
            'spot': spot,
            'dte': dte,
        }
    return out


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
        # Annualized 7-day realized vol from log returns (V3 rule input)
        btc_rv_7d = None
        if len(btc_close) >= 8:
            import math
            rets = []
            for i in range(len(btc_close) - 7, len(btc_close)):
                p_prev = float(btc_close.iloc[i - 1])
                p_curr = float(btc_close.iloc[i])
                if p_prev > 0:
                    rets.append(math.log(p_curr / p_prev))
            if len(rets) >= 5:
                mean_r = sum(rets) / len(rets)
                var_r = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1)
                btc_rv_7d = (var_r ** 0.5) * (365 ** 0.5) * 100
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
            'btc_rv_7d': round(btc_rv_7d, 1) if btc_rv_7d is not None else None,
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


def iv_trend(snapshots, iv_pct_by_date=None):
    """Track ATM IV and RV30 across days.

    Front IV: first expiry with DTE >= 5 — avoids 0/1-DTE collapse at expiration
    (e.g., Friday-close 0DTE has near-zero IV which is pathological, not signal).
    Falls back to first available term if no DTE>=5 entry exists.
    """
    points = []
    iv_pct_by_date = iv_pct_by_date or {}
    for snap in snapshots:
        term = front_term(snap)
        front_iv = term.get('atm_iv', 0) if term else 0
        cm14 = constant_maturity_iv(snap, 14)
        iv14 = cm14.get('iv', 0) or front_iv
        pct_meta = iv_pct_by_date.get(snap.get('date'), {})
        points.append({
            'date': snap['date'],
            'spot': snap['spot'],
            'rv30': snap.get('rv30', 0),
            'rv30_pct_2yr': snap.get('rv30_pct_2yr', snap.get('iv_pct_2yr', 0)),
            'iv_percentile': pct_meta.get('iv_percentile', 0),
            'iv_percentile_n': pct_meta.get('iv_percentile_n', 0),
            'iv_percentile_reliable': pct_meta.get('iv_percentile_reliable', False),
            'iv_percentile_min': pct_meta.get('iv_percentile_min', 0),
            'iv_percentile_max': pct_meta.get('iv_percentile_max', 0),
            'iv_percentile_median': pct_meta.get('iv_percentile_median', 0),
            'front_atm_iv': front_iv,
            'atm_14d_iv': iv14,
            'atm_14d_method': cm14.get('method', ''),
            'atm_14d_lower_dte': cm14.get('lower_dte'),
            'atm_14d_upper_dte': cm14.get('upper_dte'),
            'atm_14d_nearest_dte': cm14.get('nearest_dte'),
            'nvrp': round(iv14 / snap['rv30'], 2) if snap.get('rv30', 0) > 0 else 0,
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
    (">Open Interest Distribution (per expiry)</h2>", ">持仓分布（按到期日）</h2>"),
    (">OI Delta by Strike (yesterday → today, per expiry)</h2>", ">行权价 OI 变化（昨日→今日，按到期日）</h2>"),
    ("Net OI change at each strike between",
     "各行权价持仓净变化，区间："),
    ("Green up</b> = call OI increased; direction is unconfirmed without trade prints.",
     "绿色向上</b>= call OI 增加；没有逐笔成交无法确认方向。"),
    ("Green down</b> = call OI decreased. Same logic for <b style=\"color:#f85149;\">put</b> bars.",
     "绿色向下</b>= call OI 减少。<b style=\"color:#f85149;\">put</b> 红色条同理。"),
    ("Roll detection:</b> simultaneous decrease at low strikes + increase at higher strikes = possible roll-up.",
     "Roll 检测：</b>低行权价减仓 + 高行权价增仓 = 可能向上展期。"),
    ("Per-strike call/put OI for one expiry.",
     "单一到期日的逐行权价 call/put 持仓。"),
    ("Green</b> = calls, <b style=\"color:#f85149;\">red</b> = puts.",
     "绿色</b>=call，<b style=\"color:#f85149;\">红色</b>=put。"),
    ("Max Pain</b> (yellow) = strike where most options expire worthless to holders — historical dealer-pin target.",
     "最大痛点</b>（黄）= 多数期权到期作废的行权价，历史上做市商常 pin 此价位。"),
    ("Spot</b> (cyan) = current price.",
     "现价</b>（青）= 当前股价。"),
    ("Empty zone above spot</b> = no resistance from option dealers; price can drift toward the next call cluster.",
     "现价上方真空带</b>= 期权做市商在此区间无明显抵抗，股价容易漂移至下一组 call 堆积处。"),
    (">Expiry:</label>", ">到期日：</label>"),
    (">mNAV (MSTR Premium to BTC NAV)</h2>", ">mNAV（MSTR 相对 BTC 净值溢价）</h2>"),
    ("EV-based premium MSTR trades at vs its BTC stash: <b style=\"color:#e6edf3;\">(MarketCap + Debt + Pref − Cash) / BTC Reserve</b> (matches strategy.com's published mNAV). Near NAV &lt; 1.2x · Yellow zone 1.2–1.4x · Neutral 1.4–1.8x · Rich &gt; 1.8x. Historical range ~1.0–3.5x; troughs near 1.0 have marked accumulation zones, peaks &gt;2.5 marked distribution.",
     "基于企业价值的 MSTR 相对 BTC 储备溢价：<b style=\"color:#e6edf3;\">(市值 + 债务 + 优先股 − 现金) / BTC 储备价值</b>（与 strategy.com 公布的 mNAV 口径一致）。接近净值 &lt; 1.2x · 黄区 1.2–1.4x · 中性 1.4–1.8x · 偏贵 &gt; 1.8x。历史区间约 1.0–3.5x；接近 1.0 的低点常对应吸筹区，&gt;2.5 的高点常对应派发区。"),
    ("(from yfinance close)", "（yfinance 收盘价）"),
    ("· source: strategy.com", "· 数据来源：strategy.com"),
    (">mNAV</div>", ">mNAV</div>"),
    (">Verdict</div>", ">判定</div>"),
    (">BTC Held</div>", ">BTC 持仓</div>"),
    (">BTC Reserve</div>", ">BTC 储备价值</div>"),
    (">Enterprise Value</div>", ">企业价值</div>"),
    (">Market Cap</div>", ">市值</div>"),
    (">Near NAV</div>", ">接近净值</div>"),
    (">Yellow zone</div>", ">黄区</div>"),
    (">Neutral</div>", ">中性</div>"),
    (">Rich</div>", ">偏贵</div>"),
    ("Holdings as of ", "持仓数据日期 "),
    (">Zero-Crossing Alerts (P/C Spread)</h2>", ">零轴穿越预警（P/C 偏度）</h2>"),
    (">Delta-25 Put/Call IV Spread (by expiration bucket)</h2>", ">Delta-25 Put/Call IV 偏度（按到期分桶）</h2>"),
    (">Macro Skew Context (CBOE ^SKEW, 2y)</h2>", ">宏观偏度背景（CBOE ^SKEW，2 年）</h2>"),
    ("S&amp;P 500-wide tail-hedge demand.", "标普 500 市场层面的尾部对冲需求。"),
    ("NOT stock-specific to", "并非针对个股"),
    ("— read as macro regime context. SKEW measures cost of OTM puts vs ATM SPX options.",
     "—作为宏观情绪背景使用。SKEW 衡量 OTM put 相对 ATM SPX 期权的相对价格。"),
    ("&lt;120 Complacent", "&lt;120 松懈"),
    ("120–140 Normal", "120–140 正常"),
    ("140–160 Hedged", "140–160 对冲加重"),
    ("&gt;160 Stressed", "&gt;160 压力"),
    ("Statistically: high SKEW often coincides with institutions hedging into rallies; very low SKEW marks complacent tops.",
     "统计上：高 SKEW 常对应机构在上涨中加对冲；极低 SKEW 往往出现在松懈的顶部。"),
    ("Stock-specific 2y P/C spread requires accumulating ~500 daily snapshots — currently building forward.",
     "个股 2 年 P/C 偏度需要累计约 500 个日度快照，目前正在向前积累。"),
    (">Current SKEW</div>", ">当前 SKEW</div>"),
    (">2y Percentile</div>", ">2 年分位</div>"),
    (">2y Mean</div>", ">2 年均值</div>"),
    (">2y Range</div>", ">2 年区间</div>"),
    (">Days of History</div>", ">历史天数</div>"),
    (">Complacent</div>", ">松懈</div>"),
    (">Hedged</div>", ">对冲加重</div>"),
    (">Stressed</div>", ">压力</div>"),
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
    (">ATM IV %ile</div>", ">ATM IV 分位</div>"),
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
    ("Historical pattern: when delta-25 P/C IV spread breaks below zero, upside chase/squeeze demand is elevated; use it as a risk warning, not a standalone top call.",
     "历史规律：delta-25 P/C IV spread 跌破 0 时，上行追涨/挤压需求升高；把它当风险提示，不要单独用来判顶。"),
    # ===== P/C spread chart legend =====
    ("<b style=\"color:#e6edf3;\">Spread = Put IV − Call IV</b> at delta-25 strikes. Positive = puts more expensive (normal hedging demand). Negative = calls more expensive (upside chase/squeeze demand).",
     "<b style=\"color:#e6edf3;\">偏度 = Put IV − Call IV</b>（delta-25 行权价）。正 = put 贵（正常对冲需求）；负 = call 贵（上行追涨/挤压需求）。"),
    ("<b>Safe zone</b> (spread &gt; 0) — puts cost more than calls, market pricing in downside risk = healthy",
     "<b>安全区</b>（偏度 > 0）— put 贵于 call，市场定价下行风险 = 健康"),
    ("<b>Danger zone</b> (spread &lt; 0) — calls cost more than puts, upside chase/squeeze demand elevated",
     "<b>危险区</b>（偏度 < 0）— call 贵于 put，上行追涨/挤压需求升高"),
    ("Zero line (threshold)", "零轴（临界线）"),
    ("Spot price (right axis)", "现价（右轴）"),
    # ===== NVRP insight =====
    ("<b style=\"color:#e6edf3;\">NVRP = 14D IV / RV30.</b> The 1.3 threshold exists because sellers need ~30% cushion to cover bid-ask spread, gamma risk, and hedging error. &lt;1.0 = long-vol edge; 1.0-1.3 = marginal; &gt;1.3 = short-vol edge; &gt;1.5 = strong edge.",
     "<b style=\"color:#e6edf3;\">NVRP = 14D IV / RV30</b>。1.3 门槛：卖方需 ~30% 缓冲来覆盖买卖价差、gamma 风险和对冲误差。<1.0 = 做多波动率占优；1.0-1.3 = 边际；>1.3 = 做空波动率占优；>1.5 = 强优势。"),
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
    (" ATM IV %ile (n=", " ATM IV 分位 (n="),
    ("ATM IV %ile ", "ATM IV 分位 "),
    # ===== Skew percentile banner =====
    ("<b>Extreme call skew — bottom ", "<b>极端看涨偏度 — 底部 "),
    ("<b>Heavy call skew — ", "<b>看涨偏度偏重 — "),
    ("<b>Heavy put skew — ", "<b>看跌偏度偏重 — "),
    ("<b>Extreme put skew — top ", "<b>极端看跌偏度 — 顶部 "),
    ("<b>Skew within normal band</b>", "<b>偏度在正常区间</b>"),
    ("th %ile</b> of ", "分位数</b>（"),
    ("th %ile of ", "分位数（"),
    (" and 20", " 至 20"),
    ("-day history (current ", "天历史，当前 "),
    (" vs median ", " vs 中位数 "),
    (", low ", "，最低 "),
    ("). Treat as late-cycle chase/squeeze risk — consider hedging rather than assuming an immediate top.",
     "）。把它当作晚期追涨/挤压风险 — 优先考虑对冲，不要直接假设马上见顶。"),
    ("). Bullish chase/squeeze demand is building; watch for deterioration toward extreme.",
     "）。看涨追涨/挤压需求正在升高；留意继续恶化至极端。"),
    ("). No extreme positioning signal.", "）。无极端仓位信号。"),
    ("). Fear bid on puts; historically a lagging bottom signal.",
     "）。对 put 的恐慌买盘；历史上是滞后底部信号。"),
    ("-day history. Capitulation / panic bid on puts; historically near local bottoms.",
     "天历史。恐慌性 put 买盘；历史上接近阶段底。"),
    (" only ", " 仅 "),
    (" days of history — do not use percentile in the verdict until n≥60", " 天历史 — n≥60 前不要把分位数纳入裁定"),
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
    ("). Speculative chase/squeeze demand across the full term structure. Treat as late-cycle risk and consider hedging while OTM puts are relatively cheap.",
     "）。整条期限结构都有追涨/挤压需求。视为晚期风险，考虑趁 OTM put 相对便宜时对冲。"),
    (" expirations negative (avg ", " 个到期为负（均值 "),
    (" pts). Chase/squeeze demand is building; watch for further deterioration toward extreme levels.",
     " pts）。追涨/挤压需求升高；留意继续恶化至极端。"),
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
    ("<b>Call-side OI expansion</b> — OTM call OI increased ", "<b>Call 端 OI 扩张</b> — OTM call OI 增加 "),
    (" while OTM put OI changed ", "，同时 OTM put OI 变化 "),
    (". Bullish interpretation requires confirmation from price, volume, or trade prints.",
     "。看涨解读需要价格、成交量或逐笔成交确认。"),
    ("<b>Put-side OI expansion</b> — OTM put OI increased ", "<b>Put 端 OI 扩张</b> — OTM put OI 增加 "),
    (", dwarfing call changes ", "，明显大于 call 变化 "),
    (". Bearish/hedging interpretation requires confirmation from price, volume, or trade prints.",
     "。看空/对冲解读需要价格、成交量或逐笔成交确认。"),
    ("<b>Mixed flow</b> — OTM calls ", "<b>混合流</b> — OTM calls "),
    (", OTM puts ", "，OTM puts "),
    (", ITM calls ", "，ITM calls "),
    (". No clear directional signal.", "。无明确方向信号。"),
    # OI depth badges
    ("OTM Call (&gt;spot)", "OTM Call (>现价)"),
    ("ITM Call (&lt;spot)", "ITM Call (<现价)"),
    ("OTM Put (&lt;spot)", "OTM Put (<现价)"),
    ("Deep OTM Put (&lt;-15%)", "深度 OTM Put (<-15%)"),
    ("upside OI", "上行 OI"),
    ("often covered-call writing", "常为 covered-call 写卖"),
    ("downside / hedges", "下行 / 对冲"),
    ("tail insurance", "尾部保险"),
    # ===== OI Delta depth help =====
    ("How to read OI Delta by depth (click to expand)", "如何按深度读 OI Delta（点击展开）"),
    ("OI Delta = <b>net new contracts created</b> between snapshots (buyers + sellers both open). The headline total is misleading — <b>where the OI lands matters more than the total</b>.",
     "OI Delta = 两次 snapshot 之间<b>净新建合约</b>（买卖双方都开仓）。表面总数有误导性 — <b>OI 落在哪里比总量更重要</b>。"),
    ("<b style=\"color:#3fb950;\">OTM Call (above spot)</b> = call OI increased above spot. Direction is unconfirmed without trade prints; read alongside price and volume.",
     "<b style=\"color:#3fb950;\">OTM Call（现价之上）</b> = 现价上方 call OI 增加。没有逐笔成交无法确认方向；需结合价格和成交量解读。"),
    ("<b style=\"color:#d29922;\">ITM Call (below spot)</b> = usually <b>covered-call writers</b> locking gains on existing stock. Increases OI but is <b>not bullish</b> — it's profit-taking.",
     "<b style=\"color:#d29922;\">ITM Call（现价之下）</b> = 通常是 <b>covered-call 写卖方</b>对已持股锁利。OI 增加但<b>并非看涨</b> — 是获利了结。"),
    ("<b style=\"color:#f85149;\">OTM Put (below spot)</b> = downside bets or portfolio hedges.",
     "<b style=\"color:#f85149;\">OTM Put（现价之下）</b> = 下行押注或组合对冲。"),
    ("<b style=\"color:#f85149;\">Deep OTM Put (&lt;-15% spot)</b> = systematic tail insurance. Only bought by institutions/macro funds. When this surges, <b>smart money is nervous</b>.",
     "<b style=\"color:#f85149;\">深度 OTM Put（<-15% 现价）</b> = 系统性尾部保险。仅机构/宏观基金购买。激增 = <b>smart money 紧张</b>。"),
    ("<b>Classic patterns:</b>", "<b>经典模式：</b>"),
    ("<b>Bullish candidate:</b> OTM call OI expands while puts stay quiet, especially if price/volume confirm",
     "<b>看涨候选：</b>OTM call OI 扩张且 puts 安静，尤其需要价格/成交量确认"),
    ("<b>Bearish/hedge candidate:</b> OTM put OI expands while calls stay quiet, especially if price/volume confirm",
     "<b>看空/对冲候选：</b>OTM put OI 扩张且 calls 安静，尤其需要价格/成交量确认"),
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
    ("<b>Trade rules:</b> Use walls as levels first. Short-vol structures require Playbook and Rulebook approval. Watch wall migration (up = bullish re-rating, down = top risk, thinning = resistance fading).",
     "<b>交易规则：</b>先把墙作为价位。卖波动率结构需要最终计划和规则裁定同时允许。观察墙的迁移（向上 = 看涨重定价，向下 = 顶部风险，变稀 = 阻力减弱）。"),
    ("<b>⚠️ Asymmetry warning:</b> Dense call ladder + thin put side = late-bull euphoria (no hedging demand, market complacent).",
     "<b>⚠️ 不对称警告：</b>密集 call 阶梯 + 稀薄 put 端 = 晚期牛市狂热（无对冲需求，市场自满）。"),
    # ===== Wall insight headlines =====
    ("<b>Asymmetric: call-heavy, put-thin</b>", "<b>不对称：call 重、put 稀</b>"),
    (" — call side total volume is ", " — call 端总成交量是 "),
    ("× the put side</b>", "× put 端</b>"),
    (". Dense call ladder + thin put support = <b>late-bull euphoria</b> (market complacent, no structural hedging). Expect upside to stall near call wall; downside has little cushion if sentiment flips.",
     "。密集 call 阶梯 + 稀薄 put 支撑 = <b>晚期牛市狂热</b>（市场自满、无结构对冲）。上行到 call 墙大概率停滞；情绪翻转时下行缓冲很少。"),
    ("<b>Balanced two-sided positioning</b>", "<b>双向持仓均衡</b>"),
    ("<b>Balanced two-sided walls</b>", "<b>双向结构墙均衡</b>"),
    (" — both walls structural ", " — 两道墙都结构性 "),
    (" (call ", "（call "),
    (", put ", "，put "),
    ("). Use the wall range for levels, but <b>do not treat this as a short-vol signal</b> while NVRP/GEX are unfavorable.",
     "）。用墙区间做价位参考，但在 NVRP/GEX 不利时<b>不要把它当作卖波动率信号</b>。"),
    ("). Expect range-bound trade between the walls. Premium-selling structures can be considered only if the rulebook gates pass.",
     "）。预期在墙之间区间震荡；只有规则闸门通过时才考虑卖波动率结构。"),
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
    (" pts. Historical pattern: call skew dominance flags elevated upside chase/squeeze risk.",
     " pts。历史规律：call 偏度主导提示上行追涨/挤压风险升高。"),
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
    return ""


def generate_html(ticker, trend_data, output_path, mode='auto', lang='en'):
    """Generate an interactive trend dashboard."""
    # Nav: derive date from output filename, build cross-links
    fname = os.path.basename(output_path)
    m = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
    report_date = m.group(1) if m else datetime.now().strftime('%Y-%m-%d')
    is_zh = (lang == 'zh')
    lang_suffix = '_zh' if lang == 'zh' else ''
    if is_zh:
        nav_lbl_kpi, nav_lbl_skew, nav_lbl_term, nav_lbl_walls = '概览', '偏度', '期限', '墙'
        nav_lbl_verdict = '裁定'
        nav_lbl_highlights = '要点'
        nav_lbl_playbook = '计划'
    else:
        nav_lbl_kpi, nav_lbl_skew, nav_lbl_term, nav_lbl_walls = 'KPIs', 'Skew', 'Term', 'Walls'
        nav_lbl_verdict = 'Verdict'
        nav_lbl_highlights = 'Today'
        nav_lbl_playbook = 'Playbook'

    pc = trend_data['pc_spread_series']
    crossings = trend_data['zero_crossings']
    iv = trend_data['iv_trend']
    iv_history = trend_data.get('iv_history') or iv
    oi = trend_data['oi_deltas']
    persist = trend_data['volume_persistence']
    em = trend_data['em_accuracy']
    current = trend_data['current_state']
    prev_state = trend_data.get('prev_state')
    skew_pct = trend_data.get('skew_percentile')
    call_dte = trend_data.get('call_dte_ratio')
    gex_data = trend_data.get('gex_walls')
    oi_dist = trend_data.get('oi_distribution', {}) or {}
    oi_delta_dist = trend_data.get('oi_delta_distribution', {}) or {}
    skew_idx = trend_data.get('skew_index')
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

    # Load open options positions (used for rulebook verdict card + chart overlays)
    open_positions = []
    positions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'positions.json')
    try:
        with open(positions_path) as _pf:
            _all_pos = json.load(_pf).get('positions', [])
            open_positions = [p for p in _all_pos if p.get('ticker') == ticker and p.get('status', '').startswith('open')]
    except (FileNotFoundError, json.JSONDecodeError):
        open_positions = []

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

    # IV chart data: use all saved local history, not just the report window.
    iv_dates = [p['date'] for p in iv_history]
    iv_values = [round(p.get('atm_14d_iv') or p.get('front_atm_iv', 0), 1) for p in iv_history]
    rv_values = [round(p['rv30'], 1) for p in iv_history]
    nvrp_values = [p['nvrp'] for p in iv_history]
    spot_values = [round(p['spot'], 2) for p in iv_history]

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
            events = sorted(by_bucket_cross.get(b, []), key=lambda x: x['date'], reverse=True)[:5]
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
        suppress_short_vol = ((current.get('nvrp') or 0) > 0 and (current.get('nvrp') or 0) < 1.0) or bool(gex_data and gex_data.get('regime') == 'short_gamma')

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
            if suppress_short_vol:
                wi_msg = (f"<b>Balanced two-sided walls</b> — both walls structural "
                          f"(call {c_str}, put {p_str}). Use the wall range for levels, but "
                          f"<b>do not treat this as a short-vol signal</b> while NVRP/GEX are unfavorable.")
            else:
                wi_msg = (f"<b>Balanced two-sided positioning</b> — both walls structural "
                          f"(call {c_str}, put {p_str}). Expect range-bound trade between the walls. "
                          f"Premium-selling structures can be considered only if the rulebook gates pass.")
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
            oi_msg = (f"<b>Call-side OI expansion</b> — OTM call OI increased {otm_call:+,d} while "
                      f"OTM put OI changed {otm_put:+,d}. Bullish interpretation requires confirmation from price, volume, or trade prints.")
        elif abs(otm_put) > otm_call * 2 and abs(otm_put) > 5000:
            oi_color, oi_text, oi_icon = '#f85149', '#f85149', '🔥'
            oi_msg = (f"<b>Put-side OI expansion</b> — OTM put OI increased {otm_put:+,d}, dwarfing call changes "
                      f"{otm_call:+,d}. Bearish/hedging interpretation requires confirmation from price, volume, or trade prints.")
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
        OTM Call (&gt;spot) <b style="color:#3fb950;">{otm_call:+,d}</b> <span style="color:#8b949e;">upside OI</span>
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
    cur_iv = current.get('atm_14d_iv') or current.get('front_atm_iv', 0)
    cur_rv = current.get('rv30', 0)
    cur_nvrp = current.get('nvrp', 0)
    cur_iv_pct = current.get('iv_percentile', 0)
    cur_iv_pct_n = current.get('iv_percentile_n', 0)
    cur_iv_pct_reliable = current.get('iv_percentile_reliable', False)
    cur_iv_pct_min = current.get('iv_percentile_min', 0)
    cur_iv_pct_max = current.get('iv_percentile_max', 0)
    cur_iv_pct_median = current.get('iv_percentile_median', 0)
    long_vol_edge = cur_nvrp > 0 and cur_nvrp < 1.0
    short_gamma_regime = bool(gex_data and gex_data.get('regime') == 'short_gamma')
    short_vol_suppressed = long_vol_edge or short_gamma_regime
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
        if cur_iv_pct > 0 and cur_iv_pct_reliable:
            pct_str = (
                f' &nbsp;|&nbsp; ATM IV 分位: <b>{cur_iv_pct:.0f}</b>'
                if is_zh else
                f' &nbsp;|&nbsp; ATM IV %ile: <b>{cur_iv_pct:.0f}</b>'
            )
        elif cur_iv_pct > 0:
            pct_str = (
                f' &nbsp;|&nbsp; 本地 IV 样本 <b>n={cur_iv_pct_n}</b>，暂不用于裁定'
                if is_zh else
                f' &nbsp;|&nbsp; local IV sample <b>n={cur_iv_pct_n}</b>, not used for verdict'
            )
        nvrp_insight = f"""
  <div style="margin:10px 0;padding:10px 12px;border-left:3px solid {nv_color};background:{nv_color}11;border-radius:4px;font-size:12px;line-height:1.5;">
    <span style="margin-right:6px;">{nv_icon}</span><span style="color:{nv_text};">{nv_msg}</span>{pct_str}
  </div>"""

    # --- Compact IV Term Structure: latest ATM IV per bucket ---
    term_compact_html = ""
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

        if is_zh:
            if slope > 5:
                ts_color, ts_icon, ts_msg = '#f85149', '🔥', f"<b>短端压力</b> — Front 比 Far 高 {slope:+.1f} 点；短期期权贵，市场在定价事件/跳空风险。"
            elif slope > 1:
                ts_color, ts_icon, ts_msg = '#d29922', '⚠', f"<b>轻微短端升水</b> — Front−Far {slope:+.1f} 点；短期有焦虑，但未到压力状态。"
            elif slope < -5:
                ts_color, ts_icon, ts_msg = '#3fb950', '✓', f"<b>远端升水较陡</b> — Front 比 Far 低 {abs(slope):.1f} 点；短期期权相对便宜，但也可能表示短期缺少事件。"
            elif slope < -1:
                ts_color, ts_icon, ts_msg = '#7ee787', '✓', f"<b>正常远端升水</b> — Front−Far {slope:+.1f} 点；期限结构健康。"
            else:
                ts_color, ts_icon, ts_msg = '#8b949e', '·', f"<b>期限平坦</b> — Front−Far {slope:+.1f} 点；没有明显期限偏好。"
        else:
            if slope > 5:
                ts_color, ts_icon, ts_msg = '#f85149', '🔥', f"<b>Front-end stress</b> — Front is {slope:+.1f} pts above Far; short-dated options are rich and pricing event/gap risk."
            elif slope > 1:
                ts_color, ts_icon, ts_msg = '#d29922', '⚠', f"<b>Mild backwardation</b> — Front−Far {slope:+.1f} pts; some near-term anxiety, not a stress regime."
            elif slope < -5:
                ts_color, ts_icon, ts_msg = '#3fb950', '✓', f"<b>Steep contango</b> — Front is {abs(slope):.1f} pts below Far; short-dated options are relatively cheap, but may simply lack catalyst."
            elif slope < -1:
                ts_color, ts_icon, ts_msg = '#7ee787', '✓', f"<b>Normal contango</b> — Front−Far {slope:+.1f} pts; healthy term structure."
            else:
                ts_color, ts_icon, ts_msg = '#8b949e', '·', f"<b>Flat curve</b> — Front−Far {slope:+.1f} pts; no strong term preference."

        bar_rows = ""
        for bucket in ['front', '2w', '1m', '2m', 'far']:
            if bucket not in term_struct:
                continue
            iv_val = term_struct[bucket]
            bar_w = int(iv_val / max_iv * 100) if max_iv > 0 else 0
            bar_rows += (
                f"<div style='display:grid;grid-template-columns:86px 52px 1fr;gap:8px;align-items:center;padding:2px 0;font-size:11px;'>"
                f"<div style='color:#e6edf3;font-weight:500;'>{bucket_labels[bucket]}</div>"
                f"<div style='font-family:ui-monospace,SFMono-Regular,monospace;color:#e6edf3;text-align:right;'>{iv_val:.1f}%</div>"
                f"<div><div style='height:10px;width:{bar_w}%;background:linear-gradient(90deg,#58a6ff,#a371f7);border-radius:2px;min-width:4px;'></div></div>"
                f"</div>"
            )

        term_title = 'IV 期限结构' if is_zh else 'IV Term Structure'
        term_hint = '辅助：用于判断短端是否因事件变贵，以及选择期限。' if is_zh else 'Auxiliary: helps identify front-end event premium and choose tenor.'
        term_compact_html = f"""
  <div style="margin-top:12px;background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:12px;">
    <div style="display:flex;align-items:center;gap:10px;justify-content:space-between;flex-wrap:wrap;margin-bottom:8px;">
      <div>
        <div style="font-size:12px;color:#e6edf3;font-weight:650;">{term_title}</div>
        <div style="font-size:10px;color:#8b949e;margin-top:2px;">{term_hint}</div>
      </div>
      <div style="font-size:11px;color:{ts_color};font-weight:600;">{ts_icon} {ts_msg}</div>
    </div>
    <div>{bar_rows}</div>
  </div>
"""

    # --- IV dashboard: keep level, local percentile, realized vol, and curve separate ---
    def _fmt_signed(v, digits=1, suffix=''):
        if v is None:
            return ''
        if abs(v) < 0.05:
            return f'<span style="color:#8b949e;font-size:11px;">{"持平" if is_zh else "flat"}</span>'
        color = '#3fb950' if v > 0 else '#f85149'
        return f'<span style="color:{color};font-size:11px;">{v:+.{digits}f}{suffix}</span>'

    iv_day_change = None
    rv_day_change = None
    nvrp_day_change = None
    if prev_state:
        iv_day_change = cur_iv - (prev_state.get('atm_14d_iv') or prev_state.get('front_atm_iv') or 0)
        rv_day_change = cur_rv - (prev_state.get('rv30') or 0)
        nvrp_day_change = cur_nvrp - (prev_state.get('nvrp') or 0)

    if cur_iv_pct_reliable:
        pct_value = f'{cur_iv_pct:.0f}'
        pct_note = (f'n={cur_iv_pct_n}，可参与规则裁定' if is_zh
                    else f'n={cur_iv_pct_n}, eligible for rule verdicts')
        pct_color = '#e6edf3'
    else:
        pct_value = '样本不足' if is_zh else 'insufficient'
        pct_note = (f'本地 n={cur_iv_pct_n}；当前 {cur_iv_pct:.1f}% 只作参考，不当长期 IV 分位'
                    if is_zh else
                    f'local n={cur_iv_pct_n}; current {cur_iv_pct:.1f}% is reference only, not long-run IV percentile')
        pct_color = '#d29922'

    if cur_iv_pct_min and cur_iv_pct_max:
        local_range = (f'本地 14D IV 区间 {cur_iv_pct_min:.1f}%–{cur_iv_pct_max:.1f}%，中位 {cur_iv_pct_median:.1f}%'
                       if is_zh else
                       f'Local 14D IV range {cur_iv_pct_min:.1f}%–{cur_iv_pct_max:.1f}%, median {cur_iv_pct_median:.1f}%')
    else:
        local_range = '本地 14D IV 区间缺失' if is_zh else 'Local 14D IV range unavailable'

    if cur_nvrp >= 1.3:
        edge_text = '卖波动率有溢价垫' if is_zh else 'premium-selling cushion'
        edge_color = '#3fb950'
    elif cur_nvrp >= 1.0:
        edge_text = '卖波动率边际' if is_zh else 'marginal premium edge'
        edge_color = '#d29922'
    elif cur_nvrp > 0:
        edge_text = '买波动率占优' if is_zh else 'long-vol edge'
        edge_color = '#f85149'
    else:
        edge_text = 'NVRP 缺失' if is_zh else 'NVRP unavailable'
        edge_color = '#8b949e'

    term_dash = '期限结构缺失' if is_zh else 'term structure unavailable'
    term_dash_note = ''
    term_color = '#8b949e'
    if len(term_struct) >= 2:
        _front = term_struct.get('front', 0)
        _far = (term_struct.get('2m') or term_struct.get('far') or term_struct.get('1m') or term_struct.get('2w') or _front)
        _slope = _front - _far
        if _slope > 1:
            term_dash = '近月升水' if is_zh else 'backwardation'
            term_color = '#f85149' if _slope > 5 else '#d29922'
        elif _slope < -1:
            term_dash = '远月升水' if is_zh else 'contango'
            term_color = '#3fb950'
        else:
            term_dash = '期限平坦' if is_zh else 'flat curve'
            term_color = '#8b949e'
        term_dash_note = (f'Front−Far {_slope:+.1f} pts'
                          if not is_zh else f'Front−Far {_slope:+.1f} 点')

    iv_dashboard_title = 'Historical 14D 50Δ IV Trend（方差插值）' if is_zh else 'Historical 14D 50-Delta IV Trend (Variance Interpolated)'
    iv_dashboard_lead = (
        '使用快照里的到期 IV 做总方差插值，近似 14D 50Δ/ATM IV；本地样本不足两年时，只作为本地历史。'
        if is_zh else
        'Uses listed-expiry IV from snapshots and interpolates total variance to approximate 14D 50-delta/ATM IV; until local history reaches two years, treat it as local history only.'
    )
    iv_dashboard_html = f"""
  <div style="font-size:11px;color:#8b949e;margin-bottom:10px;line-height:1.6;">{iv_dashboard_lead}</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:12px;">
    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:10px;">
      <div style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">14D 50Δ/ATM IV</div>
      <div style="font-size:20px;color:#e6edf3;font-weight:650;margin-top:4px;">{cur_iv:.1f}%</div>
      <div style="margin-top:3px;">{_fmt_signed(iv_day_change, 1, ' pp')}</div>
    </div>
    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:10px;">
      <div style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">RV30</div>
      <div style="font-size:20px;color:#e6edf3;font-weight:650;margin-top:4px;">{cur_rv:.1f}%</div>
      <div style="margin-top:3px;">{_fmt_signed(rv_day_change, 1, ' pp')}</div>
    </div>
    <div style="background:#0d1117;border:1px solid {edge_color}44;border-radius:8px;padding:10px;">
      <div style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">NVRP</div>
      <div style="font-size:20px;color:{edge_color};font-weight:650;margin-top:4px;">{cur_nvrp:.2f}x</div>
      <div style="font-size:11px;color:#8b949e;margin-top:3px;">{edge_text} {_fmt_signed(nvrp_day_change, 2)}</div>
    </div>
    <div style="background:#0d1117;border:1px solid {pct_color}44;border-radius:8px;padding:10px;">
      <div style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">{'本地 14D IV 分位' if is_zh else 'Local 14D IV %ile'}</div>
      <div style="font-size:20px;color:{pct_color};font-weight:650;margin-top:4px;">{pct_value}</div>
      <div style="font-size:11px;color:#8b949e;margin-top:3px;">{pct_note}</div>
    </div>
    <div style="background:#0d1117;border:1px solid {term_color}44;border-radius:8px;padding:10px;">
      <div style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">{'期限结构' if is_zh else 'Term Structure'}</div>
      <div style="font-size:20px;color:{term_color};font-weight:650;margin-top:4px;">{term_dash}</div>
      <div style="font-size:11px;color:#8b949e;margin-top:3px;">{term_dash_note}</div>
    </div>
  </div>
  <div style="font-size:11px;color:#8b949e;margin-bottom:8px;">{local_range}</div>"""

    # Auto-generated current-state insight
    spread_insight = ""
    skew_decision_html = ""
    if latest_spread_vals:
        vals = list(latest_spread_vals.values())
        neg_count = sum(1 for v in vals if v < 0)
        total = len(vals)
        extreme_neg = min(vals)
        avg_val = sum(vals) / len(vals)
        if is_zh:
            if neg_count == total:
                state_title = '全期限 Call 偏度'
                state_text = f'{neg_count}/{total} 个到期桶为负，均值 {avg_val:+.1f} 点，最低 {extreme_neg:+.1f} 点。'
                meaning = '25Δ call IV 高于 put IV，说明上行追涨/挤压需求更强。'
                use_it = '把它当作上行 squeeze 风险提示；突破 GEX Call 墙时，不要低估加速风险。'
                avoid_it = '不要把负偏度单独当作顶部信号，也不要因为 call 贵就裸卖 call。'
                decision_color = '#d29922'
            elif neg_count >= total / 2:
                state_title = '部分期限 Call 偏度'
                state_text = f'{neg_count}/{total} 个到期桶为负，均值 {avg_val:+.1f} 点。'
                meaning = '追涨需求集中在部分期限，还不是全期限一致。'
                use_it = '重点看负偏度集中在哪个 DTE；短期负偏度更像事件/挤压，远期负偏度更结构化。'
                avoid_it = '不要把所有到期日混成一个结论；先区分短期和远期。'
                decision_color = '#d29922'
            elif neg_count > 0:
                state_title = '局部 Call 偏度'
                state_text = f'{neg_count}/{total} 个到期桶为负，最低 {extreme_neg:+.1f} 点。'
                meaning = '只有局部期限显示追涨需求，整体还不是极端状态。'
                use_it = '只对相关到期日提高警惕；其它期限仍按主规则执行。'
                avoid_it = '不要把局部异常放大成全局风险。'
                decision_color = '#58a6ff'
            else:
                state_title = 'Put 偏度正常'
                state_text = f'{total}/{total} 个到期桶为正，均值 {avg_val:+.1f} 点。'
                meaning = 'put IV 高于 call IV，属于常见保护性需求。'
                use_it = '偏度本身不阻止卖波动率；仍需看 NVRP、GEX 和事件风险。'
                avoid_it = '不要只因偏度正常就忽略价格/GEX 风险。'
                decision_color = '#3fb950'
            panel_labels = ('当前状态', '含义', '怎么用', '不要这样用')
        else:
            if neg_count == total:
                state_title = 'Full-curve call skew'
                state_text = f'{neg_count}/{total} buckets are negative, avg {avg_val:+.1f} pts, low {extreme_neg:+.1f} pts.'
                meaning = '25Δ call IV is above put IV: upside chase/squeeze demand is stronger.'
                use_it = 'Treat it as upside squeeze risk; if price breaks a GEX call wall, do not underestimate acceleration risk.'
                avoid_it = 'Do not use negative skew alone as a top signal, and do not sell naked calls just because calls are expensive.'
                decision_color = '#d29922'
            elif neg_count >= total / 2:
                state_title = 'Partial call skew'
                state_text = f'{neg_count}/{total} buckets are negative, avg {avg_val:+.1f} pts.'
                meaning = 'Chase demand is present in some expiries, but not full-curve confirmation.'
                use_it = 'Check which DTE owns the negative skew; short-dated skew is more event/squeeze-like, far skew is more structural.'
                avoid_it = 'Do not collapse all expiries into one conclusion; separate short and far dates.'
                decision_color = '#d29922'
            elif neg_count > 0:
                state_title = 'Localized call skew'
                state_text = f'{neg_count}/{total} buckets are negative, low {extreme_neg:+.1f} pts.'
                meaning = 'Only a local expiry zone shows chase demand; the whole curve is not extreme.'
                use_it = 'Raise caution only for the affected expiries; keep other expiries under the main rules.'
                avoid_it = 'Do not inflate a local anomaly into a whole-curve risk call.'
                decision_color = '#58a6ff'
            else:
                state_title = 'Normal put skew'
                state_text = f'{total}/{total} buckets are positive, avg {avg_val:+.1f} pts.'
                meaning = 'Put IV is above call IV, typical of protection demand.'
                use_it = 'Skew itself is not blocking short-vol; still check NVRP, GEX, and event risk.'
                avoid_it = 'Do not ignore price/GEX risk just because skew is normal.'
                decision_color = '#3fb950'
            panel_labels = ('Current', 'Meaning', 'How to use', 'Do not use as')

        def _skew_panel(label, body, accent):
            return (
                f'<div style="background:#0d1117;border:1px solid {accent}33;border-radius:8px;padding:10px 12px;min-height:82px;">'
                f'<div style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;">{label}</div>'
                f'<div style="font-size:12px;color:#e6edf3;line-height:1.5;">{body}</div></div>'
            )

        zero_details_label = '历史零轴穿越明细' if is_zh else 'Historical zero-crossing details'
        skew_decision_html = f"""
  <div style="margin:10px 0 12px;border:1px solid {decision_color}44;background:{decision_color}0d;border-radius:10px;padding:12px;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap;">
      <div style="color:{decision_color};font-weight:650;font-size:14px;">{state_title}</div>
      <div style="color:#8b949e;font-size:12px;">{state_text}</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;">
      {_skew_panel(panel_labels[1], meaning, decision_color)}
      {_skew_panel(panel_labels[2], use_it, '#58a6ff')}
      {_skew_panel(panel_labels[3], avoid_it, '#f85149')}
    </div>
  </div>
  <details style="margin:8px 0 12px;font-size:11px;color:#8b949e;">
    <summary style="cursor:pointer;color:#58a6ff;">{zero_details_label}</summary>
    <div style="margin-top:8px;">{alert_html}</div>
  </details>"""

        if neg_count == total and extreme_neg < -10:
            color, text_color, icon, msg = '#f85149', '#f85149', '🔥', f'<b>EXTREME CALL SKEW</b> — all {total} buckets in danger zone (avg {avg_val:+.1f} pts, low {extreme_neg:+.1f}). Speculative chase/squeeze demand across the full term structure. Treat as late-cycle risk and consider hedging while OTM puts are relatively cheap.'
        elif neg_count == total:
            color, text_color, icon, msg = '#d29922', '#d29922', '⚠️', f'<b>Call skew across all buckets</b> — all {total} expirations negative (avg {avg_val:+.1f} pts). Chase/squeeze demand is building; watch for further deterioration toward extreme levels.'
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
                f"<b>Extreme call skew — bottom {skew_pct['percentile']:.0f}th %ile</b> of {skew_pct['sample_size']}-day history (current {skew_pct['current']:+.1f} vs median {skew_pct['median']:+.1f}, low {skew_pct['min']:+.1f}). Treat as late-cycle chase/squeeze risk — consider hedging rather than assuming an immediate top."),
            'heavy_call_skew': ('#d29922', '#d29922', '⚠️',
                f"<b>Heavy call skew — {skew_pct['percentile']:.0f}th %ile</b> of {skew_pct['sample_size']}-day history (current {skew_pct['current']:+.1f} vs median {skew_pct['median']:+.1f}). Bullish chase/squeeze demand is building; watch for deterioration toward extreme."),
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
            warn = f' <span style="color:#8b949e;font-size:11px;">· ⚠ only {skew_pct["sample_size"]} days of history — do not use percentile in the verdict until n≥60</span>'
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
        if is_zh:
            regime_map = {
                'long_gamma': ('#3fb950', '#a8e6b8', '磁吸',
                    '<b>做市商净多 Gamma</b> — 价格更容易被高 gamma 行权价磁吸，波动被压制，墙之间震荡概率更高。'),
                'short_gamma': ('#f85149', '#f85149', '放大',
                    '<b>做市商净空 Gamma</b> — 价格波动容易被放大；突破变 squeeze，跌破变加速下行。裸卖波动率风险高。'),
                'neutral': ('#d29922', '#d29922', '偏软',
                    '<b>混合 Gamma 环境</b> — call 与 put 大致平衡；墙偏软，更多作为价位参考。'),
            }
        else:
            regime_map = {
                'long_gamma': ('#3fb950', '#a8e6b8', 'MAGNET',
                    '<b>Dealers net long gamma</b> — price tends to pin toward high-gamma strikes; moves are suppressed and wall-to-wall chop is more likely.'),
                'short_gamma': ('#f85149', '#f85149', 'AMPLIFY',
                    '<b>Dealers net short gamma</b> — moves tend to amplify; breakouts can squeeze and breakdowns can accelerate. Naked short-vol risk is high.'),
                'neutral': ('#d29922', '#d29922', 'SOFT',
                    '<b>Mixed gamma regime</b> — calls and puts are roughly balanced; walls are softer and mainly act as reference levels.'),
            }
        regime_names = {
            'long_gamma': '净多 Gamma' if is_zh else 'Long Gamma',
            'short_gamma': '净空 Gamma' if is_zh else 'Short Gamma',
            'neutral': '混合 Gamma' if is_zh else 'Mixed Gamma',
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
        top_call = calls[0] if calls else None
        top_put = puts[0] if puts else None
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

        def _dist_text(w):
            if not w:
                return 'n/a'
            return f'{w["otm_pct"]:.1f}%'

        if is_zh:
            call_level = f'${top_call["strike"]:.0f}' if top_call else '无'
            put_level = f'${top_put["strike"]:.0f}' if top_put else '无'
            call_note = (f'上方 {call_level}，距现价 {_dist_text(top_call)}。先当阻力；有效突破后，做市商对冲可能追买，形成 squeeze。'
                         if top_call else '25% OTM 内没有明显 Call GEX 墙；上方阻力来自期权对冲的信号较弱。')
            put_note = (f'下方 {put_level}，距现价 {_dist_text(top_put)}。先当支撑；有效跌破后，对冲流可能放大下跌。'
                        if top_put else '25% OTM 内没有明显 Put GEX 墙；下方期权对冲支撑较弱。')
            if gex_data['regime'] == 'long_gamma':
                trade_note = '价格在 Call/Put 墙之间更容易震荡；只有 NVRP、偏度和规则裁定也支持时，才考虑收租结构。'
            elif gex_data['regime'] == 'short_gamma':
                trade_note = '不要把墙当成稳固区间；墙被穿越时优先防加速，偏向限亏/买波动率结构。'
            else:
                trade_note = '墙只作为挂单/止损/止盈参考；方向仍看偏度、14D IV vs RV、BTC/mNAV。'
            labels = {
                'title': '做市商 Gamma 墙（GEX）',
                'lead': 'GEX 衡量做市商为了对冲期权 Gamma，最可能在哪些行权价附近买卖股票。它是盘中支撑/阻力与加速风险的参考，不是方向预测。',
                'regime': '净 Gamma 环境',
                'call': '上方 Call 墙',
                'put': '下方 Put 墙',
                'trade': '交易用法',
                'ladder': 'GEX 价位梯',
                'call_hdr': '上方阻力 / 突破触发',
                'put_hdr': '下方支撑 / 跌破触发',
                'gex': 'GEX',
                'details': 'GEX 墙 vs 成交量墙（点击）',
            }
        else:
            call_level = f'${top_call["strike"]:.0f}' if top_call else 'None'
            put_level = f'${top_put["strike"]:.0f}' if top_put else 'None'
            call_note = (f'Nearest upside wall {call_level}, { _dist_text(top_call) } from spot. Treat as resistance first; a clean break can force dealer chase-buying and squeeze.'
                         if top_call else 'No clear call GEX wall within 25% OTM; option-hedging resistance above spot is weak.')
            put_note = (f'Nearest downside wall {put_level}, { _dist_text(top_put) } from spot. Treat as support first; a clean break can amplify downside hedging flow.'
                        if top_put else 'No clear put GEX wall within 25% OTM; option-hedging support below spot is weak.')
            if gex_data['regime'] == 'long_gamma':
                trade_note = 'Wall-to-wall chop is more likely; only consider premium-selling structures when NVRP, skew, and rulebook also approve.'
            elif gex_data['regime'] == 'short_gamma':
                trade_note = 'Do not treat walls as a stable range; when a wall breaks, prioritize acceleration risk and defined-risk or long-vol structures.'
            else:
                trade_note = 'Use walls for entries, exits, and stops; directional bias still comes from skew, 14D IV vs RV, BTC/mNAV.'
            labels = {
                'title': 'Dealer Gamma Exposure (GEX) Walls',
                'lead': 'GEX estimates where dealers are most likely to buy or sell stock while hedging option gamma. Use it as intraday support/resistance and acceleration-risk context, not a directional forecast.',
                'regime': 'Net Gamma Regime',
                'call': 'Upside Call Wall',
                'put': 'Downside Put Wall',
                'trade': 'Trading Use',
                'ladder': 'GEX Ladder',
                'call_hdr': 'Upside resistance / breakout trigger',
                'put_hdr': 'Downside support / breakdown trigger',
                'gex': 'GEX',
                'details': 'How GEX walls differ from volume walls (click)',
            }

        def _gex_tile(icon, label, value, body, color):
            return f"""
    <div style="background:#0d1117;border:1px solid {color}44;border-radius:10px;padding:12px;min-height:112px;">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:8px;">
        <span style="display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:8px;background:{color}18;color:{color};font-weight:800;font-size:15px;">{icon}</span>
        <div>
          <div style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
          <div style="font-size:18px;color:{color};font-weight:650;margin-top:2px;">{value}</div>
        </div>
      </div>
      <div style="font-size:12px;color:#c9d1d9;line-height:1.5;">{body}</div>
    </div>"""

        gex_tiles_html = (
            _gex_tile('G', labels['regime'], regime_names.get(gex_data['regime'], gex_data['regime']), gmsg, gc) +
            _gex_tile('↑', labels['call'], call_level, call_note, '#3fb950') +
            _gex_tile('↓', labels['put'], put_level, put_note, '#f85149') +
            _gex_tile('↔', labels['trade'], 'Plan' if not is_zh else '计划', trade_note, '#58a6ff')
        )

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
  <h2>{labels['title']}</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:6px;line-height:1.6;">
    {labels['lead']}
    <br>{'净 GEX' if is_zh else 'Net GEX'}: <b style="color:{gc};">{_fmt_gex(gex_data['net_gex'])}</b> (call {_fmt_gex(gex_data['total_call_gex'])} + put {_fmt_gex(gex_data['total_put_gex'])}).
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:12px 0;">
    {gex_tiles_html}
  </div>
  <div style="margin-top:12px;">
    <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">{labels['ladder']}</div>
    <div style="display:flex;justify-content:space-between;align-items:center;font-size:10px;letter-spacing:0.5px;color:#8b949e;margin-bottom:4px;padding:0 2px;">
      <span>Strike · OTM</span><span style="color:#3fb950;">↑ {labels['call_hdr']}</span><span>{labels['gex']}</span>
    </div>
    {call_ladder}
    {spot_divider}
    {put_ladder}
    <div style="display:flex;justify-content:space-between;align-items:center;font-size:10px;letter-spacing:0.5px;color:#8b949e;margin-top:4px;padding:0 2px;">
      <span>&nbsp;</span><span style="color:#f85149;">↓ {labels['put_hdr']}</span><span>&nbsp;</span>
    </div>
  </div>
  <details style="font-size:11px;color:#8b949e;margin-top:10px;">
    <summary style="cursor:pointer;color:#58a6ff;">{labels['details']}</summary>
    <div style="margin-top:8px;line-height:1.7;padding:8px 10px;background:#0f1117;border-radius:6px;">
      <b style="color:#e6edf3;">Volume walls</b> (below, Persistent Volume Strikes) show where retail/institutional <b>flow concentrates</b> — backward-looking conviction.
      <br><b style="color:#e6edf3;">GEX walls</b> (this card) show where <b>dealer hedging pressure</b> is largest — forward-looking, drives actual intraday price behavior.
      <br><br>A strike can be a volume wall without being a GEX wall (e.g. far-dated OI has low gamma). The <b>GEX wall is what actually acts as resistance/support intraday</b> because dealers must trade stock to hedge. When they agree, the wall is robust. When they disagree, trust GEX.
    </div>
  </details>
</div>"""

    # --- OI Distribution histogram (per-expiry, with max pain) ---
    oi_dist_html = ""
    oi_dist_json = "{}"
    oi_dist_default = ""
    if oi_dist:
        expiries_sorted = sorted(oi_dist.keys())
        # Default to the first expiry with DTE >= 3 (skip 0DTE / weekend stale)
        oi_dist_default = next((e for e in expiries_sorted if (oi_dist[e].get('dte') or 0) >= 3), expiries_sorted[0])
        opts = []
        for e in expiries_sorted:
            d = oi_dist[e]
            dte_str = f" ({d['dte']}d)" if d.get('dte') is not None else ""
            sel = " selected" if e == oi_dist_default else ""
            opts.append(f'<option value="{e}"{sel}>{e}{dte_str}</option>')
        oi_dist_json = json.dumps(oi_dist)
        oi_dist_html = f"""
<div class="card" id="oidist">
  <h2>Open Interest Distribution (per expiry)</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:10px;line-height:1.6;">
    Per-strike call/put OI for one expiry. <b style="color:#3fd058;">Green</b> = calls, <b style="color:#f85149;">red</b> = puts. <b style="color:#e6edf3;">Max Pain</b> (yellow) = strike where most options expire worthless to holders — historical dealer-pin target. <b style="color:#e6edf3;">Spot</b> (cyan) = current price. <b>Empty zone above spot</b> = no resistance from option dealers; price can drift toward the next call cluster.
  </div>
  <div style="margin-bottom:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
    <label style="font-size:12px;color:#8b949e;">Expiry:</label>
    <select id="oiDistExpiry" style="background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 10px;font-size:12px;font-family:inherit;">
      {''.join(opts)}
    </select>
    <span id="oiDistMeta" style="font-size:11px;color:#8b949e;"></span>
  </div>
  <div class="chart-wrap-tall">
    <canvas id="oiDistChart"></canvas>
  </div>
</div>"""

    # --- Macro Skew Context card (CBOE ^SKEW 2y) ---
    skew_card_html = ""
    skew_idx_json = "null"
    if skew_idx:
        sv = skew_idx['current']
        if sv < 120:
            skew_color, skew_label = '#3fb950', 'Complacent'
        elif sv < 140:
            skew_color, skew_label = '#d29922', 'Normal'
        elif sv < 160:
            skew_color, skew_label = '#f0883e', 'Hedged'
        else:
            skew_color, skew_label = '#f85149', 'Stressed'
        skew_idx_json = json.dumps({
            'dates': skew_idx['dates'],
            'values': skew_idx['values'],
        })
        skew_card_html = f"""
<div class="card" id="skew">
  <h2>Macro Skew Context (CBOE ^SKEW, 2y)</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:10px;line-height:1.6;">
    S&amp;P 500-wide tail-hedge demand. <b style="color:#e6edf3;">NOT stock-specific to {ticker}</b> — read as macro regime context. SKEW measures cost of OTM puts vs ATM SPX options. <b style="color:#3fb950;">&lt;120 Complacent</b> · <b style="color:#d29922;">120–140 Normal</b> · <b style="color:#f0883e;">140–160 Hedged</b> · <b style="color:#f85149;">&gt;160 Stressed</b>. Statistically: high SKEW often coincides with institutions hedging into rallies; very low SKEW marks complacent tops.
    <br><span style="color:#6e7681;">Stock-specific 2y P/C spread requires accumulating ~500 daily snapshots — currently building forward.</span>
  </div>
  <div class="kpi-row" style="grid-template-columns:repeat(auto-fit, minmax(140px,1fr));">
    <div class="kpi">
      <div class="kpi-label">Current SKEW</div>
      <div class="kpi-value" style="color:{skew_color};">{sv:.1f}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Verdict</div>
      <div class="kpi-value" style="color:{skew_color};">{skew_label}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">2y Percentile</div>
      <div class="kpi-value">{skew_idx['pct_rank']:.0f}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">2y Mean</div>
      <div class="kpi-value">{skew_idx['mean']:.1f}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">2y Range</div>
      <div class="kpi-value" style="font-size:14px;">{skew_idx['min']:.1f} – {skew_idx['max']:.1f}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Days of History</div>
      <div class="kpi-value">{skew_idx['window_days']}</div>
    </div>
  </div>
  <div class="chart-wrap"><canvas id="skewChart"></canvas></div>
</div>"""

    # --- OI Delta Distribution histogram (yesterday → today, per expiry) ---
    # Build positions JSON for chart overlay (only short legs with strike on this ticker)
    positions_overlay = []
    for p in open_positions:
        if p.get('strike') is not None:
            label = p.get('label')
            if not label:
                struct_short = {'covered_call': 'CC', 'cash_secured_put': 'CSP', 'naked_put': 'NP', 'naked_call': 'NC'}.get(p.get('structure', ''), p.get('structure', '').upper())
                side = p.get('side', '').upper()
                label = f"{side} {struct_short} ${p['strike']}"
            positions_overlay.append({'strike': p['strike'], 'label': label})
    positions_json = json.dumps(positions_overlay)

    oi_delta_html = ""
    oi_delta_json = "{}"
    if oi_delta_dist:
        deltas_sorted = sorted(oi_delta_dist.keys())
        # Default: first expiry with DTE >= 3
        default_dexp = next((e for e in deltas_sorted if (oi_delta_dist[e].get('dte') or 0) >= 3), deltas_sorted[0])
        dopts = []
        for e in deltas_sorted:
            d = oi_delta_dist[e]
            dte_str = f" ({d['dte']}d)" if d.get('dte') is not None else ""
            sel = " selected" if e == default_dexp else ""
            dopts.append(f'<option value="{e}"{sel}>{e}{dte_str}</option>')
        oi_delta_json = json.dumps(oi_delta_dist)
        sample = oi_delta_dist[default_dexp]
        oi_delta_html = f"""
<div class="card" id="oidelta">
  <h2>OI Delta by Strike (yesterday → today, per expiry)</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:10px;line-height:1.6;">
    Net OI change at each strike between {sample['prev_date']} and {sample['date']}. <b style="color:#3fb950;">Green up</b> = call OI increased; direction is unconfirmed without trade prints. <b style="color:#3fb950;">Green down</b> = call OI decreased. Same logic for <b style="color:#f85149;">put</b> bars. <b>Roll detection:</b> simultaneous decrease at low strikes + increase at higher strikes = possible roll-up.
  </div>
  <div style="margin-bottom:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
    <label style="font-size:12px;color:#8b949e;">Expiry:</label>
    <select id="oiDeltaExpiry" style="background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:5px 10px;font-size:12px;font-family:inherit;">
      {''.join(dopts)}
    </select>
    <span id="oiDeltaMeta" style="font-size:11px;color:#8b949e;"></span>
  </div>
  <div class="chart-wrap-tall">
    <canvas id="oiDeltaChart"></canvas>
  </div>
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
            ev = mc + debt + pref - cash
            mnav = ev / btc_reserve

            def _verdict(x):
                if x < 1.2:
                    return ('#3fb950', 'Near NAV')
                if x < 1.4:
                    return ('#d29922', 'Yellow zone')
                if x < 1.8:
                    return ('#58a6ff', 'Neutral')
                return ('#f85149', 'Rich')
            color, label = _verdict(mnav)
            mnav_card_html = f"""
<div class="card" id="mnav">
  <h2>mNAV (MSTR Premium to BTC NAV)</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:10px;line-height:1.6;">
    EV-based premium MSTR trades at vs its BTC stash: <b style="color:#e6edf3;">(MarketCap + Debt + Pref − Cash) / BTC Reserve</b> (matches strategy.com's published mNAV). Near NAV &lt; 1.2x · Yellow zone 1.2–1.4x · Neutral 1.4–1.8x · Rich &gt; 1.8x. Historical range ~1.0–3.5x; troughs near 1.0 have marked accumulation zones, peaks &gt;2.5 marked distribution.
  </div>
  <div class="kpi-row" style="grid-template-columns:repeat(auto-fit, minmax(140px,1fr));">
    <div class="kpi">
      <div class="kpi-label">mNAV</div>
      <div class="kpi-value" style="color:{color};">{mnav:.2f}x</div>
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
      <div class="kpi-label">Enterprise Value</div>
      <div class="kpi-value">${ev/1e9:.1f}B</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Market Cap</div>
      <div class="kpi-value">${mc/1e9:.1f}B</div>
    </div>
  </div>
  <div style="font-size:11px;color:#6e7681;margin-top:6px;">Holdings as of {as_of} · BTC ${btc_price:,.0f} (from yfinance close) · source: strategy.com</div>
</div>"""

    # --- Daily Highlights card (5-second summary, day-over-day deltas) ---
    is_zh_hl = (lang == 'zh')
    spot_cur = current.get('spot') or 0
    iv_cur = current.get('atm_14d_iv') or current.get('front_atm_iv') or 0
    rv_cur = current.get('rv30') or 0
    nvrp_cur = current.get('nvrp') or 0
    pc_cur = current.get('front_pc_spread') or 0
    iv_pct_cur = current.get('iv_percentile') or 0
    iv_pct_n = current.get('iv_percentile_n') or 0

    # Day-over-day deltas (None when first snapshot)
    spot_chg_pct = None
    iv_chg = None
    nvrp_chg = None
    pc_chg = None
    iv_pct_chg = None
    pc_flipped = None  # 'to_safe' / 'to_danger' / None
    if prev_state and prev_state.get('spot'):
        spot_chg_pct = (spot_cur / prev_state['spot'] - 1) * 100
        iv_chg = iv_cur - (prev_state.get('atm_14d_iv') or prev_state.get('front_atm_iv') or 0)
        nvrp_chg = nvrp_cur - (prev_state.get('nvrp') or 0)
        pc_chg = pc_cur - (prev_state.get('front_pc_spread') or 0)
        iv_pct_chg = iv_pct_cur - (prev_state.get('iv_percentile') or 0)
        pc_prev = prev_state.get('front_pc_spread') or 0
        if pc_prev < 0 and pc_cur >= 0:
            pc_flipped = 'to_safe'
        elif pc_prev >= 0 and pc_cur < 0:
            pc_flipped = 'to_danger'

    # Top GEX walls today vs yesterday
    cur_call_wall = (gex_data or {}).get('call_walls', [{}])[0].get('strike') if (gex_data and gex_data.get('call_walls')) else None
    cur_put_wall = (gex_data or {}).get('put_walls', [{}])[0].get('strike') if (gex_data and gex_data.get('put_walls')) else None
    prev_call_wall = (prev_state or {}).get('top_call_wall')
    prev_put_wall = (prev_state or {}).get('top_put_wall')

    # Notable OI movers (biggest |call_delta| and |put_delta| in latest day's oi delta)
    oi_latest_for_hl = oi[-1] if oi else None
    top_call_oi_mover = None
    top_put_oi_mover = None
    if oi_latest_for_hl:
        # oi_latest is a dict with strikes-keyed call/put delta entries
        # Structure depends on oi_delta() implementation — let's be defensive
        rows = oi_latest_for_hl.get('strikes', []) if isinstance(oi_latest_for_hl, dict) else []
        if not rows and isinstance(oi_latest_for_hl, dict):
            # alternative shape: {strike: {call_delta, put_delta}}
            for k, v in oi_latest_for_hl.items():
                if isinstance(v, dict) and ('call_oi_delta' in v or 'put_oi_delta' in v):
                    rows.append({'strike': k, **v})

    # Fallback: pull from oi_delta_dist (today's date) — biggest abs change
    if oi_delta_dist:
        # oi_delta_dist is keyed by expiry, each with strikes/call_delta/put_delta arrays
        best_call = (0, None, None)
        best_put = (0, None, None)
        for exp, d in oi_delta_dist.items():
            strikes = d.get('strikes', [])
            cdeltas = d.get('call_delta', [])
            pdeltas = d.get('put_delta', [])
            for i, s in enumerate(strikes):
                if i < len(cdeltas) and abs(cdeltas[i]) > abs(best_call[0]):
                    best_call = (cdeltas[i], s, exp)
                if i < len(pdeltas) and abs(pdeltas[i]) > abs(best_put[0]):
                    best_put = (pdeltas[i], s, exp)
        if best_call[1] is not None and abs(best_call[0]) >= 1000:
            top_call_oi_mover = {'strike': best_call[1], 'delta': best_call[0], 'expiry': best_call[2]}
        if best_put[1] is not None and abs(best_put[0]) >= 1000:
            top_put_oi_mover = {'strike': best_put[1], 'delta': best_put[0], 'expiry': best_put[2]}

    # Format helpers
    def _delta_arrow(v, fmt='{:+.1f}', good_when_high=None, neutral_thresh=0.1, suffix=''):
        if v is None:
            return ''
        if abs(v) < neutral_thresh:
            return f'<span style="color:#8b949e;font-size:11px;">(持平)</span>' if is_zh_hl else f'<span style="color:#8b949e;font-size:11px;">(flat)</span>'
        color = '#8b949e'
        if good_when_high is not None:
            if (v > 0 and good_when_high) or (v < 0 and not good_when_high):
                color = '#3fb950'
            else:
                color = '#f85149'
        elif v > 0:
            color = '#3fb950'
        else:
            color = '#f85149'
        return f'<span style="color:{color};font-size:11px;">({fmt.format(v)}{suffix})</span>'

    btc_price_hl = (btc_data or {}).get('btc_price')
    btc_24h_hl = (btc_data or {}).get('btc_24h')

    # Build rows
    if is_zh_hl:
        L_hl = {
            'title': '今日速览',
            'lead': '只保留会改变计划的核心变化。',
            'price_label': '价格',
            'iv_label': 'IV / RV',
            'skew_label': '偏度',
            'gex_label': 'GEX 墙',
            'oi_label': 'OI 异动',
            'verdict_label': '裁定',
            'flat': '持平',
            'unchanged': '稳',
            'flipped_safe': '⚠ 翻正（call 偏度回归正常）',
            'flipped_danger': '⚠ 翻负（call 比 put 贵 — 挤压/追涨风险）',
            'no_prev': '（首次快照，无对比）',
            'call_built': 'Call OI 增加',
            'call_unwound': 'Call OI 减少',
            'put_built': 'Put OI 增加',
            'put_unwound': 'Put OI 减少',
            'no_oi_mover': '无显著异动',
        }
    else:
        L_hl = {
            'title': 'Signal Snapshot',
            'lead': 'Only the changes that can alter the plan.',
            'price_label': 'Price',
            'iv_label': 'IV / RV',
            'skew_label': 'Skew',
            'gex_label': 'GEX walls',
            'oi_label': 'OI movers',
            'verdict_label': 'Verdict',
            'flat': 'flat',
            'unchanged': 'unchanged',
            'flipped_safe': '⚠ flipped to positive (call-skew recovering)',
            'flipped_danger': '⚠ flipped negative (calls > puts — squeeze/chase risk)',
            'no_prev': '(first snapshot, no comparison)',
            'call_built': 'Call OI up',
            'call_unwound': 'Call OI down',
            'put_built': 'Put OI up',
            'put_unwound': 'Put OI down',
            'no_oi_mover': 'no significant movers',
        }

    # Price row
    price_html = f'Spot ${spot_cur:,.2f}'
    if spot_chg_pct is not None:
        sign = '+' if spot_chg_pct >= 0 else ''
        color = '#3fb950' if spot_chg_pct >= 0 else '#f85149'
        price_html += f' <span style="color:{color};font-size:11px;">({sign}{spot_chg_pct:.2f}%)</span>'
    if btc_price_hl:
        price_html += f' &nbsp;·&nbsp; BTC ${btc_price_hl/1000:.1f}K'
        if btc_24h_hl is not None:
            sign = '+' if btc_24h_hl >= 0 else ''
            color = '#3fb950' if btc_24h_hl >= 0 else '#f85149'
            price_html += f' <span style="color:{color};font-size:11px;">({sign}{btc_24h_hl:.2f}%)</span>'

    # IV/RV row
    if cur_iv_pct_reliable:
        iv_pct_inline = f'14D IV %ile {iv_pct_cur:.0f} <span style="color:#8b949e;font-size:11px;">(n={iv_pct_n})</span> {_delta_arrow(iv_pct_chg, "{:+.0f}")}'
    else:
        iv_pct_inline = (f'本地 IV 样本 <span style="color:#d29922;font-weight:600;">n={iv_pct_n} 不足</span>'
                         if is_zh_hl else
                         f'local IV sample <span style="color:#d29922;font-weight:600;">n={iv_pct_n} insufficient</span>')
    iv_html = f'14D IV {iv_cur:.1f}% {_delta_arrow(iv_chg, "{:+.1f}", suffix="pp")} &nbsp;·&nbsp; RV30 {rv_cur:.1f}% &nbsp;·&nbsp; NVRP {nvrp_cur:.2f}x {_delta_arrow(nvrp_chg, "{:+.2f}")} &nbsp;·&nbsp; {iv_pct_inline}'

    # Skew row
    pc_color = '#3fb950' if pc_cur >= 0 else '#f85149'
    skew_html = f'25Δ front <span style="color:{pc_color};font-weight:600;">{pc_cur:+.1f} pts</span> {_delta_arrow(pc_chg, "{:+.1f}", suffix="pp")}'
    if pc_flipped == 'to_safe':
        skew_html += f' <span style="color:#d29922;font-size:11px;font-weight:600;">{L_hl["flipped_safe"]}</span>'
    elif pc_flipped == 'to_danger':
        skew_html += f' <span style="color:#f85149;font-size:11px;font-weight:600;">{L_hl["flipped_danger"]}</span>'

    # GEX walls row
    def _wall_str(cur, prev, color):
        if cur is None:
            return f'<span style="color:#6e7681;">—</span>'
        s = f'<span style="color:{color};font-weight:600;">${cur:.0f}</span>'
        if prev is not None and prev != cur:
            move = cur - prev
            sign = '+' if move > 0 else ''
            s += f' <span style="color:#8b949e;font-size:11px;">(was ${prev:.0f}, {sign}{move:.0f})</span>'
        elif prev is not None and prev == cur:
            s += f' <span style="color:#8b949e;font-size:11px;">({L_hl["unchanged"]})</span>'
        return s
    gex_html = f'Call {_wall_str(cur_call_wall, prev_call_wall, "#3fb950")} &nbsp;·&nbsp; Put {_wall_str(cur_put_wall, prev_put_wall, "#f85149")}'

    # OI movers row
    def _expiry_tag(mover):
        exp = mover.get('expiry') if mover else None
        return f' <span style="color:#8b949e;font-size:11px;">{exp}</span>' if exp else ''

    oi_parts = []
    if top_call_oi_mover:
        d = top_call_oi_mover['delta']
        verb = L_hl['call_built'] if d > 0 else L_hl['call_unwound']
        color = '#3fb950' if d > 0 else '#f85149'
        oi_parts.append(f'<span style="color:{color};">{verb}</span> {_expiry_tag(top_call_oi_mover)} @ ${top_call_oi_mover["strike"]:.0f} <span style="color:#8b949e;font-size:11px;">({d:+,.0f})</span>')
    if top_put_oi_mover:
        d = top_put_oi_mover['delta']
        verb = L_hl['put_built'] if d > 0 else L_hl['put_unwound']
        color = '#f85149' if d > 0 else '#3fb950'
        oi_parts.append(f'<span style="color:{color};">{verb}</span> {_expiry_tag(top_put_oi_mover)} @ ${top_put_oi_mover["strike"]:.0f} <span style="color:#8b949e;font-size:11px;">({d:+,.0f})</span>')
    oi_html = ' &nbsp;·&nbsp; '.join(oi_parts) if oi_parts else f'<span style="color:#6e7681;">{L_hl["no_oi_mover"]}</span>'

    # Verdict row — placeholder; will be filled after verdict block computes
    # We'll insert a {VERDICT_PILL} placeholder and replace after verdict_label is known
    highlights_rows = [
        ('💰', L_hl['price_label'], price_html),
        ('📊', L_hl['iv_label'], iv_html),
        ('⚖️', L_hl['skew_label'], skew_html),
        ('🧱', L_hl['gex_label'], gex_html),
        ('📈', L_hl['oi_label'], oi_html),
    ]
    rows_html = []
    for icon, label, body in highlights_rows:
        rows_html.append(f'''<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:10px 12px;min-height:72px;">
      <div style="display:flex;align-items:center;gap:7px;margin-bottom:6px;">
        <span style="font-size:13px;">{icon}</span>
        <span style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;">{label}</span>
      </div>
      <div style="font-size:12px;color:#e6edf3;line-height:1.45;">{body}</div>
    </div>''')

    if not prev_state:
        no_prev_note = f'<div style="font-size:11px;color:#6e7681;font-style:italic;margin-top:8px;">{L_hl["no_prev"]}</div>'
    else:
        no_prev_note = f'<div style="font-size:10px;color:#6e7681;margin-top:8px;">vs {prev_state.get("date","")}</div>'

    # Highlights card built — verdict pill placeholder will be replaced below
    highlights_card_html = f"""
<div class="card" id="highlights">
  <h2 style="display:flex;align-items:center;gap:10px;">
    <span>{L_hl['title']}</span>
    {{VERDICT_PILL_PLACEHOLDER}}
  </h2>
  <div style="font-size:11px;color:#8b949e;margin-bottom:10px;line-height:1.5;">{L_hl['lead']}</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;">
    {''.join(rows_html)}
  </div>
  {no_prev_note}
</div>"""

    # --- Rulebook Verdict card (top-of-page decision layer) ---
    iv_pct = current.get('iv_percentile') or 0
    iv_pct_n = current.get('iv_percentile_n') or 0
    spot_now = current.get('spot') or 0

    # Recompute mNAV (mNAV card may not have run if holdings missing)
    mnav_val = None
    if ticker == 'MSTR' and holdings and btc_data:
        _shares = holdings.get('basic_shares_outstanding') or 0
        _btc_held = holdings.get('btc_holdings') or 0
        _btc_price = btc_data.get('btc_price') or 0
        _btc_reserve = _btc_held * _btc_price
        if _btc_reserve > 0:
            _mc = spot_now * _shares
            _ev = _mc + (holdings.get('debt') or 0) + (holdings.get('pref') or 0) - (holdings.get('cash') or 0)
            mnav_val = _ev / _btc_reserve

    btc_rv_7d = (btc_data or {}).get('btc_rv_7d') if btc_data else None

    # GEX call wall (for E6 TSLA — top call wall above spot)
    top_call_wall = None
    if gex_data and gex_data.get('call_walls'):
        top_call_wall = gex_data['call_walls'][0]['strike']

    # Language-aware rule labels
    is_zh = (lang == 'zh')
    L = {
        'e1_label': 'IV 底线（卖权门槛）' if is_zh else 'Premium-sell IV floor',
        'e1_iv_pct': '14D IV 分位' if is_zh else '14D IV %ile',
        'e1_local_sample': '本地 14D IV 样本' if is_zh else 'Local 14D IV sample',
        'e1_sample_na': 'n<60，分位不参与裁定' if is_zh else 'n<60; percentile excluded from verdict',
        'e1_sample_na_detail': '本地 IV 样本不足；不要把当前分位当长期 IV percentile，改看 14D IV、RV30、NVRP 与期限结构' if is_zh else 'Local IV sample is too small; do not treat the current percentile as long-run IV percentile. Use 14D IV, RV30, NVRP, and term structure instead',
        'e1_sample_nvrp_low': '样本不足且 NVRP<1；卖波动率没有边际，优先限亏/买波动率结构' if is_zh else 'Sample too small and NVRP<1; no short-vol edge, prefer limited-risk or long-vol structures',
        'e1_pass': 'IV ≥ 50 → 任何卖权结构均可' if is_zh else 'IV ≥ 50 → any premium-sell structure ok',
        'e1_warn': '30 ≤ IV &lt; 50 → 只做限亏结构（价差/IC），不开裸卖' if is_zh else '30 ≤ IV &lt; 50 → limited-risk only (spreads/IC), no naked sells',
        'e1_block': 'IV 太低 — 改用长期权 / 借方价差' if is_zh else 'Vol cheap — favor long premium / debit spreads',
        'e5_label': 'MSTR 短 Call 守护（mNAV）' if is_zh else 'MSTR short-call guardrail (mNAV)',
        'e5_pass': 'mNAV 偏贵；短 Call 可按其它信号评估' if is_zh else 'mNAV rich; short calls can be evaluated with other signals',
        'e5_warn_low': 'mNAV 黄区；只允许限亏/有保护的短 Call，不做裸卖或普通 CC' if is_zh else 'mNAV yellow zone; only limited-risk/protected short-call structures, no naked calls or plain CCs',
        'e5_warn_mid': 'mNAV 中性；短 Call 需 GEX/偏度配合，优先限亏结构' if is_zh else 'mNAV neutral; short calls need GEX/skew confirmation, prefer limited-risk structures',
        'e5_na': 'mNAV 数据缺失' if is_zh else 'mNAV unavailable',
        'e5_na_detail': '无法评估 — holdings 文件缺失' if is_zh else 'Cannot evaluate — holdings missing',
        'e5_block': 'mNAV 接近净值；不开任何 MSTR 短 Call（CC / 裸卖 / 熊 call 价差）' if is_zh else 'mNAV near NAV; no MSTR short calls in any form (CC, naked, bear-call)',
        'e6_label': 'TSLA 短 Call 纪律' if is_zh else 'TSLA short-call discipline',
        'e6_block': '低于 Call 墙 — 短 Call 被钉风险' if is_zh else 'Below the call wall — short calls trapped if pinned',
        'e6_pass': '高于 Call 墙；优先价差非裸卖' if is_zh else 'Above the wall; prefer spreads over naked',
        'e6_na': 'GEX 墙数据缺失' if is_zh else 'GEX wall unavailable',
        'e6_na_detail': '无法评估' if is_zh else 'Cannot evaluate',
        'v3_label': 'BTC 波动率冻结' if is_zh else 'BTC volatility freeze',
        'v3_pass': '低于冻结阈值' if is_zh else 'Below freeze threshold',
        'v3_block': '暂停 MSTR/IBIT 卖权' if is_zh else 'Pause MSTR/IBIT premium sells',
        'v3_na': 'RV7 数据缺失' if is_zh else 'RV7 unavailable',
        'v3_na_detail': '无法评估' if is_zh else 'Cannot evaluate',
        'spot': '现价' if is_zh else 'Spot',
        'call_wall': 'Call 墙',
        'mnav_unit': 'x',
    }

    # Rule rows: each is (code, label, status_text, status, detail)
    # status: 'pass' | 'warn' | 'block' | 'na'
    rules = []

    # E1 — premium-sell IV floor.
    # The local ATM-IV percentile is not decision-grade until at least 60 saved snapshots.
    if iv_pct_n < 60:
        status = 'warn' if cur_nvrp > 0 and cur_nvrp < 1.0 else 'na'
        detail = L['e1_sample_nvrp_low'] if status == 'warn' else L['e1_sample_na_detail']
        rules.append(('E1', L['e1_label'], f'{L["e1_local_sample"]} n={iv_pct_n} < 60', status, detail))
    elif iv_pct >= 50:
        rules.append(('E1', L['e1_label'], f'{L["e1_iv_pct"]} {iv_pct:.0f} (n={iv_pct_n})', 'pass', L['e1_pass']))
    elif iv_pct >= 30:
        rules.append(('E1', L['e1_label'], f'{L["e1_iv_pct"]} {iv_pct:.0f} (30-50, n={iv_pct_n})', 'warn', L['e1_warn']))
    else:
        rules.append(('E1', L['e1_label'], f'{L["e1_iv_pct"]} {iv_pct:.0f} < 30 (n={iv_pct_n})', 'block', L['e1_block']))

    if ticker == 'MSTR':
        # E5 — MSTR short-call mNAV gate (banded; decoupled from E1 — IV is E1's job)
        if mnav_val is None:
            rules.append(('E5', L['e5_label'], L['e5_na'], 'na', L['e5_na_detail']))
        elif mnav_val < 1.2:
            rules.append(('E5', L['e5_label'], f'mNAV {mnav_val:.2f}x < 1.2', 'block', L['e5_block']))
        elif mnav_val < 1.4:
            rules.append(('E5', L['e5_label'], f'mNAV {mnav_val:.2f}x (1.2-1.4)', 'warn', L['e5_warn_low']))
        elif mnav_val < 1.8:
            rules.append(('E5', L['e5_label'], f'mNAV {mnav_val:.2f}x (1.4-1.8)', 'warn', L['e5_warn_mid']))
        else:
            rules.append(('E5', L['e5_label'], f'mNAV {mnav_val:.2f}x ≥ 1.8', 'pass', L['e5_pass']))

        # V3 — BTC vol freeze (RV7 > 80% blocks)
        if btc_rv_7d is None:
            rules.append(('V3', L['v3_label'], L['v3_na'], 'na', L['v3_na_detail']))
        elif btc_rv_7d > 80:
            rules.append(('V3', L['v3_label'], f'BTC RV7 {btc_rv_7d:.0f}% > 80%', 'block', L['v3_block']))
        else:
            rules.append(('V3', L['v3_label'], f'BTC RV7 {btc_rv_7d:.0f}%', 'pass', L['v3_pass']))

    elif ticker == 'TSLA':
        # E6 — TSLA short-call discipline (above call wall, prefer spreads)
        if top_call_wall is not None and spot_now > 0:
            if spot_now < top_call_wall:
                rules.append(('E6', L['e6_label'], f'{L["spot"]} ${spot_now:.0f} < {L["call_wall"]} ${top_call_wall:.0f}', 'block', L['e6_block']))
            else:
                rules.append(('E6', L['e6_label'], f'{L["spot"]} ${spot_now:.0f} > {L["call_wall"]} ${top_call_wall:.0f}', 'pass', L['e6_pass']))
        else:
            rules.append(('E6', L['e6_label'], L['e6_na'], 'na', L['e6_na_detail']))

    # Aggregate verdict — most-restrictive wins (block > warn > pass)
    blocked = [r for r in rules if r[3] == 'block']
    warned = [r for r in rules if r[3] == 'warn']
    if blocked:
        verdict_color = '#f85149'
        verdict_detail = (f'{len(blocked)} 条规则阻塞' if is_zh
                          else f'{len(blocked)} rule(s) blocking')
        # Detail what's blocked vs what's still allowed
        blocked_codes = ', '.join(r[0] for r in blocked)
        if ticker == 'MSTR' and any(r[0] == 'E5' for r in blocked):
            verdict_label = '禁短 CALL' if is_zh else 'NO SHORT CALLS'
            verdict_action = (f'MSTR 短 Call 阻塞（{blocked_codes}）；其它 ticker 的限亏价差仍可视 E1 而定'
                              if is_zh
                              else f'MSTR short calls blocked ({blocked_codes}); limited-risk structures on other tickers still depend on E1')
        else:
            verdict_label = '今日不开仓' if is_zh else 'STAND ASIDE'
            verdict_action = ('改用借方价差 / 长期权 / 等 IV 回升' if is_zh
                              else 'Use defined-risk / long premium / wait for IV to recover')
    elif warned:
        verdict_color = '#d29922'
        if ticker == 'MSTR' and any(r[0] == 'E5' for r in warned):
            verdict_label = '短 CALL 限亏' if is_zh else 'SHORT CALL LIMITED-RISK'
        else:
            verdict_label = '只做限亏结构' if is_zh else 'LIMITED-RISK ONLY'
        verdict_detail = (f'{len(warned)} 条规则警告 — 不开裸卖' if is_zh
                          else f'{len(warned)} rule(s) warn — no naked sells')
        verdict_action = ('用价差 / Iron Condor / 借方结构；裸 CSP/CC 暂停' if is_zh
                          else 'Use spreads / Iron Condors / debit structures; pause naked CSPs/CCs')
    else:
        verdict_color = '#3fb950'
        verdict_label = '闸门通过' if is_zh else 'GATES OPEN'
        verdict_detail = '所有适用规则通过' if is_zh else 'All applicable rules pass'
        verdict_action = '进入仓位 sizing (S1–S6) 与开仓' if is_zh else 'Proceed to sizing (S1–S6) and entry'

    # Build rule rows HTML
    icons = {'pass': '✓', 'warn': '⚠', 'block': '✗', 'na': '·'}
    icon_colors = {'pass': '#3fb950', 'warn': '#d29922', 'block': '#f85149', 'na': '#8b949e'}
    row_html = []
    chip_html = []
    for code, label, status_text, status, detail in rules:
        ic = icons[status]
        ic_color = icon_colors[status]
        chip_html.append(
            f'''<div style="display:flex;align-items:center;gap:7px;background:#0d1117;border:1px solid {ic_color}44;border-radius:8px;padding:8px 10px;">
      <span style="color:{ic_color};font-weight:700;font-size:14px;width:14px;text-align:center;">{ic}</span>
      <span style="color:#e6edf3;font-weight:600;font-size:12px;">{code}</span>
      <span style="color:#8b949e;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{status_text}</span>
    </div>'''
        )
        row_html.append(f'''<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.05);">
      <div style="font-size:18px;font-weight:600;color:{ic_color};width:20px;text-align:center;">{ic}</div>
      <div style="font-size:13px;font-weight:600;color:#e6edf3;width:36px;font-variant-numeric:tabular-nums;">{code}</div>
      <div style="flex:1;">
        <div style="font-size:13px;color:#e6edf3;">{label}</div>
        <div style="font-size:11px;color:#8b949e;margin-top:2px;">{status_text} · {detail}</div>
      </div>
    </div>''')
    rules_table_html = '\n    '.join(row_html)
    rules_chip_html = '\n    '.join(chip_html)

    pos_html = ''

    # Card title + lead text
    if is_zh:
        verdict_title = '规则裁定'
        verdict_lead = '只显示闸门结果；细则默认折叠。'
        verdict_details_label = '展开规则细节'
    else:
        verdict_title = 'Rulebook Verdict'
        verdict_lead = 'Gate result only; rule details are collapsed by default.'
        verdict_details_label = 'Show rule details'

    # Inject verdict pill into highlights card
    verdict_pill_html = f'<a href="#verdict" style="text-decoration:none;background:{verdict_color}22;color:{verdict_color};border:1px solid {verdict_color};padding:3px 10px;border-radius:10px;font-size:11px;font-weight:600;letter-spacing:0.3px;">{verdict_label}</a>'
    highlights_card_html = highlights_card_html.replace('{VERDICT_PILL_PLACEHOLDER}', verdict_pill_html)

    rulebook_card_html = f"""
<div class="card" id="verdict" style="border:1px solid {verdict_color}40;">
  <h2 style="display:flex;align-items:center;gap:10px;">
    <span>{verdict_title}</span>
    <span style="background:{verdict_color}22;color:{verdict_color};border:1px solid {verdict_color};padding:3px 10px;border-radius:10px;font-size:11px;font-weight:600;letter-spacing:0.3px;">{verdict_label}</span>
  </h2>
  <div style="font-size:11px;color:#8b949e;margin-bottom:10px;line-height:1.5;">{verdict_lead}</div>
  <div style="padding:10px 12px;background:{verdict_color}10;border-left:3px solid {verdict_color};border-radius:6px;font-size:12px;color:#e6edf3;">
    <span style="color:{verdict_color};font-weight:600;">{verdict_detail}.</span> {verdict_action}.
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin-top:10px;">
    {rules_chip_html}
  </div>
  <details style="margin-top:10px;font-size:11px;color:#8b949e;">
    <summary style="cursor:pointer;color:#58a6ff;">{verdict_details_label}</summary>
    <div style="margin-top:8px;background:#0d1117;border:1px solid rgba(255,255,255,0.05);border-radius:10px;overflow:hidden;">
      {rules_table_html}
    </div>
  </details>
  {pos_html}
</div>"""

    # --- Final Playbook card: reconcile conflicting signals into one action layer ---
    e5_blocked = ticker == 'MSTR' and any(r[0] == 'E5' and r[3] == 'block' for r in rules)
    e5_warned = ticker == 'MSTR' and any(r[0] == 'E5' and r[3] == 'warn' for r in rules)
    e1_warn_or_block = any(r[0] == 'E1' and r[3] in ('warn', 'block') for r in rules)

    def _level(v):
        return f'${v:.0f}' if v is not None else 'n/a'

    def _persistent_level(w):
        if not w:
            return None
        if w.get('is_band'):
            return f'${w["lo"]:.0f}-${w["hi"]:.0f}'
        return f'${w["strike"]:.0f}'

    p_call_level = _persistent_level(call_wall)
    p_put_level = _persistent_level(put_wall)

    if is_zh:
        if long_vol_edge and short_gamma_regime:
            pb_bias = '偏向波动率扩张'
            pb_bias_detail = f'NVRP {cur_nvrp:.2f}x < 1.0 且净 GEX 为负；不要把区间墙解读成卖波动率信号。'
            pb_color = '#f85149'
        elif e5_blocked:
            pb_bias = '中性偏谨慎'
            pb_bias_detail = f'mNAV {mnav_val:.2f}x 接近净值；MSTR 上行 squeeze 风险不值得卖。' if mnav_val else 'MSTR 短 Call 规则阻塞。'
            pb_color = '#d29922'
        elif e5_warned:
            pb_bias = 'mNAV 黄区'
            pb_bias_detail = f'mNAV {mnav_val:.2f}x 不是绝对便宜也不算贵；短 Call 只能做限亏/有保护结构，并需 GEX/偏度配合。'
            pb_color = '#d29922'
        elif cur_pc := current.get('front_pc_spread'):
            pb_bias = 'Call 偏度升温' if cur_pc < 0 else '结构中性'
            pb_bias_detail = 'Call IV 高于 Put IV，是追涨/挤压需求信号；只作为风险提示，不单独判顶。' if cur_pc < 0 else '偏度未显示明显追涨压力。'
            pb_color = '#d29922' if cur_pc < 0 else '#58a6ff'
        else:
            pb_bias, pb_bias_detail, pb_color = '结构中性', '暂无单一主导信号。', '#58a6ff'

        allowed = []
        blocked_items = []
        if long_vol_edge:
            allowed.append('借方价差 / 长波动率结构 / 小仓位方向期权')
        if e1_warn_or_block:
            allowed.append('只做限亏结构（价差、IC、买权结构）')
        if not allowed:
            allowed.append('通过规则后按仓位系统开仓')
        if e5_blocked:
            blocked_items.append('MSTR covered call / 裸 short call / 熊 call 价差')
        elif e5_warned:
            blocked_items.append('MSTR 裸 short call / 普通 covered call；短 Call 只做限亏或有保护结构')
        if short_vol_suppressed:
            blocked_items.append('裸卖波动率；卖勒式/铁鹰只可观察，不作为主计划')
        if not blocked_items:
            blocked_items.append('无')

        key_rows = [
            ('现价', f'${spot_now:.2f}', '基准价格'),
            ('GEX Call 墙', _level(cur_call_wall), '上方阻力/突破后挤压触发点'),
            ('GEX Put 墙', _level(cur_put_wall), '下方支撑/跌破后放大风险'),
            ('持续 Call 墙', p_call_level or 'n/a', '结构性成交阻力'),
            ('持续 Put 墙', p_put_level or 'n/a', '结构性成交支撑'),
        ]
        pb_title, bias_lbl, allow_lbl, block_lbl, levels_lbl = '最终交易计划', '方向/环境', '允许', '禁止', '关键价位'
    else:
        if long_vol_edge and short_gamma_regime:
            pb_bias = 'Vol expansion bias'
            pb_bias_detail = f'NVRP {cur_nvrp:.2f}x < 1.0 and net GEX is negative; do not read range walls as a short-vol signal.'
            pb_color = '#f85149'
        elif e5_blocked:
            pb_bias = 'Neutral/cautious'
            pb_bias_detail = f'mNAV {mnav_val:.2f}x is near NAV; MSTR upside squeeze risk is not worth selling.' if mnav_val else 'MSTR short-call rule is blocking.'
            pb_color = '#d29922'
        elif e5_warned:
            pb_bias = 'mNAV yellow zone'
            pb_bias_detail = f'mNAV {mnav_val:.2f}x is neither cheap enough to block all trades nor rich enough for plain short calls; use protected/limited-risk short-call structures only with GEX/skew confirmation.'
            pb_color = '#d29922'
        else:
            cur_pc = current.get('front_pc_spread') or 0
            pb_bias = 'Call skew heating up' if cur_pc < 0 else 'Structurally neutral'
            pb_bias_detail = 'Call IV is above put IV: chase/squeeze demand risk. Treat it as a warning, not a standalone top call.' if cur_pc < 0 else 'Skew does not show major upside-chase pressure.'
            pb_color = '#d29922' if cur_pc < 0 else '#58a6ff'

        allowed = []
        blocked_items = []
        if long_vol_edge:
            allowed.append('debit spreads / long-vol structures / small directional options')
        if e1_warn_or_block:
            allowed.append('limited-risk only: spreads, ICs, long-premium structures')
        if not allowed:
            allowed.append('rulebook-approved entries subject to sizing')
        if e5_blocked:
            blocked_items.append('MSTR covered calls / naked short calls / bear-call spreads')
        elif e5_warned:
            blocked_items.append('MSTR naked short calls / plain covered calls; short calls only as protected or limited-risk structures')
        if short_vol_suppressed:
            blocked_items.append('naked short vol; strangles/iron condors are watchlist only, not the main plan')
        if not blocked_items:
            blocked_items.append('None')

        key_rows = [
            ('Spot', f'${spot_now:.2f}', 'baseline'),
            ('GEX call wall', _level(cur_call_wall), 'resistance / squeeze trigger if broken'),
            ('GEX put wall', _level(cur_put_wall), 'support / downside amplification if broken'),
            ('Persistent call wall', p_call_level or 'n/a', 'structural volume resistance'),
            ('Persistent put wall', p_put_level or 'n/a', 'structural volume support'),
        ]
        pb_title, bias_lbl, allow_lbl, block_lbl, levels_lbl = 'Final Playbook', 'Bias / Regime', 'Allowed', 'Avoid', 'Key Levels'

    key_rows_html = ''.join(
        f"<tr><td style='text-align:left;color:#8b949e;'>{name}</td><td style='text-align:left;color:#e6edf3;font-weight:600;'>{level}</td><td style='text-align:left;color:#8b949e;'>{note}</td></tr>"
        for name, level, note in key_rows
    )
    allowed_html = ''.join(f"<li>{item}</li>" for item in allowed)
    blocked_html = ''.join(f"<li>{item}</li>" for item in blocked_items)

    playbook_card_html = f"""
<div class="card" id="playbook" style="border:1px solid {pb_color}40;">
  <h2>{pb_title}</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;align-items:stretch;">
    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:12px;">
      <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">{bias_lbl}</div>
      <div style="font-size:18px;color:{pb_color};font-weight:600;margin-bottom:6px;">{pb_bias}</div>
      <div style="font-size:12px;color:#c9d1d9;line-height:1.5;">{pb_bias_detail}</div>
    </div>
    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:12px;font-size:12px;line-height:1.6;">
      <div style="font-size:11px;color:#3fb950;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">{allow_lbl}</div>
      <ul style="margin:0;padding-left:18px;color:#e6edf3;">{allowed_html}</ul>
    </div>
    <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.05);border-radius:10px;padding:12px;font-size:12px;line-height:1.6;">
      <div style="font-size:11px;color:#f85149;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">{block_lbl}</div>
      <ul style="margin:0;padding-left:18px;color:#e6edf3;">{blocked_html}</ul>
    </div>
  </div>
  <div style="margin-top:12px;">
    <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">{levels_lbl}</div>
    <table>
      {key_rows_html}
    </table>
  </div>
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
    <a href="#highlights">{nav_lbl_highlights}</a>
    <a href="#playbook">{nav_lbl_playbook}</a>
    <a href="#kpis">{nav_lbl_kpi}</a>
    <a href="#verdict">{nav_lbl_verdict}</a>
    <a href="#btc">BTC</a>
    {('<a href="#mnav">mNAV</a>' if ticker == 'MSTR' else '')}
    <a href="#spread">{nav_lbl_skew}</a>
    <a href="#skew">SKEW</a>
    <a href="#iv">IV</a>
    <a href="#term">{nav_lbl_term}</a>
    <a href="#gex">GEX</a>
    <a href="#oidist">OI Dist</a>
    <a href="#oidelta">OI Δ</a>
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
    <div class="kpi-label">14D 50Δ IV</div>
    <div class="kpi-value">{current.get('atm_14d_iv', current.get('front_atm_iv', 0)):.1f}%</div>
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
    <div class="kpi-label">{'本地 IV 分位' if is_zh else 'Local IV %ile'}</div>
    <div class="kpi-value" style="color:{pct_color};font-size:{'22px' if cur_iv_pct_reliable else '18px'};">{f'{cur_iv_pct:.0f}' if cur_iv_pct_reliable else ('n<60' if not is_zh else '样本不足')}</div>
    <div style="font-size:10px;color:#6e7681;margin-top:2px;">n={cur_iv_pct_n}{'' if cur_iv_pct_reliable else (' · 不用于裁定' if is_zh else ' · not used')}</div>
  </div>
</div>

{mode_banner_html}

{playbook_card_html}

{highlights_card_html}

{rulebook_card_html}

{btc_card_html}

{mnav_card_html}

<!-- P/C Spread Time Series (full width) -->
<div class="card" id="spread">
  <h2>{'Delta-25 Put/Call IV 偏度：怎么用' if is_zh else 'Delta-25 Put/Call IV Skew: How to Use It'}</h2>
  <div style="font-size:11px;color:#8b949e;margin-bottom:6px;line-height:1.6;">
    <b style="color:#e6edf3;">Spread = Put IV − Call IV</b> at delta-25 strikes. Positive = puts more expensive (normal hedging demand). Negative = calls more expensive (upside chase/squeeze demand).
  </div>
  <div style="font-size:11px;margin-bottom:6px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;">
    <span style="color:#a8e6b8;"><span style="display:inline-block;width:14px;height:10px;background:#3fb95022;border:1px solid #3fb95055;vertical-align:middle;margin-right:4px;"></span><b>Safe zone</b> (spread &gt; 0) — puts cost more than calls, market pricing in downside risk = healthy</span>
    <span style="color:#f85149;"><span style="display:inline-block;width:14px;height:10px;background:#f8514922;border:1px solid #f8514955;vertical-align:middle;margin-right:4px;"></span><b>Danger zone</b> (spread &lt; 0) — calls cost more than puts, upside chase/squeeze demand elevated</span>
    <span><span style="display:inline-block;width:16px;border-top:1.5px dashed #e6edf3;vertical-align:middle;margin-right:4px;"></span>Zero line (threshold)</span>
    <span><span style="display:inline-block;width:16px;border-top:1.5px dashed #8b949e;vertical-align:middle;margin-right:4px;"></span>Spot price (right axis)</span>
  </div>
  {skew_decision_html}
  {skew_pct_html}
  <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px;" id="spreadLegend"></div>
  <div class="chart-wrap-tall"><canvas id="spreadChart"></canvas></div>
</div>

{skew_card_html}

<!-- IV + NVRP Trend -->
<div class="card" id="iv">
  <h2>{iv_dashboard_title}</h2>
  {iv_dashboard_html}
  <div style="font-size:11px;color:#8b949e;margin-bottom:6px;line-height:1.6;">
    <b style="color:#e6edf3;">NVRP = 14D IV / RV30.</b> The 1.3 threshold exists because sellers need ~30% cushion to cover bid-ask spread, gamma risk, and hedging error. &lt;1.0 = long-vol edge; 1.0-1.3 = marginal; &gt;1.3 = short-vol edge; &gt;1.5 = strong edge.
  </div>
  {nvrp_insight}
  <div class="chart-wrap"><canvas id="ivChart"></canvas></div>
  {term_compact_html}
</div>

{gex_card_html}

{oi_dist_html}

{oi_delta_html}

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
        <b style="color:#3fb950;">OTM Call (above spot)</b> = call OI increased above spot. Direction is unconfirmed without trade prints; read alongside price and volume.
        <br><b style="color:#d29922;">ITM Call (below spot)</b> = usually <b>covered-call writers</b> locking gains on existing stock. Increases OI but is <b>not bullish</b> — it's profit-taking.
        <br><b style="color:#f85149;">OTM Put (below spot)</b> = downside bets or portfolio hedges.
        <br><b style="color:#f85149;">Deep OTM Put (&lt;-15% spot)</b> = systematic tail insurance. Only bought by institutions/macro funds. When this surges, <b>smart money is nervous</b>.
        <br><br>
        <b>Classic patterns:</b>
        <br>• <b>Bullish candidate:</b> OTM call OI expands while puts stay quiet, especially if price/volume confirm
        <br>• <b>Bearish/hedge candidate:</b> OTM put OI expands while calls stay quiet, especially if price/volume confirm
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
        <br><b>Trade rules:</b> Use walls as levels first. Short-vol structures require Playbook and Rulebook approval. Watch wall migration (up = bullish re-rating, down = top risk, thinning = resistance fading).
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

  // Historical 14D IV + underlying close chart
  const ivDates = {json.dumps(iv_dates)};
  new Chart(document.getElementById('ivChart'), {{
    type: 'line',
    data: {{
      labels: ivDates,
      datasets: [
        {{ label: '14D 50Δ IV', data: {json.dumps(iv_values)}, borderColor: '#f2cc60', borderWidth: 2.5, pointRadius: 2, tension: 0.15, yAxisID: 'y' }},
        {{ label: 'RV30', data: {json.dumps(rv_values)}, borderColor: '#58a6ff', borderWidth: 1.5, pointRadius: 2, tension: 0.15, borderDash: [4,4], yAxisID: 'y' }},
        {{ label: 'Underlying Close', data: {json.dumps(spot_values)}, borderColor: '#8b949e', borderWidth: 1.5, pointRadius: 0, tension: 0.15, borderDash: [6,3], yAxisID: 'yPrice' }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      layout: {{ padding: {{ right: 44 }} }},
      plugins: {{ legend: {{ position: 'top', labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ display: false }} }},
        y: {{ position: 'left', ticks: {{ color: '#8b949e', callback: v => v.toFixed(0) + '%' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, title: {{ display: true, text: '14D IV / RV30 (%)', color: '#8b949e' }} }},
        yPrice: {{ position: 'right', ticks: {{ color: '#8b949e', callback: v => '$' + v }}, grid: {{ display: false }}, title: {{ display: true, text: 'Underlying Close ($)', color: '#8b949e' }} }}
      }}
    }},
    plugins: [{{
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
          const fmt = ds.label === 'Underlying Close' ? '$' + val.toFixed(0) : val.toFixed(1) + '%';
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

  // --- OI Distribution histogram ---
  const oiAll = {oi_dist_json};
  const oiSel = document.getElementById('oiDistExpiry');
  const oiCanvas = document.getElementById('oiDistChart');
  if (oiSel && oiCanvas && Object.keys(oiAll).length) {{
    let oiChart;
    function pixelAtStrike(scale, val, strikes) {{
      let lo = -1, hi = -1;
      for (let i = 0; i < strikes.length; i++) {{
        if (strikes[i] <= val) lo = i;
        if (strikes[i] >= val && hi < 0) {{ hi = i; }}
      }}
      if (lo < 0) return scale.getPixelForValue(strikes[0].toString());
      if (hi < 0 || hi === lo) return scale.getPixelForValue(strikes[lo].toString());
      const px_lo = scale.getPixelForValue(strikes[lo].toString());
      const px_hi = scale.getPixelForValue(strikes[hi].toString());
      const t = (val - strikes[lo]) / (strikes[hi] - strikes[lo]);
      return px_lo + t * (px_hi - px_lo);
    }}
    function renderOi() {{
      const exp = oiSel.value;
      const d = oiAll[exp];
      if (!d) return;
      const labels = d.strikes.map(s => s.toString());
      const meta = document.getElementById('oiDistMeta');
      const totalCall = d.call_oi.reduce((a, b) => a + b, 0);
      const totalPut = d.put_oi.reduce((a, b) => a + b, 0);
      const pcr = totalCall ? (totalPut / totalCall) : 0;
      meta.textContent = `Max Pain $${{d.max_pain.toFixed(2)}} · Spot $${{d.spot.toFixed(2)}} · Calls ${{totalCall.toLocaleString()}} · Puts ${{totalPut.toLocaleString()}} · PCR ${{pcr.toFixed(2)}}`;
      if (oiChart) {{ oiChart.destroy(); }}
      oiChart = new Chart(oiCanvas, {{
        type: 'bar',
        data: {{
          labels: labels,
          datasets: [
            {{ label: 'Calls', data: d.call_oi, backgroundColor: '#3fb950cc', borderColor: '#3fb950', borderWidth: 0, barPercentage: 0.95, categoryPercentage: 0.85 }},
            {{ label: 'Puts',  data: d.put_oi,  backgroundColor: '#f85149cc', borderColor: '#f85149', borderWidth: 0, barPercentage: 0.95, categoryPercentage: 0.85 }},
          ],
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          layout: {{ padding: {{ top: 24 }} }},
          plugins: {{
            legend: {{ display: true, labels: {{ color: '#e6edf3', font: {{ size: 11 }}, boxWidth: 12 }} }},
            tooltip: {{
              backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
              titleColor: '#e6edf3', bodyColor: '#e6edf3', padding: 10,
              callbacks: {{
                title: items => '$' + items[0].label,
                label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toLocaleString(),
              }},
            }},
          }},
          scales: {{
            x: {{
              stacked: false,
              ticks: {{ color: '#8b949e', font: {{ size: 10 }}, maxRotation: 0, autoSkip: true, maxTicksLimit: 18, callback: function(v) {{ return '$' + this.getLabelForValue(v); }} }},
              grid: {{ display: false }},
            }},
            y: {{
              ticks: {{ color: '#8b949e', font: {{ size: 11 }}, callback: v => v >= 1000 ? (v/1000).toFixed(0) + 'k' : v }},
              grid: {{ color: 'rgba(255,255,255,0.04)' }},
              title: {{ display: true, text: 'Open Interest', color: '#8b949e', font: {{ size: 11 }} }},
            }},
          }},
        }},
        plugins: [{{
          id: 'oi-markers',
          afterDatasetsDraw(chart) {{
            const ctx = chart.ctx;
            const xs = chart.scales.x;
            const area = chart.chartArea;
            ctx.save();
            const markers = [
              {{ value: d.spot,     color: '#22d3ee', label: 'Spot $' + d.spot.toFixed(2) }},
              {{ value: d.max_pain, color: '#facc15', label: 'Max Pain $' + d.max_pain.toFixed(2) }},
            ];
            for (const m of markers) {{
              const px = pixelAtStrike(xs, m.value, d.strikes);
              if (px < area.left || px > area.right) continue;
              ctx.strokeStyle = m.color;
              ctx.lineWidth = 1.5;
              ctx.setLineDash([4, 4]);
              ctx.beginPath();
              ctx.moveTo(px, area.top);
              ctx.lineTo(px, area.bottom);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.fillStyle = m.color;
              ctx.font = '600 10px -apple-system,sans-serif';
              ctx.textBaseline = 'top';
              const w = ctx.measureText(m.label).width;
              const tx = Math.max(area.left + 4, Math.min(area.right - w - 4, px + 4));
              ctx.fillText(m.label, tx, area.top - 18);
            }}
            ctx.restore();
          }},
        }}],
      }});
    }}
    oiSel.addEventListener('change', renderOi);
    renderOi();
  }}

  // --- OI Delta histogram (yesterday → today) ---
  const oiDeltaAll = {oi_delta_json};
  const odSel = document.getElementById('oiDeltaExpiry');
  const odCanvas = document.getElementById('oiDeltaChart');
  if (odSel && odCanvas && Object.keys(oiDeltaAll).length) {{
    let odChart;
    function odPixelAtStrike(scale, val, strikes) {{
      let lo = -1, hi = -1;
      for (let i = 0; i < strikes.length; i++) {{
        if (strikes[i] <= val) lo = i;
        if (strikes[i] >= val && hi < 0) {{ hi = i; }}
      }}
      if (lo < 0) return scale.getPixelForValue(strikes[0].toString());
      if (hi < 0 || hi === lo) return scale.getPixelForValue(strikes[lo].toString());
      const px_lo = scale.getPixelForValue(strikes[lo].toString());
      const px_hi = scale.getPixelForValue(strikes[hi].toString());
      const t = (val - strikes[lo]) / (strikes[hi] - strikes[lo]);
      return px_lo + t * (px_hi - px_lo);
    }}
    function renderOd() {{
      const exp = odSel.value;
      const d = oiDeltaAll[exp];
      if (!d) return;
      const labels = d.strikes.map(s => s.toString());
      const meta = document.getElementById('oiDeltaMeta');
      const sumCall = d.call_delta.reduce((a, b) => a + b, 0);
      const sumPut = d.put_delta.reduce((a, b) => a + b, 0);
      meta.textContent = `${{d.prev_date}} → ${{d.date}} · Net Call Δ ${{sumCall >= 0 ? '+' : ''}}${{sumCall.toLocaleString()}} · Net Put Δ ${{sumPut >= 0 ? '+' : ''}}${{sumPut.toLocaleString()}} · Spot $${{d.spot.toFixed(2)}}`;
      if (odChart) {{ odChart.destroy(); }}
      odChart = new Chart(odCanvas, {{
        type: 'bar',
        data: {{
          labels: labels,
          datasets: [
            {{ label: 'Call Δ', data: d.call_delta, backgroundColor: ctx => (ctx.raw >= 0 ? '#3fb950cc' : '#3fb95055'), borderColor: '#3fb950', borderWidth: 0, barPercentage: 0.95, categoryPercentage: 0.85 }},
            {{ label: 'Put Δ',  data: d.put_delta,  backgroundColor: ctx => (ctx.raw >= 0 ? '#f85149cc' : '#f8514955'), borderColor: '#f85149', borderWidth: 0, barPercentage: 0.95, categoryPercentage: 0.85 }},
          ],
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          plugins: {{
            legend: {{ display: true, labels: {{ color: '#e6edf3', font: {{ size: 11 }}, boxWidth: 12 }} }},
            tooltip: {{
              backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
              titleColor: '#e6edf3', bodyColor: '#e6edf3', padding: 10,
              callbacks: {{
                title: items => '$' + items[0].label,
                label: ctx => ctx.dataset.label + ': ' + (ctx.parsed.y >= 0 ? '+' : '') + ctx.parsed.y.toLocaleString(),
              }},
            }},
          }},
          scales: {{
            x: {{
              ticks: {{ color: '#8b949e', font: {{ size: 10 }}, maxRotation: 0, autoSkip: true, maxTicksLimit: 18, callback: function(v) {{ return '$' + this.getLabelForValue(v); }} }},
              grid: {{ display: false }},
            }},
            y: {{
              ticks: {{ color: '#8b949e', font: {{ size: 11 }}, callback: v => (v >= 0 ? '+' : '') + (Math.abs(v) >= 1000 ? (v/1000).toFixed(0) + 'k' : v) }},
              grid: {{ color: ctx => ctx.tick.value === 0 ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.04)' }},
              title: {{ display: true, text: 'OI Δ (built ↑ / unwound ↓)', color: '#8b949e', font: {{ size: 11 }} }},
            }},
          }},
        }},
        plugins: [{{
          id: 'od-spot',
          afterDatasetsDraw(chart) {{
            const ctx = chart.ctx;
            const xs = chart.scales.x;
            const area = chart.chartArea;
            ctx.save();
            const markers = [{{ value: d.spot, color: '#22d3ee', label: 'Spot $' + d.spot.toFixed(2) }}];
            for (const m of markers) {{
              const px = odPixelAtStrike(xs, m.value, d.strikes);
              if (px < area.left || px > area.right) continue;
              ctx.strokeStyle = m.color;
              ctx.lineWidth = 1.5;
              ctx.setLineDash([4, 4]);
              ctx.beginPath();
              ctx.moveTo(px, area.top);
              ctx.lineTo(px, area.bottom);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.fillStyle = m.color;
              ctx.font = '600 10px -apple-system,sans-serif';
              ctx.textBaseline = 'top';
              const w = ctx.measureText(m.label).width;
              const tx = Math.max(area.left + 4, Math.min(area.right - w - 4, px + 4));
              ctx.fillText(m.label, tx, area.top + 4);
            }}
            ctx.restore();
          }},
        }}],
      }});
    }}
    odSel.addEventListener('change', renderOd);
    renderOd();
  }}

  // --- Macro SKEW (^SKEW 2y) line chart ---
  const skewData = {skew_idx_json};
  const skewCanvas = document.getElementById('skewChart');
  if (skewData && skewCanvas) {{
    const skewCurrent = skewData.values[skewData.values.length - 1];
    let skewLineColor = '#3fb950';
    if (skewCurrent >= 160) skewLineColor = '#f85149';
    else if (skewCurrent >= 140) skewLineColor = '#f0883e';
    else if (skewCurrent >= 120) skewLineColor = '#d29922';
    new Chart(skewCanvas, {{
      type: 'line',
      data: {{
        labels: skewData.dates,
        datasets: [{{
          label: '^SKEW',
          data: skewData.values,
          borderColor: skewLineColor,
          backgroundColor: skewLineColor + '20',
          borderWidth: 1.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
          tension: 0.15,
        }}],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
            titleColor: '#e6edf3', bodyColor: '#e6edf3', padding: 10,
            callbacks: {{ label: ctx => '^SKEW: ' + ctx.parsed.y.toFixed(2) }},
          }},
        }},
        scales: {{
          x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }}, maxRotation: 0, autoSkip: true, maxTicksLimit: 10 }}, grid: {{ display: false }} }},
          y: {{
            ticks: {{ color: '#8b949e', font: {{ size: 11 }} }},
            grid: {{ color: 'rgba(255,255,255,0.05)' }},
          }},
        }},
      }},
      plugins: [{{
        id: 'skew-bands',
        beforeDatasetsDraw(chart) {{
          const ys = chart.scales.y;
          const area = chart.chartArea;
          const ctx = chart.ctx;
          const bands = [
            {{ from: ys.min, to: 120, color: 'rgba(63,185,80,0.06)' }},
            {{ from: 120, to: 140, color: 'rgba(210,153,34,0.06)' }},
            {{ from: 140, to: 160, color: 'rgba(240,136,62,0.08)' }},
            {{ from: 160, to: ys.max, color: 'rgba(248,81,73,0.10)' }},
          ];
          ctx.save();
          for (const b of bands) {{
            const y1 = ys.getPixelForValue(Math.max(b.from, ys.min));
            const y2 = ys.getPixelForValue(Math.min(b.to, ys.max));
            const top = Math.min(y1, y2), bot = Math.max(y1, y2);
            if (bot < area.top || top > area.bottom) continue;
            ctx.fillStyle = b.color;
            ctx.fillRect(area.left, Math.max(top, area.top), area.right - area.left, Math.min(bot, area.bottom) - Math.max(top, area.top));
          }}
          // Threshold lines at 120, 140, 160
          ctx.strokeStyle = 'rgba(255,255,255,0.10)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          for (const t of [120, 140, 160]) {{
            if (t < ys.min || t > ys.max) continue;
            const y = ys.getPixelForValue(t);
            ctx.beginPath();
            ctx.moveTo(area.left, y);
            ctx.lineTo(area.right, y);
            ctx.stroke();
            ctx.fillStyle = '#6e7681';
            ctx.font = '500 10px -apple-system,sans-serif';
            ctx.fillText(t.toString(), area.left + 4, y - 3);
          }}
          ctx.setLineDash([]);
          ctx.restore();
        }},
      }}],
    }});
  }}
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
    history_snapshots = load_snapshots(ticker, None)
    iv_pct_by_date = iv_percentile_history(history_snapshots)

    # Build all analyses
    pc_series = pc_spread_timeseries(snapshots)
    crossings = detect_zero_crossings(pc_series)
    oi_deltas = oi_delta(snapshots)
    persist = volume_persistence(snapshots)
    em = em_accuracy(snapshots)
    iv = iv_trend(snapshots, iv_pct_by_date)
    iv_history = iv_trend(history_snapshots, iv_pct_by_date)
    skew_pct = skew_percentile_analysis(pc_series, 'front')
    call_dte = dte_call_build_ratio(snapshots)
    gex_data = gex_walls(snapshots[-1]) if snapshots else None
    oi_dist = oi_distribution(snapshots[-1]) if snapshots else {}
    oi_delta_dist = oi_delta_distribution(snapshots)
    skew_idx = skew_index_history()
    btc_data = btc_context(ticker, snapshots)

    # Current state summary
    latest = snapshots[-1]
    front = front_term(latest)
    latest_cm14 = constant_maturity_iv(latest, 14)
    latest_iv14 = latest_cm14.get('iv', 0) or front.get('atm_iv', 0)
    latest_iv_meta = iv_pct_by_date.get(latest.get('date'), {})
    current = {
        'spot': latest['spot'],
        'rv30': latest.get('rv30', 0),
        'rv30_pct_2yr': latest.get('rv30_pct_2yr', latest.get('iv_pct_2yr', 0)),
        'iv_percentile': latest_iv_meta.get('iv_percentile', 0),
        'iv_percentile_n': latest_iv_meta.get('iv_percentile_n', 0),
        'iv_percentile_reliable': latest_iv_meta.get('iv_percentile_reliable', False),
        'iv_percentile_min': latest_iv_meta.get('iv_percentile_min', 0),
        'iv_percentile_max': latest_iv_meta.get('iv_percentile_max', 0),
        'iv_percentile_median': latest_iv_meta.get('iv_percentile_median', 0),
        'front_atm_iv': front.get('atm_iv', 0),
        'atm_14d_iv': latest_iv14,
        'atm_14d_method': latest_cm14.get('method', ''),
        'atm_14d_lower_dte': latest_cm14.get('lower_dte'),
        'atm_14d_upper_dte': latest_cm14.get('upper_dte'),
        'atm_14d_nearest_dte': latest_cm14.get('nearest_dte'),
        'front_pc_spread': front.get('pc_iv_spread', 0),
        'nvrp': round(latest_iv14 / latest['rv30'], 2) if latest.get('rv30', 0) > 0 else 0,
        'skew_level': skew_alert_level(front.get('pc_iv_spread', 0)),
    }

    # Previous-day state for day-over-day deltas in the Highlights card
    prev_state = None
    if len(snapshots) >= 2:
        prev = snapshots[-2]
        prev_front = front_term(prev)
        prev_cm14 = constant_maturity_iv(prev, 14)
        prev_iv14 = prev_cm14.get('iv', 0) or prev_front.get('atm_iv', 0)
        prev_gex = gex_walls(prev) or {}
        prev_iv_meta = iv_pct_by_date.get(prev.get('date'), {})
        prev_state = {
            'date': prev.get('date', ''),
            'spot': prev.get('spot', 0),
            'rv30': prev.get('rv30', 0),
            'rv30_pct_2yr': prev.get('rv30_pct_2yr', prev.get('iv_pct_2yr', 0)),
            'iv_percentile': prev_iv_meta.get('iv_percentile', 0),
            'iv_percentile_n': prev_iv_meta.get('iv_percentile_n', 0),
            'iv_percentile_reliable': prev_iv_meta.get('iv_percentile_reliable', False),
            'iv_percentile_min': prev_iv_meta.get('iv_percentile_min', 0),
            'iv_percentile_max': prev_iv_meta.get('iv_percentile_max', 0),
            'iv_percentile_median': prev_iv_meta.get('iv_percentile_median', 0),
            'front_atm_iv': prev_front.get('atm_iv', 0),
            'atm_14d_iv': prev_iv14,
            'atm_14d_method': prev_cm14.get('method', ''),
            'atm_14d_lower_dte': prev_cm14.get('lower_dte'),
            'atm_14d_upper_dte': prev_cm14.get('upper_dte'),
            'atm_14d_nearest_dte': prev_cm14.get('nearest_dte'),
            'front_pc_spread': prev_front.get('pc_iv_spread', 0),
            'nvrp': round(prev_iv14 / prev.get('rv30', 0), 2) if prev.get('rv30', 0) > 0 else 0,
            'top_call_wall': (prev_gex.get('call_walls') or [{}])[0].get('strike') if prev_gex.get('call_walls') else None,
            'top_put_wall': (prev_gex.get('put_walls') or [{}])[0].get('strike') if prev_gex.get('put_walls') else None,
        }

    # Check all expirations for negative spread
    negative_spreads = []
    for t in latest.get('term', []):
        if t.get('pc_iv_spread', 999) < 0:
            negative_spreads.append(f"{t['expiry']} (DTE {t['dte']}): {t['pc_iv_spread']:+.1f} pts")

    trend_data = {
        'ticker': ticker,
        'date': latest.get('date') or datetime.now().strftime('%Y-%m-%d'),
        'num_snapshots': len(snapshots),
        'date_range': [snapshots[0]['date'], snapshots[-1]['date']],
        'current_state': current,
        'prev_state': prev_state,
        'pc_spread_series': pc_series,
        'zero_crossings': crossings,
        'oi_deltas': oi_deltas,
        'volume_persistence': persist,
        'em_accuracy': em,
        'iv_trend': iv,
        'iv_history': iv_history,
        'negative_spreads': negative_spreads,
        'skew_percentile': skew_pct,
        'call_dte_ratio': call_dte,
        'gex_walls': gex_data,
        'oi_distribution': oi_dist,
        'oi_delta_distribution': oi_delta_dist,
        'skew_index': skew_idx,
        'btc_context': btc_data,
    }

    # Save trend JSON
    trend_path = os.path.join(SNAPSHOT_DIR, f"trend-{ticker}.json")
    with open(trend_path, 'w') as f:
        json.dump(trend_data, f, default=str, indent=2)
    print(f"  Trend saved to {trend_path}")

    # Console summary
    print(f"\n  === {ticker} TREND SUMMARY ===")
    print(f"  Spot: ${current['spot']:.2f}  |  14D IV: {current.get('atm_14d_iv', current['front_atm_iv']):.1f}%  |  RV30: {current['rv30']:.1f}%  |  NVRP: {current['nvrp']:.2f}x")
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
        report_date = trend_data.get('date') or datetime.now().strftime('%Y-%m-%d')
        html_path = os.path.join(REPORT_DIR, f"trend_{ticker}_{report_date}{suffix}.html")
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
