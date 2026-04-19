#!/usr/bin/env python3
"""
Daily options flow snapshot.

Usage:
    python3 snapshot.py MSTR TSLA [MORE_TICKERS...]
    # no args = uses watchlist (MSTR, TSLA)

Saves: ~/options_portfolio/flow_snapshots/YYYY-MM-DD-TICKER.json
"""
import os, sys, json, warnings, subprocess
from datetime import datetime
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance", "numpy", "pandas", "scipy"])

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings('ignore')

SNAPSHOT_DIR = os.environ.get('FLOW_SNAPSHOT_DIR') or os.path.expanduser("~/options_portfolio/flow_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def bs(S, K, T, r, sig, opt):
    if T <= 0 or sig <= 0:
        return max(0, (S-K) if opt == 'call' else (K-S))
    d1 = (np.log(S/K) + (r + 0.5*sig*sig)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    if opt == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def iv_inv(price, S, K, T, opt, r=0.045):
    if T <= 0 or price <= 0:
        return 0
    intrinsic = max(0, (S-K) if opt == 'call' else (K-S))
    if price < intrinsic * 0.95:
        return 0
    lo, hi = 0.01, 5.0
    for _ in range(80):
        mid = (lo+hi)/2
        if bs(S, K, T, r, mid, opt) < price: lo = mid
        else: hi = mid
    return (lo+hi)/2


def bs_delta(S, K, T, r, sig, opt):
    """Black-Scholes delta for finding delta-25 strikes."""
    if T <= 0 or sig <= 0:
        if opt == 'call':
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sig*sig)*T) / (sig*np.sqrt(T))
    if opt == 'call':
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def bs_gamma(S, K, T, r, sig):
    if T <= 0 or sig <= 0: return 0
    d1 = (np.log(S/K) + (r + 0.5*sig*sig)*T) / (sig*np.sqrt(T))
    return norm.pdf(d1) / (S * sig * np.sqrt(T))


def market_context():
    """Fetch VIX, VIX3M, SPY, BTC for macro backdrop."""
    ctx = {}
    for sym, key in [("^VIX", "vix"), ("^VIX3M", "vix3m"), ("SPY", "spy"), ("BTC-USD", "btc")]:
        try:
            h = yf.Ticker(sym).history(period='5d')
            if len(h) > 0:
                ctx[key] = float(h['Close'].iloc[-1])
                if len(h) >= 2:
                    ctx[f"{key}_prev"] = float(h['Close'].iloc[-2])
        except:
            pass
    if 'vix' in ctx and 'vix3m' in ctx and ctx['vix3m'] > 0:
        ctx['vix_term'] = 'contango' if ctx['vix3m'] > ctx['vix'] else 'backwardation'
        ctx['vix_ratio'] = round(ctx['vix'] / ctx['vix3m'], 3)
    return ctx


def snapshot(sym):
    t = yf.Ticker(sym)
    try: spot = float(t.info.get('currentPrice') or t.history(period='1d')['Close'].iloc[-1])
    except: spot = float(t.history(period='1d')['Close'].iloc[-1])

    exps = list(t.options)[:8]
    term = []
    chains = {}
    expected_moves = {}

    for exp in exps:
        try:
            ch = t.option_chain(exp)
            c, p = ch.calls.copy(), ch.puts.copy()
            dte = max((pd.to_datetime(exp) - pd.Timestamp.now()).days, 0)
            T = max(dte, 1) / 365.0

            # Back-compute IV from lastPrice
            c['iv'] = c.apply(lambda r: iv_inv(r['lastPrice'], spot, r['strike'], T, 'call'), axis=1)
            p['iv'] = p.apply(lambda r: iv_inv(r['lastPrice'], spot, r['strike'], T, 'put'), axis=1)

            # ATM band IV
            atm_band_c = c[(c['strike'] >= spot*0.98) & (c['strike'] <= spot*1.02) & (c['iv'] > 0.05) & (c['iv'] < 3)]
            atm_band_p = p[(p['strike'] >= spot*0.98) & (p['strike'] <= spot*1.02) & (p['iv'] > 0.05) & (p['iv'] < 3)]
            atm_iv = float(np.nanmean(list(atm_band_c['iv']) + list(atm_band_p['iv']))) if (len(atm_band_c) or len(atm_band_p)) else 0

            # OTM skew (put IV - call IV at ~5% OTM)
            otm_c = c[(c['strike'] >= spot*1.04) & (c['strike'] <= spot*1.08) & (c['iv'] > 0.05)]
            otm_p = p[(p['strike'] >= spot*0.92) & (p['strike'] <= spot*0.96) & (p['iv'] > 0.05)]
            skew = (np.nanmean(otm_p['iv']) - np.nanmean(otm_c['iv'])) if (len(otm_c) and len(otm_p)) else 0
            if np.isnan(skew): skew = 0

            # --- Delta-25 Put/Call IV spread ---
            use_iv = atm_iv if atm_iv > 0.05 else 0.5
            c['delta'] = c.apply(lambda r: bs_delta(spot, r['strike'], T, 0.045, r['iv'] if r['iv'] > 0.05 else use_iv, 'call'), axis=1)
            p['delta'] = p.apply(lambda r: bs_delta(spot, r['strike'], T, 0.045, r['iv'] if r['iv'] > 0.05 else use_iv, 'put'), axis=1)

            valid_c = c[(c['iv'] > 0.05) & (c['iv'] < 3) & (c['delta'] > 0.05) & (c['delta'] < 0.75)]
            valid_p = p[(p['iv'] > 0.05) & (p['iv'] < 3) & (p['delta'] < -0.05) & (p['delta'] > -0.75)]

            d25_call_iv = d25_put_iv = 0.0
            d25_call_strike = d25_put_strike = 0.0
            pc_iv_spread = 0.0

            if len(valid_c) > 0:
                row = valid_c.iloc[(valid_c['delta'] - 0.25).abs().argsort()[:1]]
                d25_call_iv = float(row['iv'].iloc[0])
                d25_call_strike = float(row['strike'].iloc[0])
            if len(valid_p) > 0:
                row = valid_p.iloc[(valid_p['delta'] + 0.25).abs().argsort()[:1]]
                d25_put_iv = float(row['iv'].iloc[0])
                d25_put_strike = float(row['strike'].iloc[0])

            if d25_call_iv > 0 and d25_put_iv > 0:
                pc_iv_spread = (d25_put_iv - d25_call_iv) * 100  # vol points

            # ATM straddle for Expected Move
            atm_c_row = c.iloc[(c['strike'] - spot).abs().argsort()[:1]]
            atm_p_row = p.iloc[(p['strike'] - spot).abs().argsort()[:1]]
            atm_call_price = float(atm_c_row['lastPrice'].iloc[0])
            atm_put_price = float(atm_p_row['lastPrice'].iloc[0])
            expected_move = 0.85 * (atm_call_price + atm_put_price)
            em_pct = expected_move / spot * 100

            expected_moves[exp] = {
                'expected_move_dollars': expected_move,
                'expected_move_pct': em_pct,
                'upper_1sigma': spot + expected_move,
                'lower_1sigma': spot - expected_move,
            }

            # Volume / OI
            c['volume'] = c['volume'].fillna(0)
            p['volume'] = p['volume'].fillna(0)
            c['openInterest'] = c['openInterest'].fillna(0)
            p['openInterest'] = p['openInterest'].fillna(0)

            # GEX (use volume as proxy when OI=0)
            c['gamma'] = c.apply(lambda r: bs_gamma(spot, r['strike'], T, 0.045, r['iv'] if r['iv'] > 0.05 else atm_iv), axis=1)
            p['gamma'] = p.apply(lambda r: bs_gamma(spot, r['strike'], T, 0.045, r['iv'] if r['iv'] > 0.05 else atm_iv), axis=1)
            c['gex'] = c['gamma'] * c['openInterest'] * 100 * spot * spot
            p['gex'] = -p['gamma'] * p['openInterest'] * 100 * spot * spot
            if c['openInterest'].sum() == 0 and p['openInterest'].sum() == 0:
                c['gex'] = c['gamma'] * c['volume'] * 100 * spot * spot
                p['gex'] = -p['gamma'] * p['volume'] * 100 * spot * spot

            term.append({
                'expiry': exp, 'dte': dte,
                'atm_iv': atm_iv * 100,
                'skew': skew * 100,
                # Delta-25 spread fields
                'd25_call_iv': d25_call_iv * 100,
                'd25_put_iv': d25_put_iv * 100,
                'd25_call_strike': d25_call_strike,
                'd25_put_strike': d25_put_strike,
                'pc_iv_spread': pc_iv_spread,
                # Volume / OI
                'call_vol': int(c['volume'].sum()),
                'put_vol': int(p['volume'].sum()),
                'call_oi': int(c['openInterest'].sum()),
                'put_oi': int(p['openInterest'].sum()),
                'pc_vol': float(p['volume'].sum() / max(c['volume'].sum(), 1)),
                # Expected Move
                'expected_move_pct': em_pct,
                'expected_move_dollars': expected_move,
                'em_upper': spot + expected_move,
                'em_lower': spot - expected_move,
            })

            chains[exp] = {
                'calls': c[['strike', 'lastPrice', 'volume', 'openInterest', 'iv', 'delta', 'gex']].to_dict('records'),
                'puts': p[['strike', 'lastPrice', 'volume', 'openInterest', 'iv', 'delta', 'gex']].to_dict('records'),
            }
        except Exception as e:
            print(f"  skip {exp}: {e}")

    # RV30 over 2 years
    h = t.history(period='2y')
    r = h['Close'].pct_change().dropna()
    rv30 = (r.rolling(30).std() * np.sqrt(252) * 100).dropna()
    cur_rv = float(rv30.iloc[-1])
    iv_pct_2yr = float((rv30 < cur_rv).mean() * 100)

    # Market context
    ctx = market_context()

    return {
        'ticker': sym,
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'spot': spot,
        'rv30': cur_rv,
        'iv_pct_2yr': iv_pct_2yr,
        'term': term,
        'chains': chains,
        'expected_moves': expected_moves,
        'market_context': ctx,
    }


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ['MSTR', 'TSLA']
    date = datetime.now().strftime('%Y-%m-%d')

    for sym in tickers:
        print(f"snapshot {sym}...")
        try:
            data = snapshot(sym)
            path = os.path.join(SNAPSHOT_DIR, f"{date}-{sym}.json")
            with open(path, 'w') as f:
                json.dump(data, f, default=str)
            print(f"  -> {path}")
            print(f"  spot={data['spot']:.2f}  RV30={data['rv30']:.1f}%  IV%ile={data['iv_pct_2yr']:.0f}")
            if data['term']:
                em = data['term'][0]
                print(f"  EM (front): +/-{em['expected_move_pct']:.1f}%  [{em['em_lower']:.2f}, {em['em_upper']:.2f}]")
                print(f"  D25 P/C spread (front): {em['pc_iv_spread']:+.1f} pts  (call={em['d25_call_iv']:.1f}% put={em['d25_put_iv']:.1f}%)")
                # Flag zero-crossing
                if em['pc_iv_spread'] < 0:
                    print(f"  ** CALL SKEW ALERT: P/C spread negative — bullish positioning dominant **")
            ctx = data.get('market_context', {})
            if 'vix' in ctx:
                vix3m_str = f"  VIX3M={ctx['vix3m']:.1f}" if 'vix3m' in ctx else ''
                term_str = f"  term={ctx['vix_term']}" if 'vix_term' in ctx else ''
                print(f"  VIX={ctx['vix']:.1f}{vix3m_str}{term_str}")
        except Exception as e:
            print(f"  ERR {sym}: {e}")

    print(f"\nAll snapshots saved to {SNAPSHOT_DIR}")


if __name__ == '__main__':
    main()
