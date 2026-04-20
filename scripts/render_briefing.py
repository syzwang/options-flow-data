#!/usr/bin/env python3
"""Render a weekly briefing from JSON → HTML styled to match dashboards.

Usage:
  python3 scripts/render_briefing.py path/to/briefing.json [output.html]

JSON schema (see scripts/briefing_schema.md for docs):
{
  "date": "2026-04-19",
  "week_ending": "2026-04-24",
  "headline": "One-line overview of the week",
  "tickers": [
    {
      "symbol": "MSTR",
      "spot": 320.5,
      "setup": {
        "gamma_regime": "short" | "long" | "neutral",
        "gamma_note": "string",
        "skew_posture": "call-dominant" | "put-dominant" | "balanced",
        "skew_percentile": 72,
        "nvrp": 1.15,
        "nvrp_signal": "seller edge" | "neutral" | "buyer edge",
        "gex_upper": 340.0,
        "gex_lower": 300.0,
        "gex_magnet": 325.0
      },
      "scenarios": {
        "bull":  {"prob": 30, "path": "...", "triggers": ["...", "..."]},
        "base":  {"prob": 50, "path": "...", "triggers": ["..."]},
        "bear":  {"prob": 20, "path": "...", "triggers": ["..."]}
      },
      "trades": [
        {"structure": "CSP", "strike": 290, "expiry": "2026-05-16",
         "dte": 27, "credit": "$6.50", "rationale": "...", "exit": "..."}
      ],
      "risks": ["FOMC minutes Wed", "CPI Thu"],
      "exits": ["Skew flips", "VIX > 20"]
    }
  ]
}
"""
import json, sys, os, html
from datetime import datetime

def esc(s):
    return html.escape(str(s)) if s is not None else ""

def render_setup(s):
    return f"""
    <div class="kpis">
      <div class="kpi"><div class="k-label">Gamma</div><div class="k-val">{esc(s.get('gamma_regime','—')).title()}</div><div class="k-note">{esc(s.get('gamma_note',''))}</div></div>
      <div class="kpi"><div class="k-label">Skew</div><div class="k-val">{esc(s.get('skew_posture','—')).title()}</div><div class="k-note">{esc(s.get('skew_percentile','—'))}th pctl</div></div>
      <div class="kpi"><div class="k-label">NVRP</div><div class="k-val">{esc(s.get('nvrp','—'))}</div><div class="k-note">{esc(s.get('nvrp_signal',''))}</div></div>
      <div class="kpi"><div class="k-label">GEX Walls</div><div class="k-val">${esc(s.get('gex_lower','—'))} – ${esc(s.get('gex_upper','—'))}</div><div class="k-note">magnet ${esc(s.get('gex_magnet','—'))}</div></div>
    </div>
    """

def render_scenarios(scn):
    colors = {'bull': '#3fb950', 'base': '#58a6ff', 'bear': '#f85149'}
    out = ['<div class="scenarios">']
    for key in ('bull', 'base', 'bear'):
        s = scn.get(key, {})
        if not s:
            continue
        triggers = ''.join(f'<li>{esc(t)}</li>' for t in s.get('triggers', []))
        out.append(f"""
          <div class="scn" style="border-left:3px solid {colors[key]};">
            <div class="scn-head"><span class="scn-name" style="color:{colors[key]}">{key.upper()}</span><span class="scn-prob">{esc(s.get('prob','—'))}%</span></div>
            <div class="scn-path">{esc(s.get('path',''))}</div>
            <ul class="scn-triggers">{triggers}</ul>
          </div>
        """)
    out.append('</div>')
    return ''.join(out)

def render_trades(trades):
    if not trades:
        return '<div class="empty">No trades this week.</div>'
    rows = []
    for t in trades:
        rows.append(f"""
          <tr>
            <td><strong>{esc(t.get('structure','—'))}</strong></td>
            <td>${esc(t.get('strike','—'))}</td>
            <td>{esc(t.get('expiry','—'))}<br><span class="muted">{esc(t.get('dte','—'))}d</span></td>
            <td>{esc(t.get('credit','—'))}</td>
            <td class="rationale">{esc(t.get('rationale',''))}</td>
            <td class="exit">{esc(t.get('exit',''))}</td>
          </tr>
        """)
    return f"""
    <table class="trades">
      <thead><tr><th>Structure</th><th>Strike</th><th>Expiry</th><th>Credit/Debit</th><th>Rationale</th><th>Exit</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """

def render_list(items, cls=""):
    if not items:
        return ""
    lis = ''.join(f'<li>{esc(x)}</li>' for x in items)
    return f'<ul class="{cls}">{lis}</ul>'

def render_ticker(t):
    return f"""
    <section class="ticker">
      <h2>{esc(t['symbol'])} <span class="spot">${esc(t.get('spot','—'))}</span></h2>
      <h3>Setup</h3>
      {render_setup(t.get('setup', {}))}
      <h3>Scenarios</h3>
      {render_scenarios(t.get('scenarios', {}))}
      <h3>Trade Ideas</h3>
      {render_trades(t.get('trades', []))}
      <div class="two-col">
        <div><h3>Risk Events</h3>{render_list(t.get('risks', []), 'risks')}</div>
        <div><h3>Exit Triggers</h3>{render_list(t.get('exits', []), 'exits')}</div>
      </div>
    </section>
    """

CSS = """
* { box-sizing: border-box; }
body { background:#0f1117; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,sans-serif; max-width:900px; margin:40px auto; padding:0 20px; line-height:1.5; }
h1 { font-size:26px; font-weight:500; margin:0 0 4px; }
h2 { font-size:22px; font-weight:500; margin:40px 0 8px; padding-bottom:8px; border-bottom:1px solid #30363d; color:#58a6ff; }
h3 { font-size:14px; font-weight:500; text-transform:uppercase; letter-spacing:0.5px; color:#8b949e; margin:24px 0 10px; }
.subtitle { color:#8b949e; font-size:13px; margin-bottom:8px; }
.headline { color:#e6edf3; font-size:15px; margin:12px 0 28px; padding:12px 16px; background:#161b22; border-left:3px solid #58a6ff; border-radius:0 4px 4px 0; }
.spot { color:#8b949e; font-size:16px; font-weight:400; margin-left:8px; }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
.kpi { background:#161b22; border:1px solid #21262d; border-radius:6px; padding:12px 14px; }
.k-label { font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; }
.k-val { font-size:16px; font-weight:500; margin:4px 0; }
.k-note { font-size:11px; color:#8b949e; }
.scenarios { display:grid; gap:10px; }
.scn { background:#161b22; padding:12px 16px; border-radius:4px; }
.scn-head { display:flex; justify-content:space-between; margin-bottom:6px; }
.scn-name { font-weight:600; font-size:13px; letter-spacing:0.5px; }
.scn-prob { font-size:13px; color:#8b949e; }
.scn-path { font-size:14px; margin-bottom:8px; }
.scn-triggers { margin:0; padding-left:18px; font-size:13px; color:#8b949e; }
.scn-triggers li { margin:2px 0; }
table.trades { width:100%; border-collapse:collapse; font-size:13px; margin-top:6px; }
table.trades th { text-align:left; padding:8px 10px; color:#8b949e; font-weight:500; border-bottom:1px solid #30363d; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; }
table.trades td { padding:10px; border-bottom:1px solid #21262d; vertical-align:top; }
table.trades .rationale, table.trades .exit { color:#c9d1d9; font-size:12px; max-width:220px; }
table.trades .muted { color:#8b949e; font-size:11px; }
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
ul.risks, ul.exits { margin:0; padding-left:18px; font-size:13px; }
ul.risks li, ul.exits li { margin:4px 0; }
.footer { margin-top:60px; padding-top:16px; border-top:1px solid #30363d; font-size:11px; color:#8b949e; }
.footer a { color:#58a6ff; text-decoration:none; }
.empty { color:#8b949e; font-style:italic; font-size:13px; }
@media (max-width:600px) { .kpis { grid-template-columns:repeat(2,1fr); } .two-col { grid-template-columns:1fr; } }
"""

def main():
    if len(sys.argv) < 2:
        print("usage: render_briefing.py <input.json> [output.html]", file=sys.stderr)
        sys.exit(1)
    inp = sys.argv[1]
    with open(inp) as f:
        data = json.load(f)

    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.environ.get('FLOW_REPORT_DIR', 'reports'),
        f'briefing_{date}.html'
    )

    tickers_html = ''.join(render_ticker(t) for t in data.get('tickers', []))
    headline = data.get('headline', '')
    headline_html = f'<div class="headline">{esc(headline)}</div>' if headline else ''
    week_ending = data.get('week_ending', '')
    week_label = f" — week ending {esc(week_ending)}" if week_ending else ""

    html_out = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Options Briefing {esc(date)}</title>
<style>{CSS}</style>
</head><body>
<h1>Weekly Options Briefing</h1>
<div class="subtitle">{esc(date)}{week_label} · MSTR · TSLA</div>
{headline_html}
{tickers_html}
<div class="footer">
Auto-generated from <code>snapshots/trend-*.json</code> ·
<a href="index.html">All dashboards →</a>
</div>
</body></html>"""

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(html_out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
