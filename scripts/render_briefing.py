#!/usr/bin/env python3
"""Render a weekly briefing from JSON → HTML styled to match dashboards.

Usage:
  python3 scripts/render_briefing.py <input.json> [output.html] [--lang en|zh]

Text fields (headline, gamma_note, scenarios.*.path, scenarios.*.triggers,
trades.*.rationale, trades.*.exit, risks, exits) can be either a plain string
(used for all langs) or a bilingual object {"en": "...", "zh": "..."}.
"""
import argparse, json, sys, os, html
from datetime import datetime

# Fixed-chrome translations (headers, labels, etc.)
CHROME = {
    'en': {
        'title': 'Weekly Options Briefing',
        'week_ending': 'week ending',
        'setup': 'Setup',
        'scenarios': 'Scenarios',
        'trades': 'Trade Ideas',
        'risks': 'Risk Events',
        'exits': 'Exit Triggers',
        'gamma': 'Gamma',
        'skew': 'Skew',
        'nvrp': 'NVRP',
        'gex_walls': 'GEX Walls',
        'pctl': 'th pctl',
        'magnet': 'magnet',
        'no_trades': 'No trades this week.',
        'th_structure': 'Structure',
        'th_strike': 'Strike',
        'th_expiry': 'Expiry',
        'th_credit': 'Credit/Debit',
        'th_rationale': 'Rationale',
        'th_exit': 'Exit',
        'footer_prefix': 'Auto-generated from',
        'footer_link': 'All dashboards →',
    },
    'zh': {
        'title': '每周期权简报',
        'week_ending': '截至',
        'setup': '市场定位',
        'scenarios': '情景推演',
        'trades': '交易想法',
        'risks': '风险事件',
        'exits': '退出信号',
        'gamma': 'Gamma 状态',
        'skew': '偏度',
        'nvrp': 'NVRP',
        'gex_walls': 'GEX 价位',
        'pctl': ' 百分位',
        'magnet': '磁吸位',
        'no_trades': '本周无交易建议。',
        'th_structure': '结构',
        'th_strike': '行权价',
        'th_expiry': '到期',
        'th_credit': '权利金',
        'th_rationale': '理由',
        'th_exit': '退出',
        'footer_prefix': '自动生成 ·',
        'footer_link': '所有 dashboard →',
    },
}

# Enum value translations
ENUM = {
    'en': {
        'short': 'Short', 'long': 'Long', 'neutral': 'Neutral',
        'call-dominant': 'Call-dominant', 'put-dominant': 'Put-dominant', 'balanced': 'Balanced',
        'seller edge': 'seller edge', 'neutral': 'neutral', 'buyer edge': 'buyer edge',
    },
    'zh': {
        'short': '空 (放大波动)', 'long': '多 (抑制波动)', 'neutral': '中性',
        'call-dominant': '看涨主导', 'put-dominant': '看跌主导', 'balanced': '均衡',
        'seller edge': '卖方占优', 'neutral': '中性', 'buyer edge': '买方占优',
    },
}

def esc(s):
    return html.escape(str(s)) if s is not None else ""

def txt(value, lang):
    """Pick lang from bilingual dict, fall back to 'en' or str."""
    if isinstance(value, dict):
        return value.get(lang) or value.get('en') or ""
    return value if value is not None else ""

def enum(key, lang):
    return ENUM.get(lang, ENUM['en']).get(key, key) if key else "—"

def render_setup(s, lang, C):
    gamma = enum(s.get('gamma_regime'), lang)
    skew = enum(s.get('skew_posture'), lang)
    nvrp_sig = enum(s.get('nvrp_signal'), lang)
    return f"""
    <div class="kpis">
      <div class="kpi"><div class="k-label">{C['gamma']}</div><div class="k-val">{esc(gamma)}</div><div class="k-note">{esc(txt(s.get('gamma_note'), lang))}</div></div>
      <div class="kpi"><div class="k-label">{C['skew']}</div><div class="k-val">{esc(skew)}</div><div class="k-note">{esc(s.get('skew_percentile','—'))}{C['pctl']}</div></div>
      <div class="kpi"><div class="k-label">{C['nvrp']}</div><div class="k-val">{esc(s.get('nvrp','—'))}</div><div class="k-note">{esc(nvrp_sig)}</div></div>
      <div class="kpi"><div class="k-label">{C['gex_walls']}</div><div class="k-val">${esc(s.get('gex_lower','—'))} – ${esc(s.get('gex_upper','—'))}</div><div class="k-note">{C['magnet']} ${esc(s.get('gex_magnet','—'))}</div></div>
    </div>
    """

def render_scenarios(scn, lang):
    colors = {'bull': '#3fb950', 'base': '#58a6ff', 'bear': '#f85149'}
    labels = {'bull': {'en': 'BULL', 'zh': '看涨'}, 'base': {'en': 'BASE', 'zh': '基准'}, 'bear': {'en': 'BEAR', 'zh': '看跌'}}
    out = ['<div class="scenarios">']
    for key in ('bull', 'base', 'bear'):
        s = scn.get(key, {})
        if not s:
            continue
        triggers = ''.join(f'<li>{esc(txt(t, lang))}</li>' for t in s.get('triggers', []))
        out.append(f"""
          <div class="scn" style="border-left:3px solid {colors[key]};">
            <div class="scn-head"><span class="scn-name" style="color:{colors[key]}">{labels[key][lang]}</span><span class="scn-prob">{esc(s.get('prob','—'))}%</span></div>
            <div class="scn-path">{esc(txt(s.get('path'), lang))}</div>
            <ul class="scn-triggers">{triggers}</ul>
          </div>
        """)
    out.append('</div>')
    return ''.join(out)

def render_trades(trades, lang, C):
    if not trades:
        return f'<div class="empty">{C["no_trades"]}</div>'
    rows = []
    for t in trades:
        rows.append(f"""
          <tr>
            <td><strong>{esc(txt(t.get('structure'), lang))}</strong></td>
            <td>${esc(t.get('strike','—'))}</td>
            <td>{esc(t.get('expiry','—'))}<br><span class="muted">{esc(t.get('dte','—'))}d</span></td>
            <td>{esc(t.get('credit','—'))}</td>
            <td class="rationale">{esc(txt(t.get('rationale'), lang))}</td>
            <td class="exit">{esc(txt(t.get('exit'), lang))}</td>
          </tr>
        """)
    return f"""
    <table class="trades">
      <thead><tr><th>{C['th_structure']}</th><th>{C['th_strike']}</th><th>{C['th_expiry']}</th><th>{C['th_credit']}</th><th>{C['th_rationale']}</th><th>{C['th_exit']}</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """

def render_list(items, lang, cls=""):
    if not items:
        return ""
    lis = ''.join(f'<li>{esc(txt(x, lang))}</li>' for x in items)
    return f'<ul class="{cls}">{lis}</ul>'

def render_ticker(t, lang, C):
    return f"""
    <section class="ticker">
      <h2>{esc(t['symbol'])} <span class="spot">${esc(t.get('spot','—'))}</span></h2>
      <h3>{C['setup']}</h3>
      {render_setup(t.get('setup', {}), lang, C)}
      <h3>{C['scenarios']}</h3>
      {render_scenarios(t.get('scenarios', {}), lang)}
      <h3>{C['trades']}</h3>
      {render_trades(t.get('trades', []), lang, C)}
      <div class="two-col">
        <div><h3>{C['risks']}</h3>{render_list(t.get('risks', []), lang, 'risks')}</div>
        <div><h3>{C['exits']}</h3>{render_list(t.get('exits', []), lang, 'exits')}</div>
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
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output', nargs='?')
    p.add_argument('--lang', choices=['en', 'zh'], default='en')
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    lang = args.lang
    C = CHROME[lang]

    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    suffix = '' if lang == 'en' else f'_{lang}'
    out = args.output or os.path.join(
        os.environ.get('FLOW_REPORT_DIR', 'reports'),
        f'briefing_{date}{suffix}.html'
    )

    tickers_html = ''.join(render_ticker(t, lang, C) for t in data.get('tickers', []))
    headline = txt(data.get('headline'), lang)
    headline_html = f'<div class="headline">{esc(headline)}</div>' if headline else ''
    week_ending = data.get('week_ending', '')
    week_label = f" — {C['week_ending']} {esc(week_ending)}" if week_ending else ""

    html_out = f"""<!DOCTYPE html>
<html lang="{lang}"><head><meta charset="utf-8">
<title>{C['title']} {esc(date)}</title>
<style>{CSS}</style>
</head><body>
<h1>{C['title']}</h1>
<div class="subtitle">{esc(date)}{week_label} · MSTR · TSLA</div>
{headline_html}
{tickers_html}
<div class="footer">
{C['footer_prefix']} <code>snapshots/trend-*.json</code> ·
<a href="index.html">{C['footer_link']}</a>
</div>
</body></html>"""

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(html_out)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
