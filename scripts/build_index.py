#!/usr/bin/env python3
"""Build reports/index.html — lists all dashboards newest first, for GitHub Pages."""
import os, re
from collections import defaultdict

REPORT_DIR = os.environ.get('FLOW_REPORT_DIR') or 'reports'

PATTERN = re.compile(r'trend_([A-Z]+)_(\d{4}-\d{2}-\d{2})(?:_(zh|en))?\.html$')
BRIEFING_PATTERN = re.compile(r'briefing_(\d{4}-\d{2}-\d{2})(?:_(zh|en))?\.html$')

def main():
    by_date = defaultdict(dict)
    briefings = defaultdict(dict)
    for fn in os.listdir(REPORT_DIR):
        m = PATTERN.match(fn)
        if m:
            ticker, date, lang = m.group(1), m.group(2), m.group(3) or 'en'
            by_date[date].setdefault(ticker, {})[lang] = fn
            continue
        m = BRIEFING_PATTERN.match(fn)
        if m:
            date, lang = m.group(1), m.group(2) or 'en'
            briefings[date][lang] = fn

    rows = []
    for date in sorted(briefings.keys(), reverse=True):
        langs = briefings[date]
        links = []
        for lang_key, label in (('zh', '中文'), ('en', 'EN')):
            if lang_key in langs:
                links.append(f'<a href="{langs[lang_key]}">{label}</a>')
        rows.append(f'<tr><td>{date}</td><td><strong>Weekly Briefing</strong></td><td>{" · ".join(links)}</td></tr>')
    for date in sorted(by_date.keys(), reverse=True):
        tickers = by_date[date]
        for ticker in sorted(tickers.keys()):
            langs = tickers[ticker]
            links = []
            for lang_key, label in (('zh', '中文'), ('en', 'EN')):
                if lang_key in langs:
                    links.append(f'<a href="{langs[lang_key]}">{label}</a>')
            rows.append(f'<tr><td>{date}</td><td>{ticker}</td><td>{" · ".join(links)}</td></tr>')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Options Flow Dashboards</title>
<style>
body {{ background:#0f1117; color:#e6edf3; font-family:-apple-system,sans-serif; max-width:720px; margin:40px auto; padding:0 20px; }}
h1 {{ font-size:22px; font-weight:500; margin-bottom:4px; }}
.sub {{ color:#8b949e; font-size:13px; margin-bottom:24px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ color:#8b949e; font-weight:500; text-align:left; padding:8px 10px; border-bottom:1px solid #30363d; }}
td {{ padding:8px 10px; border-bottom:1px solid #21262d; }}
a {{ color:#58a6ff; text-decoration:none; margin-right:4px; }}
a:hover {{ text-decoration:underline; }}
</style>
</head><body>
<h1>Options Flow Dashboards</h1>
<div class="sub">MSTR · TSLA · updated each US market close by GitHub Actions</div>
<table>
<tr><th>Date</th><th>Ticker</th><th>Dashboard</th></tr>
{chr(10).join(rows) if rows else '<tr><td colspan="3" style="color:#8b949e;">No reports yet.</td></tr>'}
</table>
</body></html>"""
    out = os.path.join(REPORT_DIR, 'index.html')
    with open(out, 'w') as f:
        f.write(html)
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == '__main__':
    main()
