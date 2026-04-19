# options-flow-data

Daily MSTR/TSLA options flow snapshots + trend dashboards. Runs via GitHub Actions so data keeps accumulating even when the laptop is off.

## Layout

- `scripts/snapshot.py` — daily snapshot runner
- `scripts/trend.py` — multi-day trend analysis + HTML dashboard
- `snapshots/` — daily JSON (`YYYY-MM-DD-TICKER.json`) committed by CI
- `reports/` — generated HTML dashboards (EN + ZH), served via GitHub Pages

## Local usage

```bash
python3 scripts/snapshot.py MSTR TSLA
python3 scripts/trend.py MSTR TSLA --html --mode postclose --lang zh
```

By default scripts read/write `~/options_portfolio/flow_snapshots` and `~/options_reports`. Override with env vars:

```bash
FLOW_SNAPSHOT_DIR=./snapshots FLOW_REPORT_DIR=./reports python3 scripts/trend.py MSTR TSLA --html
```

## Schedule

`.github/workflows/daily.yml` runs Mon–Fri at **21:30 UTC** (after US market close in both EDT and EST), runs `snapshot.py` → `trend.py` (EN + ZH) → commits results.

Manually trigger from the Actions tab (`workflow_dispatch`) to test.
