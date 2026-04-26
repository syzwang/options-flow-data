#!/usr/bin/env python3
"""Scrape MSTR BTC holdings + capital structure from strategy.com.

Writes data/mstr_holdings.json:
  {btc_holdings, as_of_date, basic_shares_outstanding, debt, pref, cash, source, fetched_at}

The strategy.com homepage embeds a JSON blob with these fields — no auth, no API.
Requires a browser User-Agent (plain curl gets 403).
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

URL = "https://www.strategy.com/"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
FIELDS = ["btc_holdings", "as_of_date", "basic_shares_outstanding", "debt", "pref", "cash"]
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "mstr_holdings.json")


def fetch_html() -> str:
    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def extract(html: str) -> dict:
    out = {}
    for field in FIELDS:
        m = re.search(rf'"{field}"\s*:\s*("[^"]+"|[0-9.]+)', html)
        if not m:
            raise RuntimeError(f"field {field!r} not found in strategy.com homepage")
        raw = m.group(1)
        out[field] = raw.strip('"') if raw.startswith('"') else (int(float(raw)) if "." not in raw else float(raw))
    return out


def main():
    html = fetch_html()
    data = extract(html)
    data["source"] = URL
    data["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out_path = os.path.abspath(OUT)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}: btc={data['btc_holdings']:,} as_of={data['as_of_date']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"fetch_mstr_holdings failed: {e}", file=sys.stderr)
        sys.exit(1)
