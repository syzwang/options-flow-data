# Trade Journal

One markdown file per trade. Filename: `YYYY-MM-DD-TICKER-STRUCTURE.md` (e.g., `2026-04-13-MSTR-CC.md`).

The weekly briefing agent reads this directory every Sunday to:
1. Show open positions with their entry context.
2. Post-mortem closed trades from the past week.
3. Grade prediction accuracy and signal performance over time.

See `feedback_trade_journal_format.md` in `~/.claude/projects/-Users-xinwang/memory/` for the full template, or ask Claude in any conversation: "scaffold a journal entry for {trade}".

## Conventions

- Entry fields filled at trade open; exit fields filled at trade close.
- Always cite rule codes (E1, S1, M2 etc.) from the trading rulebook in the "Rules respected / overridden" lines.
- If a trade is rolled, close the old file with exit fields, then start a new file for the rolled trade and cross-link.
- Don't delete losing-trade journals — they're the most valuable.
