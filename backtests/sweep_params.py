"""
Parameter Sweep — Stop Loss & Profit Target Optimiser
======================================================
Tests every combination of stop loss and profit target from the
ranges defined below, and ranks them by total profit.

This answers: "What stop and target settings work best on this data?"

How to run:
    python3 sweep_params.py

Output:
    sweep_results.csv   — full table of every combination tested
    sweep_report.html   — heatmap + ranked table, open in your browser

Warning on "overfitting": the best combination on past data may not be
the best in the future. Look for a REGION of good settings (where many
nearby combinations all work), not just the single best row.
"""

import pandas as pd
import os
import itertools

# ---------------------------------------------------------------------------
# Settings — edit these ranges to widen or narrow the search
# ---------------------------------------------------------------------------

STOP_LOSS_RANGE    = [50, 75, 100, 125, 150, 175, 200]   # points
TARGET_RANGE       = [100, 125, 150, 175, 200, 250, 300]  # points
HOLD_HOURS_RANGE   = [4, 6]                               # hours

SIGNALS_FILE       = "orb_signals.csv"
PRICE_DATA         = "../data/nq_data.csv"
OUTPUT_CSV         = "sweep_results.csv"
OUTPUT_HTML        = "sweep_report.html"

NQ_DOLLARS_PER_POINT = 20

# ---------------------------------------------------------------------------
# Load data once (shared across all runs)
# ---------------------------------------------------------------------------

def load_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    signals = pd.read_csv(os.path.join(script_dir, SIGNALS_FILE))
    signals = signals[signals["Signal"] != "NONE"].copy()
    _st = pd.to_datetime(signals["Signal_Time"])
    if _st.dt.tz is not None:
        _st = _st.dt.tz_localize(None)
    signals["Signal_Time"] = _st

    prices = pd.read_csv(
        os.path.join(script_dir, PRICE_DATA),
        index_col="Datetime", parse_dates=True
    )
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index = prices.index.astype("datetime64[us]")

    print(f"Loaded {len(signals)} signals and {len(prices)} price bars.")
    return signals, prices

# ---------------------------------------------------------------------------
# Simulate one trade (same logic as backtest_orb.py)
# ---------------------------------------------------------------------------

def simulate_trade(signal_row, prices, stop_pts, target_pts, hold_hours):
    direction    = signal_row["Signal"]
    entry_price  = float(signal_row["Signal_Price"])
    signal_time  = pd.Timestamp(signal_row["Signal_Time"])
    exit_deadline = signal_time + pd.Timedelta(hours=hold_hours)

    if direction == "BUY":
        stop_level   = entry_price - stop_pts
        target_level = entry_price + target_pts
    else:
        stop_level   = entry_price + stop_pts
        target_level = entry_price - target_pts

    bars_after_entry = prices[prices.index > signal_time]
    exit_price  = None
    exit_reason = None

    for ts, bar in bars_after_entry.iterrows():
        if ts > exit_deadline:
            exit_price  = bar["Open"]
            exit_reason = "TIME"
            break
        if direction == "BUY" and bar["Low"] <= stop_level:
            exit_price  = stop_level
            exit_reason = "STOP"
            break
        if direction == "SELL" and bar["High"] >= stop_level:
            exit_price  = stop_level
            exit_reason = "STOP"
            break
        if direction == "BUY" and bar["High"] >= target_level:
            exit_price  = target_level
            exit_reason = "TARGET"
            break
        if direction == "SELL" and bar["Low"] <= target_level:
            exit_price  = target_level
            exit_reason = "TARGET"
            break

    if exit_price is None:
        last_bar    = bars_after_entry.iloc[-1] if not bars_after_entry.empty else None
        exit_price  = last_bar["Close"] if last_bar is not None else entry_price
        exit_reason = "END_OF_DATA"

    if direction == "BUY":
        points_pnl = exit_price - entry_price
    else:
        points_pnl = entry_price - exit_price

    return points_pnl * NQ_DOLLARS_PER_POINT

# ---------------------------------------------------------------------------
# Run one combination and return a summary row
# ---------------------------------------------------------------------------

def run_combination(signals, prices, stop_pts, target_pts, hold_hours):
    pnls = []
    for _, row in signals.iterrows():
        pnl = simulate_trade(row, prices, stop_pts, target_pts, hold_hours)
        pnls.append(pnl)

    pnls   = pd.Series(pnls)
    wins   = (pnls > 0).sum()
    total  = len(pnls)
    cumulative = pnls.cumsum()
    max_dd = (cumulative.cummax() - cumulative).max()

    gross_profit = pnls[pnls > 0].sum()
    gross_loss   = abs(pnls[pnls < 0].sum())
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0

    return {
        "Stop_Pts":      stop_pts,
        "Target_Pts":    target_pts,
        "Hold_Hours":    hold_hours,
        "RR_Ratio":      round(target_pts / stop_pts, 2),
        "Trades":        total,
        "Wins":          int(wins),
        "Win_Rate_%":    round(wins / total * 100, 1),
        "Total_PnL_$":   round(pnls.sum(), 0),
        "Avg_PnL_$":     round(pnls.mean(), 0),
        "Profit_Factor": profit_factor,
        "Max_DD_$":      round(max_dd, 0),
    }

# ---------------------------------------------------------------------------
# Build HTML report
# ---------------------------------------------------------------------------

def build_html(results_df):
    # Top 20 rows sorted by total P&L
    top20 = results_df.sort_values("Total_PnL_$", ascending=False).head(20)

    table_rows = ""
    for i, (_, r) in enumerate(top20.iterrows()):
        pnl_color  = "#22c55e" if r["Total_PnL_$"] >= 0 else "#ef4444"
        rank_badge = f'<span style="background:#334155;color:#94a3b8;padding:1px 7px;border-radius:4px;font-size:0.8rem">#{i+1}</span>'
        table_rows += f"""
        <tr>
          <td>{rank_badge}</td>
          <td><b>{r['Stop_Pts']}</b></td>
          <td><b>{r['Target_Pts']}</b></td>
          <td>{r['Hold_Hours']}h</td>
          <td>{r['RR_Ratio']}:1</td>
          <td>{r['Win_Rate_%']}%</td>
          <td style="color:{pnl_color};font-weight:bold">${r['Total_PnL_$']:,.0f}</td>
          <td>{r['Avg_PnL_$']:+,.0f}</td>
          <td>{r['Profit_Factor']}</td>
          <td style="color:#f87171">${r['Max_DD_$']:,.0f}</td>
        </tr>"""

    # Heatmap: stop vs target, colour = total P&L (fixed hold = most common)
    # Pick the hold_hours value that appears most in top20
    best_hold = int(top20["Hold_Hours"].mode().iloc[0])
    hm_data = results_df[results_df["Hold_Hours"] == best_hold].copy()

    stops   = sorted(STOP_LOSS_RANGE)
    targets = sorted(TARGET_RANGE)

    # min/max for colour scale
    pnl_min = hm_data["Total_PnL_$"].min()
    pnl_max = hm_data["Total_PnL_$"].max()

    def pnl_to_color(pnl):
        if pnl_max == pnl_min:
            return "#94a3b8"
        ratio = (pnl - pnl_min) / (pnl_max - pnl_min)
        if pnl < 0:
            r = 239; g = int(68 + (ratio * 0.5) * 150); b = 68
        else:
            r = int(34 + (1 - ratio) * 60); g = int(197 - (1 - ratio) * 80); b = int(94 - (1 - ratio) * 40)
        return f"rgb({r},{g},{b})"

    hm_rows = ""
    for stop in stops:
        hm_rows += "<tr>"
        hm_rows += f'<td style="color:#94a3b8;font-size:0.8rem;padding:6px 10px">{stop}pt stop</td>'
        for target in targets:
            cell = hm_data[(hm_data["Stop_Pts"] == stop) & (hm_data["Target_Pts"] == target)]
            if cell.empty:
                hm_rows += '<td style="background:#1e293b">—</td>'
            else:
                pnl = cell["Total_PnL_$"].values[0]
                bg  = pnl_to_color(pnl)
                text_color = "#000" if pnl > (pnl_min + (pnl_max - pnl_min) * 0.6) else "#fff"
                hm_rows += f'<td style="background:{bg};color:{text_color};text-align:center;padding:6px 4px;font-size:0.8rem">${pnl:,.0f}</td>'
        hm_rows += "</tr>"

    target_headers = "".join(
        f'<th style="text-align:center;color:#94a3b8;font-size:0.78rem">{t}pt target</th>'
        for t in targets
    )

    best = results_df.sort_values("Total_PnL_$", ascending=False).iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NQ ORB — Parameter Sweep</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1  {{ font-size: 1.6rem; margin-bottom: 4px; color: #f8fafc; }}
  .sub {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 32px; }}
  h2  {{ font-size: 1.1rem; color: #cbd5e1; margin: 32px 0 12px; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 8px; }}
  .card {{ background: #1e293b; border-radius: 10px; padding: 16px 20px; }}
  .card .label {{ font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }}
  .card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
  .green {{ color: #22c55e; }} .blue {{ color: #60a5fa; }} .yellow {{ color: #fbbf24; }}
  table  {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; }}
  th     {{ background: #0f172a; color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;
            letter-spacing: .05em; padding: 10px 14px; text-align: left; }}
  td     {{ padding: 10px 14px; border-top: 1px solid #334155; font-size: 0.88rem; }}
  tr:hover td {{ background: #263347; }}
  .hm-wrap {{ background: #1e293b; border-radius: 10px; padding: 20px; overflow-x: auto; }}
  .hm-wrap table {{ background: transparent; }}
  .hm-wrap td {{ border: 1px solid #0f172a; }}
  .hint {{ color: #64748b; font-size: 0.82rem; margin-top: 8px; }}
</style>
</head>
<body>

<h1>NQ ORB — Parameter Sweep</h1>
<p class="sub">Testing {len(results_df)} stop/target/hold combinations &nbsp;·&nbsp; Real Yahoo Finance data</p>

<h2>Best Single Combination Found</h2>
<div class="grid">
  <div class="card"><div class="label">Best Stop Loss</div><div class="value blue">{int(best['Stop_Pts'])} pts</div></div>
  <div class="card"><div class="label">Best Target</div><div class="value blue">{int(best['Target_Pts'])} pts</div></div>
  <div class="card"><div class="label">Hold Time</div><div class="value">{int(best['Hold_Hours'])}h</div></div>
  <div class="card"><div class="label">Risk:Reward</div><div class="value yellow">{best['RR_Ratio']}:1</div></div>
  <div class="card"><div class="label">Total P&amp;L</div><div class="value green">${best['Total_PnL_$']:,.0f}</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value">{best['Win_Rate_%']}%</div></div>
</div>
<p class="hint" style="margin-top:12px">⚠ Tip: Don't just use the #1 row. Look for a region where many nearby combinations are also profitable — that's more likely to hold up in live trading.</p>

<h2>P&L Heatmap — Stop vs Target (hold={best_hold}h)</h2>
<div class="hm-wrap">
  <table>
    <tr><th></th>{target_headers}</tr>
    {hm_rows}
  </table>
  <p class="hint" style="margin-top:10px">Green = profitable &nbsp;·&nbsp; Red = losing &nbsp;·&nbsp; Numbers are total $ P&L across all {results_df['Trades'].iloc[0]} trades</p>
</div>

<h2>Top 20 Combinations (ranked by Total P&L)</h2>
<table>
  <tr><th>Rank</th><th>Stop</th><th>Target</th><th>Hold</th><th>R:R</th><th>Win Rate</th><th>Total P&L</th><th>Avg/Trade</th><th>Prof Factor</th><th>Max Drawdown</th></tr>
  {table_rows}
</table>

</body>
</html>"""
    return html

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    signals, prices = load_files()

    combinations = list(itertools.product(STOP_LOSS_RANGE, TARGET_RANGE, HOLD_HOURS_RANGE))
    total = len(combinations)
    print(f"\nTesting {total} combinations ...\n")

    rows = []
    for i, (stop, target, hold) in enumerate(combinations, 1):
        result = run_combination(signals, prices, stop, target, hold)
        rows.append(result)
        status = "✓" if result["Total_PnL_$"] >= 0 else "✗"
        print(f"  [{i:3}/{total}]  stop={stop:3}  target={target:3}  hold={hold}h  "
              f"→  {status}  ${result['Total_PnL_$']:>8,.0f}  (win rate {result['Win_Rate_%']}%)")

    results_df = pd.DataFrame(rows).sort_values("Total_PnL_$", ascending=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, OUTPUT_CSV)
    results_df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {OUTPUT_CSV}")

    html = build_html(results_df)
    html_path = os.path.join(script_dir, OUTPUT_HTML)
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to:  {OUTPUT_HTML}")
    print("→ Open that file in your browser to see the heatmap.")

    best = results_df.iloc[0]
    print(f"\nBest combination:")
    print(f"  Stop loss    : {int(best['Stop_Pts'])} pts")
    print(f"  Target       : {int(best['Target_Pts'])} pts")
    print(f"  Hold time    : {int(best['Hold_Hours'])} hours")
    print(f"  Total P&L    : ${best['Total_PnL_$']:,.0f}")
    print(f"  Win rate     : {best['Win_Rate_%']}%")
    print(f"  Profit factor: {best['Profit_Factor']}")
    print(f"\nTo use these settings, edit backtests/backtest_orb.py:")
    print(f"  STOP_LOSS_POINTS  = {int(best['Stop_Pts'])}")
    print(f"  PROFIT_TARGET_PTS = {int(best['Target_Pts'])}")
    print(f"  HOLD_HOURS        = {int(best['Hold_Hours'])}")
