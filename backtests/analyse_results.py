"""
Combined Analysis Report
========================
Joins the backtest results (did each trade win/lose?) with the scores
(how good did the setup look?) to answer the core question:

    Do high-scoring signals actually outperform low-scoring ones?

If yes → the scoring system is useful and we should only take Grade A/B trades.
If no  → the scoring factors need rethinking.

Also produces a full HTML report you can open in any browser.

How to run:
    python3 analyse_results.py

Outputs:
    orb_analysis.csv   — full merged table
    orb_report.html    — visual report, open in your browser
"""

import pandas as pd
import os

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BACKTEST_FILE = "orb_backtest_results.csv"
SCORED_FILE   = "orb_scored.csv"
OUTPUT_CSV    = "orb_analysis.csv"
OUTPUT_HTML   = "orb_report.html"

NQ_DOLLARS_PER_POINT = 20

# ---------------------------------------------------------------------------
# Load and merge
# ---------------------------------------------------------------------------

def load_and_merge():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    bt = pd.read_csv(os.path.join(script_dir, BACKTEST_FILE))
    sc = pd.read_csv(os.path.join(script_dir, SCORED_FILE))

    # Keep only the score columns we need from the scored file
    score_cols = ["Date", "Total_Score", "Grade", "Take_Trade",
                  "Score_Trend", "Score_RangeATR",
                  "Score_Strength", "Score_Volume", "Score_TimeOfDay"]
    sc = sc[score_cols]

    merged = bt.merge(sc, on="Date", how="inner")
    print(f"Merged {len(merged)} trades (backtest + scores).")
    return merged

# ---------------------------------------------------------------------------
# Analysis by grade
# ---------------------------------------------------------------------------

def analyse_by_grade(df):
    """
    Groups trades by letter grade and calculates win rate and avg P&L per group.
    This is the key table — if A trades beat F trades, the scorer works.
    """
    grade_order = ["A", "B", "C", "D", "F"]

    rows = []
    for grade in grade_order:
        group = df[df["Grade"] == grade]
        if group.empty:
            continue
        wins      = group["Win"].sum()
        total     = len(group)
        win_rate  = round(wins / total * 100, 1)
        avg_pnl   = round(group["Dollar_PnL"].mean(), 0)
        total_pnl = round(group["Dollar_PnL"].sum(), 0)
        rows.append({
            "Grade":       grade,
            "Trades":      total,
            "Wins":        int(wins),
            "Win_Rate_%":  win_rate,
            "Avg_PnL_$":   avg_pnl,
            "Total_PnL_$": total_pnl,
        })

    return pd.DataFrame(rows)


def analyse_filtered_vs_unfiltered(df):
    """
    Compares two hypothetical traders:
      Trader A: takes every signal regardless of score
      Trader B: only takes signals with Grade A or B
    """
    all_trades  = df
    high_grade  = df[df["Grade"].isin(["A", "B"])]
    low_grade   = df[~df["Grade"].isin(["A", "B"])]

    def summary(group, label):
        if group.empty:
            return {"Approach": label, "Trades": 0}
        wins     = group["Win"].sum()
        total    = len(group)
        win_rate = round(wins / total * 100, 1)
        total_pnl = round(group["Dollar_PnL"].sum(), 0)
        avg_pnl  = round(group["Dollar_PnL"].mean(), 0)
        return {
            "Approach":    label,
            "Trades":      total,
            "Win_Rate_%":  win_rate,
            "Total_PnL_$": total_pnl,
            "Avg_PnL_$":   avg_pnl,
        }

    return pd.DataFrame([
        summary(all_trades, "All signals (no filter)"),
        summary(high_grade, "Grade A + B only"),
        summary(low_grade,  "Grade C, D, F only"),
    ])


def cumulative_pnl_by_grade(df):
    """Returns separate cumulative P&L series for A/B vs C/D/F trades."""
    df = df.sort_values("Date").copy()
    df["Cumulative_All"]    = df["Dollar_PnL"].cumsum()
    high = df[df["Grade"].isin(["A","B"])].copy()
    low  = df[~df["Grade"].isin(["A","B"])].copy()
    high["Cumulative_High"] = high["Dollar_PnL"].cumsum()
    low["Cumulative_Low"]   = low["Dollar_PnL"].cumsum()
    return df, high, low

# ---------------------------------------------------------------------------
# Print report to terminal
# ---------------------------------------------------------------------------

def print_terminal_report(by_grade, comparison, df):
    print("\n" + "=" * 60)
    print("  COMBINED ANALYSIS — Does scoring predict winners?")
    print("=" * 60)

    print("\n  Performance by Grade:")
    print(by_grade.to_string(index=False))

    print("\n  Filtered vs Unfiltered:")
    print(comparison.to_string(index=False))

    # Verdict
    a_trades = by_grade[by_grade["Grade"] == "A"]
    f_trades = by_grade[by_grade["Grade"] == "F"]

    print("\n  Verdict:")
    if not a_trades.empty and not f_trades.empty:
        a_wr = a_trades["Win_Rate_%"].values[0]
        f_wr = f_trades["Win_Rate_%"].values[0]
        if a_wr > f_wr:
            print(f"  ✓ Grade A trades win {a_wr}% vs Grade F win {f_wr}%")
            print("    The scoring system IS adding value.")
        else:
            print(f"  ✗ Grade A wins {a_wr}% vs Grade F wins {f_wr}%")
            print("    The scoring factors need adjustment.")
    else:
        print("  Not enough data across all grades to compare A vs F.")

# ---------------------------------------------------------------------------
# Build HTML report
# ---------------------------------------------------------------------------

def build_html_report(df, by_grade, comparison):
    """Builds a self-contained HTML file with tables and a chart."""

    # Prepare chart data — cumulative P&L over time for each approach
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    df_sorted["Cum_All"]  = df_sorted["Dollar_PnL"].cumsum()

    high = df_sorted[df_sorted["Grade"].isin(["A","B"])].copy()
    low  = df_sorted[~df_sorted["Grade"].isin(["A","B"])].copy()
    high["Cum_High"] = high["Dollar_PnL"].cumsum()
    low["Cum_Low"]   = low["Dollar_PnL"].cumsum()

    # All three lines must share the same x-axis labels or Chart.js drops points.
    # Build a unified date list and forward-fill cumulative values for each series.
    all_dates   = df_sorted["Date"].tolist()
    high_map    = dict(zip(high["Date"], high["Cum_High"]))
    low_map     = dict(zip(low["Date"],  low["Cum_Low"]))

    all_labels  = [f'"{d}"' for d in all_dates]
    all_values  = df_sorted["Cum_All"].tolist()

    high_values, low_values = [], []
    last_h = last_l = 0
    for d in all_dates:
        if d in high_map: last_h = high_map[d]
        if d in low_map:  last_l = low_map[d]
        high_values.append(last_h)
        low_values.append(last_l)

    # Grade table rows
    grade_rows = ""
    for _, r in by_grade.iterrows():
        color = {"A":"#22c55e","B":"#86efac","C":"#fbbf24","D":"#f97316","F":"#ef4444"}.get(r["Grade"],"#ccc")
        pnl_color = "#22c55e" if r["Total_PnL_$"] >= 0 else "#ef4444"
        grade_rows += f"""
        <tr>
          <td><span style="background:{color};color:#000;padding:2px 10px;border-radius:4px;font-weight:bold">{r['Grade']}</span></td>
          <td>{r['Trades']}</td>
          <td>{r['Win_Rate_%']}%</td>
          <td>${r['Avg_PnL_$']:,}</td>
          <td style="color:{pnl_color};font-weight:bold">${r['Total_PnL_$']:,}</td>
        </tr>"""

    # Comparison table rows
    comp_rows = ""
    for _, r in comparison.iterrows():
        pnl_color = "#22c55e" if r["Total_PnL_$"] >= 0 else "#ef4444"
        comp_rows += f"""
        <tr>
          <td>{r['Approach']}</td>
          <td>{r['Trades']}</td>
          <td>{r['Win_Rate_%']}%</td>
          <td>${r['Avg_PnL_$']:,}</td>
          <td style="color:{pnl_color};font-weight:bold">${r['Total_PnL_$']:,}</td>
        </tr>"""

    # Recent trades table (last 15)
    recent = df_sorted.tail(15)[["Date","Direction","Entry_Price","Exit_Price",
                                  "Exit_Reason","Points_PnL","Dollar_PnL","Grade","Total_Score"]]
    trade_rows = ""
    for _, r in recent.iterrows():
        pnl_color = "#22c55e" if r["Dollar_PnL"] >= 0 else "#ef4444"
        dir_color = "#60a5fa" if r["Direction"] == "BUY" else "#f87171"
        grade_color = {"A":"#22c55e","B":"#86efac","C":"#fbbf24","D":"#f97316","F":"#ef4444"}.get(r["Grade"],"#ccc")
        trade_rows += f"""
        <tr>
          <td>{r['Date']}</td>
          <td style="color:{dir_color};font-weight:bold">{r['Direction']}</td>
          <td>{r['Entry_Price']}</td>
          <td>{r['Exit_Price']}</td>
          <td>{r['Exit_Reason']}</td>
          <td style="color:{pnl_color}">{r['Points_PnL']:+.1f} pts</td>
          <td style="color:{pnl_color};font-weight:bold">${r['Dollar_PnL']:,.0f}</td>
          <td><span style="background:{grade_color};color:#000;padding:1px 8px;border-radius:4px">{r['Grade']}</span></td>
          <td>{r['Total_Score']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NQ ORB Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1   {{ font-size: 1.6rem; margin-bottom: 4px; color: #f8fafc; }}
  .sub {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 32px; }}
  h2   {{ font-size: 1.1rem; color: #cbd5e1; margin: 32px 0 12px; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 8px; }}
  .card {{ background: #1e293b; border-radius: 10px; padding: 16px 20px; }}
  .card .label {{ font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }}
  .card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
  .green {{ color: #22c55e; }} .red {{ color: #ef4444; }} .blue {{ color: #60a5fa; }}
  table  {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; }}
  th     {{ background: #0f172a; color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;
            letter-spacing: .05em; padding: 10px 14px; text-align: left; }}
  td     {{ padding: 10px 14px; border-top: 1px solid #334155; font-size: 0.88rem; }}
  tr:hover td {{ background: #263347; }}
  .chart-wrap {{ background: #1e293b; border-radius: 10px; padding: 20px; margin-top: 8px; }}
</style>
</head>
<body>

<h1>NQ Futures — Opening Range Breakout</h1>
<p class="sub">Backtest Report &nbsp;·&nbsp; Real NQ futures data via Yahoo Finance &nbsp;·&nbsp; Re-run the pipeline to refresh</p>

<h2>Overall Summary</h2>
<div class="grid">
  <div class="card"><div class="label">Total Trades</div><div class="value blue">{len(df_sorted)}</div></div>
  <div class="card"><div class="label">Win Rate</div><div class="value">{round(df_sorted['Win'].mean()*100,1)}%</div></div>
  <div class="card"><div class="label">Total P&amp;L</div><div class="value {'green' if df_sorted['Dollar_PnL'].sum()>=0 else 'red'}">${df_sorted['Dollar_PnL'].sum():,.0f}</div></div>
  <div class="card"><div class="label">Avg Win</div><div class="value green">${df_sorted.loc[df_sorted['Win'],'Dollar_PnL'].mean():,.0f}</div></div>
  <div class="card"><div class="label">Avg Loss</div><div class="value red">${df_sorted.loc[~df_sorted['Win'],'Dollar_PnL'].mean():,.0f}</div></div>
</div>

<h2>Cumulative P&L by Approach</h2>
<div class="chart-wrap">
  <canvas id="pnlChart" height="100"></canvas>
</div>

<h2>Performance by Grade</h2>
<table>
  <tr><th>Grade</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th><th>Total P&L</th></tr>
  {grade_rows}
</table>

<h2>Filtered vs Unfiltered</h2>
<table>
  <tr><th>Approach</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th><th>Total P&L</th></tr>
  {comp_rows}
</table>

<h2>Recent Trades (last 15)</h2>
<table>
  <tr><th>Date</th><th>Dir</th><th>Entry</th><th>Exit</th><th>How</th><th>Points</th><th>P&L</th><th>Grade</th><th>Score</th></tr>
  {trade_rows}
</table>

<script>
const labels = [{','.join(all_labels)}];
const ctx = document.getElementById('pnlChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: labels,
    datasets: [
      {{
        label: 'All signals',
        data: [{','.join(str(v) for v in all_values)}],
        borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.08)',
        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
      }},
      {{
        label: 'Grade A + B only',
        data: [{','.join(str(v) for v in high_values)}],
        borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.08)',
        borderWidth: 2.5, pointRadius: 0, tension: 0.3, fill: false,
      }},
      {{
        label: 'Grade C, D, F only',
        data: [{','.join(str(v) for v in low_values)}],
        borderColor: '#f87171', backgroundColor: 'rgba(248,113,113,0.08)',
        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
      }},
    ]
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ ticks: {{ color:'#64748b', maxTicksLimit: 10 }}, grid: {{ color:'#1e293b' }} }},
      y: {{ ticks: {{ color:'#64748b', callback: v => '$'+v.toLocaleString() }}, grid: {{ color:'#334155' }} }}
    }},
    plugins: {{
      legend: {{ labels: {{ color:'#cbd5e1' }} }},
      tooltip: {{ callbacks: {{ label: ctx => ' $' + ctx.parsed.y.toLocaleString() }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    return html

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_and_merge()

    by_grade   = analyse_by_grade(df)
    comparison = analyse_filtered_vs_unfiltered(df)

    print_terminal_report(by_grade, comparison, df)

    # Save merged CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(script_dir, OUTPUT_CSV), index=False)
    print(f"\nMerged data saved to: {OUTPUT_CSV}")

    # Save HTML report
    html = build_html_report(df, by_grade, comparison)
    html_path = os.path.join(script_dir, OUTPUT_HTML)
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to:  {OUTPUT_HTML}")
    print("→ Open that file in your browser to see the visual report.")
