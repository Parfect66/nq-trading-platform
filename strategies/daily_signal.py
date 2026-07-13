"""
Daily Signal — runs after 10:00am ET each weekday
==================================================
1. Downloads fresh NQ data
2. Detects today's ORB signal
3. Scores it
4. Sends an email with the result

Run manually:
    py -3.12 strategies/daily_signal.py

Or triggered automatically by GitHub Actions (.github/workflows/daily_signal.yml).

Environment variables required (set as GitHub Secrets):
    GMAIL_USER          your Gmail address (used to send)
    GMAIL_APP_PASSWORD  Gmail app password (not your login password)
    ALERT_EMAIL         address to send the alert TO
"""

import os
import sys
import smtplib
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date, datetime

# Add project root to path so we can import from sibling folders
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Step 1 — Download fresh data
# ---------------------------------------------------------------------------

def download_data():
    print("Downloading fresh NQ data...")
    import yfinance as yf
    from datetime import timedelta

    end   = datetime.today()
    start = end - timedelta(days=60)

    df = yf.download(
        tickers="NQ=F",
        start=start,
        end=end,
        interval="30m",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Datetime"

    save_path = os.path.join(project_root, "data", "nq_data.csv")
    df.to_csv(save_path)
    print(f"  Saved {len(df)} bars to {save_path}")
    return df

# ---------------------------------------------------------------------------
# Step 2 — Detect today's ORB signal
# ---------------------------------------------------------------------------

def get_todays_signal(df):
    from strategies.orb import find_orb_signals

    signals = find_orb_signals(df, 30, 9, 30)
    if not signals:
        return None

    today_str = str(date.today())
    for s in signals:
        if s["Date"] == today_str:
            return s

    # Market may not have opened yet or no signal today
    return None

# ---------------------------------------------------------------------------
# Step 3 — Score today's signal
# ---------------------------------------------------------------------------

def score_signal(signal, df):
    """Score a single signal dict using the same factors as score_trades.py."""
    from strategies.score_trades import (
        add_indicators,
        score_trend_alignment,
        score_range_vs_atr,
        score_breakout_strength,
        score_volume_conviction,
        score_time_of_day,
        grade,
    )

    df = add_indicators(df.copy())
    df.index = df.index.astype("datetime64[us]")

    signal_time = pd.to_datetime(signal["Signal_Time"])
    if signal_time.tzinfo is not None:
        signal_time = signal_time.tz_localize(None)

    direction  = signal["Signal"]
    range_size = signal["Range_Size_Pts"]

    # Build a fake row as a Series so score functions work
    row = pd.Series({
        "Signal_Time":  signal_time,
        "Signal":       direction,
        "Range_High":   signal["Range_High"],
        "Range_Low":    signal["Range_Low"],
        "Signal_Price": signal["Signal_Price"],
        "Range_Size_Pts": range_size,
    })

    s_trend    = score_trend_alignment(signal_time, direction, df)
    s_atr      = score_range_vs_atr(range_size, signal_time, df)
    s_strength = score_breakout_strength(row, df)
    s_volume   = score_volume_conviction(signal_time, df)
    s_time     = score_time_of_day(signal_time)

    total = s_trend + s_atr + s_strength + s_volume + s_time

    return {
        "total":     total,
        "grade":     grade(total),
        "trend":     s_trend,
        "atr":       s_atr,
        "strength":  s_strength,
        "volume":    s_volume,
        "time":      s_time,
        "take":      "YES" if total >= 60 else "no",
    }

# ---------------------------------------------------------------------------
# Step 4 — Send email
# ---------------------------------------------------------------------------

def send_email(subject, body_html, body_text):
    gmail_user     = os.environ.get("GMAIL_USER")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")
    alert_email    = os.environ.get("ALERT_EMAIL")

    if not all([gmail_user, gmail_password, alert_email]):
        print("\n--- EMAIL NOT SENT (env vars not set) ---")
        print(f"Subject: {subject}")
        print(body_text)
        print("-----------------------------------------")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = alert_email
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, alert_email, msg.as_string())

    print(f"Email sent to {alert_email}")

# ---------------------------------------------------------------------------
# Build email content
# ---------------------------------------------------------------------------

STOP_PTS   = 50
TARGET_PTS = 250
DOLLARS_PER_PT = 20

def build_email(signal, score):
    today     = date.today().strftime("%A %d %B %Y")
    direction = signal["Signal"]
    entry     = float(signal["Signal_Price"])
    grade_str = score["grade"]
    take      = score["take"]

    stop_level   = entry - STOP_PTS  if direction == "BUY" else entry + STOP_PTS
    target_level = entry + TARGET_PTS if direction == "BUY" else entry - TARGET_PTS
    risk_dollars = STOP_PTS   * DOLLARS_PER_PT
    reward_dollars = TARGET_PTS * DOLLARS_PER_PT

    grade_colors = {"A": "#22c55e", "B": "#86efac", "C": "#fbbf24", "D": "#f97316", "F": "#ef4444"}
    grade_color  = grade_colors.get(grade_str, "#94a3b8")
    dir_color    = "#22c55e" if direction == "BUY" else "#ef4444"
    action_text  = "TAKE THE TRADE" if take == "YES" else "SKIP THIS TRADE"
    action_color = "#22c55e" if take == "YES" else "#ef4444"

    subject = f"NQ ORB Signal — {direction} Grade {grade_str} — {date.today()}"

    html = f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            background:#0f172a;color:#e2e8f0;padding:32px;border-radius:12px;max-width:520px">

  <h2 style="margin:0 0 4px;color:#f8fafc">NQ Opening Range Breakout</h2>
  <p style="color:#64748b;margin:0 0 24px">{today}</p>

  <div style="background:#1e293b;border-radius:8px;padding:20px;margin-bottom:16px;text-align:center">
    <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em">Signal</div>
    <div style="font-size:2.5rem;font-weight:700;color:{dir_color}">{direction}</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px">
    <div style="background:#1e293b;border-radius:8px;padding:14px;text-align:center">
      <div style="font-size:0.75rem;color:#94a3b8">Entry</div>
      <div style="font-size:1.2rem;font-weight:600">{entry:,.2f}</div>
    </div>
    <div style="background:#1e293b;border-radius:8px;padding:14px;text-align:center">
      <div style="font-size:0.75rem;color:#ef4444">Stop</div>
      <div style="font-size:1.2rem;font-weight:600">{stop_level:,.2f}</div>
    </div>
    <div style="background:#1e293b;border-radius:8px;padding:14px;text-align:center">
      <div style="font-size:0.75rem;color:#22c55e">Target</div>
      <div style="font-size:1.2rem;font-weight:600">{target_level:,.2f}</div>
    </div>
  </div>

  <div style="background:#1e293b;border-radius:8px;padding:16px;margin-bottom:16px">
    <div style="display:flex;justify-content:space-between;margin-bottom:8px">
      <span style="color:#94a3b8">Grade</span>
      <span style="background:{grade_color};color:#000;padding:2px 12px;border-radius:4px;font-weight:bold">{grade_str}</span>
    </div>
    <div style="display:flex;justify-content:space-between;margin-bottom:8px">
      <span style="color:#94a3b8">Score</span>
      <span>{score['total']}/100</span>
    </div>
    <div style="display:flex;justify-content:space-between;margin-bottom:8px">
      <span style="color:#94a3b8">Risk</span>
      <span style="color:#ef4444">${risk_dollars:,} ({STOP_PTS}pts)</span>
    </div>
    <div style="display:flex;justify-content:space-between">
      <span style="color:#94a3b8">Reward</span>
      <span style="color:#22c55e">${reward_dollars:,} ({TARGET_PTS}pts)</span>
    </div>
  </div>

  <div style="background:{action_color}20;border:1px solid {action_color};border-radius:8px;
              padding:14px;text-align:center;font-weight:700;color:{action_color};font-size:1.1rem">
    {action_text}
  </div>

  <div style="margin-top:16px;background:#1e293b;border-radius:8px;padding:14px;font-size:0.82rem;color:#64748b">
    Score breakdown: Trend {score['trend']}/20 &nbsp;·&nbsp;
    Range/ATR {score['atr']}/20 &nbsp;·&nbsp;
    Strength {score['strength']}/20 &nbsp;·&nbsp;
    Volume {score['volume']}/20 &nbsp;·&nbsp;
    Time {score['time']}/20
  </div>

  <p style="margin-top:16px;font-size:0.78rem;color:#475569">
    Paper trading only. Not financial advice. Settings: stop={STOP_PTS}pts, target={TARGET_PTS}pts, 1 NQ contract.
  </p>
</div>"""

    text = f"""NQ Opening Range Breakout — {today}

Signal   : {direction}
Entry    : {entry:,.2f}
Stop     : {stop_level:,.2f}  (-${risk_dollars:,})
Target   : {target_level:,.2f}  (+${reward_dollars:,})
Grade    : {grade_str}  ({score['total']}/100)
Action   : {action_text}

Score breakdown:
  Trend alignment  : {score['trend']}/20
  Range vs ATR     : {score['atr']}/20
  Breakout strength: {score['strength']}/20
  Volume conviction: {score['volume']}/20
  Time of day      : {score['time']}/20

Paper trading only. Not financial advice.
"""

    return subject, html, text


def build_no_signal_email():
    today   = date.today().strftime("%A %d %B %Y")
    subject = f"NQ ORB — No signal today ({date.today()})"
    html = f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            background:#0f172a;color:#e2e8f0;padding:32px;border-radius:12px;max-width:520px">
  <h2 style="margin:0 0 4px;color:#f8fafc">NQ Opening Range Breakout</h2>
  <p style="color:#64748b;margin:0 0 24px">{today}</p>
  <div style="background:#1e293b;border-radius:8px;padding:24px;text-align:center">
    <div style="font-size:1.4rem;color:#94a3b8">No breakout today</div>
    <p style="color:#64748b;margin-top:8px;font-size:0.9rem">
      NQ stayed inside the opening range. No trade to take.
    </p>
  </div>
</div>"""
    text = f"NQ ORB — {today}\n\nNo breakout today. NQ stayed inside the opening range.\n"
    return subject, html, text

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = download_data()

    signal = get_todays_signal(df)

    if signal is None or signal["Signal"] == "NONE":
        print("No signal today.")
        subject, html, text = build_no_signal_email()
    else:
        print(f"Signal: {signal['Signal']} at {signal['Signal_Price']}")
        score = score_signal(signal, df)
        print(f"Grade: {score['grade']}  Score: {score['total']}/100  →  {score['take']}")
        subject, html, text = build_email(signal, score)

    send_email(subject, html, text)
