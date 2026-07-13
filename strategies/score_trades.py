"""
Trade Scorer — Improved
========================
Gives each ORB signal a score out of 100 across five factors that
are each independently meaningful on real NQ price data.

WHY THE OLD SCORER WAS REPLACED
--------------------------------
The original four factors had two problems:
  - Risk/reward: stop and target are fixed numbers, so this scored
    identically on almost every trade. It wasn't measuring anything.
  - Range size: compared to the whole-dataset average instead of
    recent volatility, so "tight" had no consistent meaning.

THE FIVE NEW FACTORS (20 pts each = 100 max)
---------------------------------------------
  1. Trend alignment   — does the breakout direction match the trend?
                         (BUY when price has been rising, SELL when falling)
  2. Range vs ATR      — how tight was the range relative to recent volatility?
                         ATR = Average True Range, a standard volatility measure
  3. Breakout strength — did price close convincingly beyond the range,
                         or barely scratch through by 1 point?
  4. Volume conviction — was breakout bar volume above its recent daily average?
  5. Time of day       — earlier breakouts tend to capture stronger moves

How to run:
    python3 score_trades.py

Output: backtests/orb_scored.csv
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SIGNALS_FILE          = "../backtests/orb_signals.csv"
PRICE_DATA            = "../data/nq_data.csv"
RESULTS_FILE          = "../backtests/orb_scored.csv"
HIGH_QUALITY_THRESHOLD = 60   # scores at or above this = "take the trade"

# How many past bars to use when calculating trend and average volume
TREND_LOOKBACK_BARS   = 10    # bars for moving average (trend direction)
VOLUME_LOOKBACK_BARS  = 20    # bars for average daily volume comparison
ATR_LOOKBACK_BARS     = 14    # bars for Average True Range (volatility)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    signals = pd.read_csv(os.path.join(script_dir, SIGNALS_FILE))
    signals = signals[signals["Signal"] != "NONE"].copy()
    signals["Signal_Time"] = (
        pd.to_datetime(signals["Signal_Time"], utc=True)
        .dt.tz_convert("America/New_York")
    )

    prices = pd.read_csv(
        os.path.join(script_dir, PRICE_DATA),
        index_col="Datetime", parse_dates=True
    )
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize(
            "America/New_York", ambiguous="infer", nonexistent="shift_forward"
        )

    # Pre-calculate indicators once here so every scoring function can use them
    prices = add_indicators(prices)

    print(f"Loaded {len(signals)} signals to score.")
    return signals, prices


def add_indicators(df):
    """
    Adds columns to the price data that the scoring functions need.

    Moving average (MA): the average closing price over the last N bars.
      If today's close is above the MA, the short-term trend is up.

    ATR (Average True Range): measures how much NQ typically moves per bar.
      True Range = the biggest of:
        - High minus Low of this bar
        - High minus previous Close (gap up)
        - previous Close minus Low (gap down)
      ATR = rolling average of True Range over N bars.
      Used to judge whether today's opening range is tight or wide.
    """
    df = df.copy()

    # Moving average of closing prices
    df["MA"] = df["Close"].rolling(window=TREND_LOOKBACK_BARS).mean()

    # True Range
    prev_close   = df["Close"].shift(1)
    high_low     = df["High"] - df["Low"]
    high_pc      = (df["High"] - prev_close).abs()
    low_pc       = (df["Low"]  - prev_close).abs()
    df["TR"]     = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)

    # ATR = rolling average of TR
    df["ATR"]    = df["TR"].rolling(window=ATR_LOOKBACK_BARS).mean()

    # Rolling average volume
    df["Avg_Volume"] = df["Volume"].rolling(window=VOLUME_LOOKBACK_BARS).mean()

    return df

# ---------------------------------------------------------------------------
# Scoring factors — each returns 0 to 20
# ---------------------------------------------------------------------------

def score_trend_alignment(signal_time, direction, prices):
    """
    FACTOR 1: Trend alignment (20 pts)

    Checks whether the breakout direction matches the short-term trend.
    Uses a simple moving average: if today's price is above the MA,
    the trend is UP — so a BUY signal is "with the trend" (good).
    A SELL signal against an uptrend is "counter-trend" (risky).

    Trend with signal  → 20 pts
    No trend (choppy)  → 10 pts
    Signal vs trend    →  0 pts
    """
    # Get the bar just before the breakout (we don't know the future)
    bars_before = prices[prices.index < signal_time]
    if bars_before.empty or bars_before["MA"].isna().all():
        return 10  # not enough data — neutral score

    last_bar   = bars_before.iloc[-1]
    close      = last_bar["Close"]
    ma         = last_bar["MA"]

    if pd.isna(ma):
        return 10

    # Price well above MA = uptrend; well below = downtrend
    # "Well" means more than 0.1% difference (avoids scoring flat markets)
    threshold = close * 0.001

    trending_up   = close > ma + threshold
    trending_down = close < ma - threshold

    if   direction == "BUY"  and trending_up:   return 20  # with trend
    elif direction == "SELL" and trending_down:  return 20  # with trend
    elif direction == "BUY"  and trending_down:  return  0  # against trend
    elif direction == "SELL" and trending_up:    return  0  # against trend
    else:                                        return 10  # unclear trend


def score_range_vs_atr(range_size, signal_time, prices):
    """
    FACTOR 2: Range size relative to ATR (20 pts)

    A tight opening range relative to recent volatility means the market
    compressed before the move — like a coiled spring.

    We compare today's range to ATR: if the range is much smaller than
    ATR, the market was unusually quiet in the opening window.

    Range < 30% of ATR  → 20 pts  (very tight — coiled)
    Range < 50% of ATR  → 15 pts
    Range < 75% of ATR  → 10 pts
    Range < 100% of ATR →  5 pts
    Range ≥ ATR         →  0 pts  (range was already large — less room to move)
    """
    bars_before = prices[prices.index < signal_time]
    if bars_before.empty or bars_before["ATR"].isna().all():
        return 10

    atr = bars_before["ATR"].dropna().iloc[-1]
    if atr == 0 or pd.isna(atr):
        return 10

    ratio = range_size / atr  # e.g. 0.4 = range was 40% of ATR

    if   ratio < 0.30: return 20
    elif ratio < 0.50: return 15
    elif ratio < 0.75: return 10
    elif ratio < 1.00: return  5
    else:              return  0


def score_breakout_strength(signal_row, prices):
    """
    FACTOR 3: Breakout strength (20 pts)

    Measures HOW FAR beyond the range price closed on the breakout bar.
    A strong close well beyond the range = conviction.
    A close that barely scraped through by 1 point = weak.

    We measure this as a ratio:
      (close - range boundary) / ATR

    i.e. how many ATR-widths beyond the range did price close?

    > 0.30 ATR beyond the range → 20 pts
    > 0.20 ATR beyond           → 15 pts
    > 0.10 ATR beyond           → 10 pts
    > 0.05 ATR beyond           →  5 pts
    ≤ 0.05 ATR beyond           →  0 pts  (barely a breakout)
    """
    signal_time  = signal_row["Signal_Time"]
    direction    = signal_row["Signal"]
    range_high   = signal_row["Range_High"]
    range_low    = signal_row["Range_Low"]
    signal_price = float(signal_row["Signal_Price"])

    bars_before = prices[prices.index < signal_time]
    if bars_before.empty or bars_before["ATR"].isna().all():
        return 10

    atr = bars_before["ATR"].dropna().iloc[-1]
    if atr == 0 or pd.isna(atr):
        return 10

    if direction == "BUY":
        extension = signal_price - range_high   # how far above the high
    else:
        extension = range_low - signal_price    # how far below the low

    ratio = extension / atr

    if   ratio > 0.30: return 20
    elif ratio > 0.20: return 15
    elif ratio > 0.10: return 10
    elif ratio > 0.05: return  5
    else:              return  0


def score_volume_conviction(signal_time, prices):
    """
    FACTOR 4: Volume conviction (20 pts)

    Compares the breakout bar's volume to the rolling average volume
    of the past N bars (across all hours of the day).

    This is better than comparing to same-day opening bars only,
    because on real data the opening is always the busiest period —
    so almost every breakout would look "low volume" vs the open.

    Ratio = breakout volume / rolling average volume

    ≥ 2.0x average → 20 pts  (very high conviction)
    ≥ 1.5x average → 15 pts
    ≥ 1.0x average → 10 pts  (at or above average)
    ≥ 0.7x average →  5 pts
    < 0.7x average →  0 pts  (suspiciously quiet breakout)
    """
    breakout_bar = prices[prices.index == signal_time]
    if breakout_bar.empty:
        return 10

    vol = breakout_bar["Volume"].iloc[0]

    bars_before = prices[prices.index < signal_time]
    if bars_before.empty or bars_before["Avg_Volume"].isna().all():
        return 10

    avg_vol = bars_before["Avg_Volume"].dropna().iloc[-1]
    if avg_vol == 0 or pd.isna(avg_vol):
        return 10

    ratio = vol / avg_vol

    if   ratio >= 2.0: return 20
    elif ratio >= 1.5: return 15
    elif ratio >= 1.0: return 10
    elif ratio >= 0.7: return  5
    else:              return  0


def score_time_of_day(signal_time):
    """
    FACTOR 5: Time of day (20 pts)

    Earlier breakouts are stronger. The market's main directional energy
    is in the morning session. Afternoon breakouts are often low-energy
    drifts that reverse easily.

    Before 11:00am ET → 20 pts
    11:00 – 12:00pm   → 14 pts
    12:00 –  1:00pm   →  8 pts
    1:00  –  2:00pm   →  4 pts
    After  2:00pm     →  0 pts
    """
    hour = signal_time.hour + signal_time.minute / 60

    if   hour < 11: return 20
    elif hour < 12: return 14
    elif hour < 13: return  8
    elif hour < 14: return  4
    else:           return  0

# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_all_signals(signals, prices):
    scored_rows = []

    for _, row in signals.iterrows():
        signal_time = pd.Timestamp(row["Signal_Time"])
        direction   = row["Signal"]
        range_size  = row["Range_Size_Pts"]

        s_trend    = score_trend_alignment(signal_time, direction, prices)
        s_atr      = score_range_vs_atr(range_size, signal_time, prices)
        s_strength = score_breakout_strength(row, prices)
        s_volume   = score_volume_conviction(signal_time, prices)
        s_time     = score_time_of_day(signal_time)

        total = s_trend + s_atr + s_strength + s_volume + s_time

        scored_rows.append({
            **row.to_dict(),
            "Score_Trend":     s_trend,
            "Score_RangeATR":  s_atr,
            "Score_Strength":  s_strength,
            "Score_Volume":    s_volume,
            "Score_TimeOfDay": s_time,
            "Total_Score":     total,
            "Grade":           grade(total),
            "Take_Trade":      "YES" if total >= HIGH_QUALITY_THRESHOLD else "no",
        })

    df = pd.DataFrame(scored_rows)
    df = df.sort_values("Total_Score", ascending=False)
    return df


def grade(score):
    if   score >= 80: return "A"
    elif score >= 65: return "B"
    elif score >= 50: return "C"
    elif score >= 35: return "D"
    else:             return "F"

# ---------------------------------------------------------------------------
# Display summary
# ---------------------------------------------------------------------------

def print_summary(df):
    total      = len(df)
    take_count = (df["Take_Trade"] == "YES").sum()
    skip_count = total - take_count

    print("\n" + "=" * 58)
    print("  IMPROVED TRADE SCORES — Opening Range Breakout")
    print("=" * 58)
    print(f"  Total signals scored : {total}")
    print(f"  High quality (≥{HIGH_QUALITY_THRESHOLD})  : {take_count}  → would take these")
    print(f"  Low quality  (<{HIGH_QUALITY_THRESHOLD})  : {skip_count}  → would skip these")
    print()

    grade_counts = df["Grade"].value_counts().sort_index()
    print("  Grade breakdown:")
    for g, count in grade_counts.items():
        bar = "█" * count
        print(f"    {g}  {bar}  ({count})")

    print()
    print("  Top 10 highest-scoring setups:")
    cols = ["Date", "Signal", "Score_Trend", "Score_RangeATR",
            "Score_Strength", "Score_Volume", "Score_TimeOfDay",
            "Total_Score", "Grade"]
    print(df[cols].head(10).to_string(index=False))

    print()
    best = df.iloc[0]
    print(f"  Breakdown of best setup ({best['Date']}, {best['Signal']}):")
    print(f"    1. Trend alignment    : {best['Score_Trend']}/20")
    print(f"    2. Range vs ATR       : {best['Score_RangeATR']}/20")
    print(f"    3. Breakout strength  : {best['Score_Strength']}/20")
    print(f"    4. Volume conviction  : {best['Score_Volume']}/20")
    print(f"    5. Time of day        : {best['Score_TimeOfDay']}/20")
    print(f"    ─────────────────────────────")
    print(f"    TOTAL                 : {best['Total_Score']}/100  (Grade {best['Grade']})")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    signals, prices = load_files()
    scored = score_all_signals(signals, prices)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path  = os.path.join(script_dir, RESULTS_FILE)
    scored.to_csv(save_path, index=False)
    print(f"Scored signals saved to: {save_path}")

    print_summary(scored)
