"""
Opening Range Breakout (ORB) Strategy Detector
===============================================
Reads NQ price data and finds every day where price broke out of
the first 30 minutes of trading.

What this file does, section by section:
  1. Load the price data from a CSV file
  2. For each trading day, find the opening range (9:30–10:00am ET)
  3. Scan the rest of that day for the first breakout above or below that range
  4. Record every signal in a results table and save it to a file

How to run:
    python3 orb.py
"""

import pandas as pd
import os

# ---------------------------------------------------------------------------
# Settings — the only things you need to change to experiment
# ---------------------------------------------------------------------------

# Path to your price data file (relative to this script's location)
DATA_FILE = "../data/nq_sample.csv"

# The opening range window (minutes after market open)
# 30 = use 9:30–10:00am as the range
OPENING_RANGE_MINUTES = 30

# Market open time (US Eastern)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30

# Where to save the results
OUTPUT_FILE = "../backtests/orb_signals.csv"

# ---------------------------------------------------------------------------
# Step 1 — Load the price data
# ---------------------------------------------------------------------------

def load_data(filepath):
    """
    Reads the CSV file into a pandas DataFrame.

    A DataFrame is like a spreadsheet in Python — rows and columns,
    where each row is one price bar (one hour in our sample data).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filepath)

    print(f"Loading data from: {full_path}")
    df = pd.read_csv(full_path, index_col="Datetime", parse_dates=True)

    # Tell pandas our timestamps are in Eastern Time so it handles
    # daylight saving correctly. "UTC" is used here because the sample
    # data has no timezone — real Yahoo data needs "America/New_York".
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York", ambiguous="infer",
                                         nonexistent="shift_forward")

    print(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    return df

# ---------------------------------------------------------------------------
# Step 2 & 3 — Find the opening range and detect breakouts
# ---------------------------------------------------------------------------

def find_orb_signals(df, opening_range_minutes, open_hour, open_minute):
    """
    The main logic. For each trading day:
      a) Collect all bars inside the opening range window
      b) Record the highest High and lowest Low of those bars (the range)
      c) Look at every bar AFTER the window closes
      d) The first bar that closes above the range high  → BUY signal
         The first bar that closes below the range low   → SELL signal
         If neither happens before close                 → No signal

    Returns a list of signal dictionaries, one per trading day.
    """
    signals = []

    # "groupby" splits the data into separate groups — one group per calendar day.
    # date() extracts just the date part from a full timestamp (drops the time).
    for date, day_data in df.groupby(df.index.date):

        # --- Find the opening range bars ---
        # We want bars that fall between 9:30am and 9:30am + 30 minutes
        market_open = pd.Timestamp(date).tz_localize("America/New_York").replace(
            hour=open_hour, minute=open_minute, second=0
        )
        range_end = market_open + pd.Timedelta(minutes=opening_range_minutes)

        # Boolean mask: True for bars inside the opening range window
        in_range = (day_data.index >= market_open) & (day_data.index < range_end)
        range_bars = day_data[in_range]

        # Need at least one bar in the opening range to proceed
        if range_bars.empty:
            continue

        # The range is simply the highest high and lowest low in that window
        range_high = range_bars["High"].max()
        range_low  = range_bars["Low"].min()
        range_size = round(range_high - range_low, 2)

        # --- Scan bars after the opening range for a breakout ---
        after_range = day_data[day_data.index >= range_end]

        signal_time      = None
        signal_direction = None   # "BUY" or "SELL"
        signal_price     = None

        for timestamp, bar in after_range.iterrows():
            # A bar's Close breaking above the range high = BUY
            if bar["Close"] > range_high:
                signal_time      = timestamp
                signal_direction = "BUY"
                signal_price     = round(bar["Close"], 2)
                break  # stop at the FIRST breakout — we only take one trade per day

            # A bar's Close breaking below the range low = SELL
            elif bar["Close"] < range_low:
                signal_time      = timestamp
                signal_direction = "SELL"
                signal_price     = round(bar["Close"], 2)
                break

        # Record this day's result (even if no signal fired)
        signals.append({
            "Date":            str(date),
            "Range_High":      round(range_high, 2),
            "Range_Low":       round(range_low, 2),
            "Range_Size_Pts":  range_size,
            "Signal":          signal_direction if signal_direction else "NONE",
            "Signal_Time":     str(signal_time) if signal_time else "",
            "Signal_Price":    signal_price if signal_price else "",
        })

    return signals

# ---------------------------------------------------------------------------
# Step 4 — Display and save the results
# ---------------------------------------------------------------------------

def save_and_display(signals, output_file):
    """
    Turns the list of signal dictionaries into a DataFrame,
    prints a summary to the screen, and saves a CSV file.
    """
    results = pd.DataFrame(signals)

    if results.empty:
        print("No signals found. Check your data covers market hours (9:30am ET).")
        return

    # Count how many BUY, SELL, and NONE days there were
    counts = results["Signal"].value_counts()
    total_days = len(results)
    signal_days = total_days - counts.get("NONE", 0)

    print("\n" + "=" * 50)
    print("OPENING RANGE BREAKOUT — RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Trading days analysed : {total_days}")
    print(f"  Days with a signal    : {signal_days}")
    print(f"  BUY signals           : {counts.get('BUY',  0)}")
    print(f"  SELL signals          : {counts.get('SELL', 0)}")
    print(f"  No breakout days      : {counts.get('NONE', 0)}")
    print(f"  Signal rate           : {round(signal_days / total_days * 100, 1)}% of days")
    print()
    print("First 10 signals:")
    print(results[results["Signal"] != "NONE"].head(10).to_string(index=False))

    # Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, output_file)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results.to_csv(save_path, index=False)
    print(f"\nFull results saved to: {save_path}")


# ---------------------------------------------------------------------------
# Entry point — this block runs when you do: python3 orb.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df      = load_data(DATA_FILE)
    signals = find_orb_signals(df, OPENING_RANGE_MINUTES, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    save_and_display(signals, OUTPUT_FILE)
