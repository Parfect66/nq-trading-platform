"""
Generates realistic-looking NQ futures sample data for development and testing.
Use this when you don't have internet access or want reproducible test data.

Run with:  python3 generate_sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

BARS = 500          # How many 1-hour bars to generate
START_PRICE = 21500  # Approximate NQ price to start from
OUTPUT_FILE = "nq_sample.csv"

np.random.seed(42)  # Makes the random numbers reproducible (same result every run)

def generate_ohlcv(n_bars, start_price):
    prices = [start_price]

    # Simulate price as a random walk (small random steps each bar)
    for _ in range(n_bars - 1):
        change = np.random.normal(loc=0.5, scale=30)  # slight upward drift, ±30 pts
        prices.append(max(prices[-1] + change, 1000))  # floor at 1000 to avoid negatives

    rows = []
    # Build hourly timestamps, skipping weekends and outside trading hours (8am–5pm ET)
    # Start at 9:30am ET on a weekday — the exact market open for ORB
    current = datetime(2026, 5, 1, 9, 30)
    for i in range(n_bars):
        # Skip weekends
        while current.weekday() >= 5:
            current += timedelta(hours=1)
        # Skip outside 9:30am–4pm ET window (regular US session)
        while (current.hour < 9 or (current.hour == 9 and current.minute < 30)) \
               or current.hour >= 16:
            current += timedelta(hours=1)
            while current.weekday() >= 5:
                current += timedelta(hours=1)

        close = prices[i]
        spread = abs(np.random.normal(0, 15))  # range of the bar
        high = close + spread
        low = close - spread
        open_ = low + np.random.uniform(0, spread * 2)
        volume = int(np.random.uniform(5000, 50000))

        rows.append({
            "Datetime": current,
            "Open": round(open_, 2),
            "High": round(high, 2),
            "Low": round(low, 2),
            "Close": round(close, 2),
            "Volume": volume,
        })
        current += timedelta(hours=1)

    return pd.DataFrame(rows).set_index("Datetime")


df = generate_ohlcv(BARS, START_PRICE)
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
df.to_csv(save_path)
print(f"Generated {len(df)} bars of sample NQ data → {save_path}")
print(df.head().to_string())
