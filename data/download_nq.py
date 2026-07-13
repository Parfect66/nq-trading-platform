"""
Download NQ futures data from Yahoo Finance and save it as a CSV file.

What this script does, step by step:
1. Connects to Yahoo Finance (free, no account needed)
2. Downloads price bars for NQ futures
3. Saves the data to a file so we can analyse it offline

How to run this script:
    python3 download_nq.py

After running, look for a file called nq_data.csv in this folder.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# ---------------------------------------------------------------------------
# Settings — change these to adjust what data you download
# ---------------------------------------------------------------------------

TICKER = "NQ=F"          # NQ continuous futures on Yahoo Finance
INTERVAL = "1h"          # Bar size: "1m"=1 minute, "5m"=5 min, "1h"=1 hour, "1d"=daily
DAYS_BACK = 60           # How many days of history to fetch
OUTPUT_FILE = "nq_data.csv"  # Where to save the data (same folder as this script)

# Note on interval limits (Yahoo Finance free tier):
#   1m  → max 7 days back
#   5m  → max 60 days back
#   1h  → max 730 days back
#   1d  → unlimited history

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_nq_data(ticker, interval, days_back, output_file):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_back)

    print(f"Downloading {ticker} | interval={interval} | {days_back} days of history...")
    print(f"Date range: {start_date.date()} → {end_date.date()}")

    # yf.download fetches the data. auto_adjust=True corrects prices for splits.
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        print("ERROR: No data returned. The ticker may be wrong or the market is closed.")
        return

    # Flatten multi-level column headers that yfinance sometimes produces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the columns we care about
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Datetime"

    # Save to CSV (a plain spreadsheet you can open in Excel)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, output_file)
    df.to_csv(save_path)

    print(f"\nSuccess! Downloaded {len(df)} bars.")
    print(f"Saved to: {save_path}")
    print(f"\nFirst 5 rows of data:")
    print(df.head().to_string())
    print(f"\nLast 5 rows of data:")
    print(df.tail().to_string())


if __name__ == "__main__":
    download_nq_data(TICKER, INTERVAL, DAYS_BACK, OUTPUT_FILE)
