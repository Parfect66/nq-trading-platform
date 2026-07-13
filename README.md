# NQ Trading Platform

A Python research platform for analysing NQ futures (Nasdaq 100) trading strategies.

Built for beginners — every file is commented in plain English.

## What it does

1. **Downloads** real NQ price data from Yahoo Finance (free, no account needed)
2. **Detects** Opening Range Breakout signals each trading day
3. **Backtests** those signals with configurable stop loss and profit target
4. **Scores** each signal A–F across five quality factors
5. **Reports** whether high-scoring signals outperform low-scoring ones

## Folder structure

```
data/           Raw price data + download script
strategies/     Signal detection + trade scoring
backtests/      Backtest engine + analysis report
notebooks/      Reserved for interactive Jupyter analysis
```

## How to run

**1. Install dependencies**
```bash
pip install yfinance pandas matplotlib
```

**2. Download real NQ data**
```bash
python3 data/download_nq.py
```

**3. Run the full pipeline**
```bash
python3 strategies/orb.py
python3 backtests/backtest_orb.py
python3 strategies/score_trades.py
python3 backtests/analyse_results.py
```

**4. Open the visual report**

Open `backtests/orb_report.html` in your browser.

## Strategy: Opening Range Breakout (ORB)

The first 30 minutes of the US session (9:30–10:00am ET) creates a high and a low.
If price breaks above that range later in the day → BUY signal.
If price breaks below → SELL signal.

## Trade scoring factors (20 pts each, 100 max)

| Factor | What it measures |
|---|---|
| Trend alignment | Breakout direction vs 10-bar moving average |
| Range vs ATR | Opening range tightness relative to recent volatility |
| Breakout strength | How far price closed beyond the range boundary |
| Volume conviction | Breakout bar volume vs 20-bar rolling average |
| Time of day | Earlier breakouts score higher |

Grades: A (80–100), B (65–79), C (50–64), D (35–49), F (0–34)

## Settings to experiment with

In `backtests/backtest_orb.py`:
```python
STOP_LOSS_POINTS   = 30   # try: 20, 40, 50
PROFIT_TARGET_PTS  = 60   # try: 45, 90, 120
HOLD_HOURS         = 4    # try: 2, 6
```

In `strategies/score_trades.py`:
```python
HIGH_QUALITY_THRESHOLD = 60   # minimum score to flag as "take this trade"
```
