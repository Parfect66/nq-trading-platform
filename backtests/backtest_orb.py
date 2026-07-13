"""
Backtest: Opening Range Breakout
=================================
Takes the signals detected by strategies/orb.py and measures
whether each one actually made money.

For every signal we simulate:
  - Entry  : buy/sell at the signal price
  - Exit 1 : hit a STOP LOSS  (we give up and close the trade at a loss)
  - Exit 2 : hit a PROFIT TARGET (we close the trade with a gain)
  - Exit 3 : neither hit → close at end of day at whatever price we get

Key concept — NQ point value:
  Each 1-point move in NQ = $20 per contract (the standard "micro" NQ = $2/pt).
  We'll use $20/pt (1 full contract) in the P&L calculation.
  You can change NQ_DOLLARS_PER_POINT below to match your actual contract size.

How to run:
    python3 backtest_orb.py
"""

import pandas as pd
import os

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SIGNALS_FILE  = "orb_signals.csv"          # output from strategies/orb.py
PRICE_DATA    = "../data/nq_data.csv"      # the raw bar data
RESULTS_FILE  = "orb_backtest_results.csv" # where we save the full trade log

# Risk parameters — the most important numbers to experiment with
STOP_LOSS_POINTS   = 30   # close the trade if it moves this many points against us
PROFIT_TARGET_PTS  = 60   # close the trade when we're up this many points
HOLD_HOURS         = 4    # maximum hours to stay in the trade if neither level hit

# Contract size
NQ_DOLLARS_PER_POINT = 20  # $20 per point for 1 full NQ contract (use $2 for Micro NQ)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    signals_path = os.path.join(script_dir, SIGNALS_FILE)
    price_path   = os.path.join(script_dir, PRICE_DATA)

    signals = pd.read_csv(signals_path)
    # Filter out days where no signal fired
    signals = signals[signals["Signal"] != "NONE"].copy()

    prices = pd.read_csv(price_path, index_col="Datetime", parse_dates=True)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("America/New_York",
                                                  ambiguous="infer",
                                                  nonexistent="shift_forward")

    print(f"Loaded {len(signals)} signals and {len(prices)} price bars.")
    return signals, prices

# ---------------------------------------------------------------------------
# Simulate each trade
# ---------------------------------------------------------------------------

def simulate_trade(signal_row, prices, stop_pts, target_pts, hold_hours):
    """
    Given one signal row, walk through subsequent bars and work out
    what would have happened if we entered that trade.

    Returns a dictionary with the trade outcome.
    """
    direction    = signal_row["Signal"]        # "BUY" or "SELL"
    entry_price  = float(signal_row["Signal_Price"])
    signal_time  = pd.Timestamp(signal_row["Signal_Time"])
    trade_date   = signal_row["Date"]

    # The latest we will stay in the trade
    exit_deadline = signal_time + pd.Timedelta(hours=hold_hours)

    # For a BUY:  profit if price goes UP,  stop if price goes DOWN
    # For a SELL: profit if price goes DOWN, stop if price goes UP
    if direction == "BUY":
        stop_level   = entry_price - stop_pts
        target_level = entry_price + target_pts
    else:
        stop_level   = entry_price + stop_pts
        target_level = entry_price - target_pts

    # Walk through every bar after entry
    bars_after_entry = prices[prices.index > signal_time]

    exit_price  = None
    exit_reason = None
    exit_time   = None

    for ts, bar in bars_after_entry.iterrows():
        # Don't go past our time limit
        if ts > exit_deadline:
            exit_price  = bar["Open"]   # exit at the open of the first bar past deadline
            exit_reason = "TIME"
            exit_time   = ts
            break

        # Check stop loss first (worst case scenario)
        if direction == "BUY"  and bar["Low"]  <= stop_level:
            exit_price  = stop_level
            exit_reason = "STOP"
            exit_time   = ts
            break

        if direction == "SELL" and bar["High"] >= stop_level:
            exit_price  = stop_level
            exit_reason = "STOP"
            exit_time   = ts
            break

        # Check profit target
        if direction == "BUY"  and bar["High"] >= target_level:
            exit_price  = target_level
            exit_reason = "TARGET"
            exit_time   = ts
            break

        if direction == "SELL" and bar["Low"]  <= target_level:
            exit_price  = target_level
            exit_reason = "TARGET"
            exit_time   = ts
            break

    # If we ran out of bars (end of data) without an exit
    if exit_price is None:
        last_bar    = bars_after_entry.iloc[-1] if not bars_after_entry.empty else None
        exit_price  = last_bar["Close"] if last_bar is not None else entry_price
        exit_reason = "END_OF_DATA"
        exit_time   = bars_after_entry.index[-1] if not bars_after_entry.empty else signal_time

    # Calculate profit/loss in points and dollars
    if direction == "BUY":
        points_pnl = exit_price - entry_price
    else:
        points_pnl = entry_price - exit_price

    dollar_pnl = points_pnl * NQ_DOLLARS_PER_POINT

    return {
        "Date":          trade_date,
        "Direction":     direction,
        "Entry_Price":   round(entry_price, 2),
        "Entry_Time":    str(signal_time),
        "Stop_Level":    round(stop_level, 2),
        "Target_Level":  round(target_level, 2),
        "Exit_Price":    round(exit_price, 2),
        "Exit_Time":     str(exit_time),
        "Exit_Reason":   exit_reason,
        "Points_PnL":    round(points_pnl, 2),
        "Dollar_PnL":    round(dollar_pnl, 2),
        "Win":           points_pnl > 0,
    }

# ---------------------------------------------------------------------------
# Summarise results
# ---------------------------------------------------------------------------

def print_summary(results_df):
    total_trades  = len(results_df)
    wins          = results_df["Win"].sum()
    losses        = total_trades - wins
    win_rate      = round(wins / total_trades * 100, 1)

    total_pnl     = results_df["Dollar_PnL"].sum()
    avg_win       = results_df.loc[results_df["Win"],  "Dollar_PnL"].mean()
    avg_loss      = results_df.loc[~results_df["Win"], "Dollar_PnL"].mean()

    # Profit factor = total dollars won / total dollars lost (ignoring sign)
    gross_profit  = results_df.loc[results_df["Dollar_PnL"] > 0, "Dollar_PnL"].sum()
    gross_loss    = abs(results_df.loc[results_df["Dollar_PnL"] < 0, "Dollar_PnL"].sum())
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Running cumulative P&L (to spot drawdowns)
    results_df = results_df.copy()
    results_df["Cumulative_PnL"] = results_df["Dollar_PnL"].cumsum()
    max_drawdown = (results_df["Cumulative_PnL"].cummax() - results_df["Cumulative_PnL"]).max()

    exits = results_df["Exit_Reason"].value_counts()

    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS — Opening Range Breakout (NQ)")
    print("=" * 55)
    print(f"  Strategy settings:")
    print(f"    Stop loss      : {STOP_LOSS_POINTS} pts  (${STOP_LOSS_POINTS * NQ_DOLLARS_PER_POINT:,.0f} per trade)")
    print(f"    Profit target  : {PROFIT_TARGET_PTS} pts  (${PROFIT_TARGET_PTS * NQ_DOLLARS_PER_POINT:,.0f} per trade)")
    print(f"    Max hold time  : {HOLD_HOURS} hours")
    print()
    print(f"  Performance:")
    print(f"    Total trades   : {total_trades}")
    print(f"    Wins / Losses  : {wins} / {losses}")
    print(f"    Win rate       : {win_rate}%")
    print(f"    Avg win        : ${avg_win:,.0f}")
    print(f"    Avg loss       : ${avg_loss:,.0f}")
    print(f"    Profit factor  : {profit_factor}  (>1.0 = overall profitable)")
    print()
    print(f"  Overall P&L:")
    print(f"    Total profit   : ${total_pnl:,.0f}")
    print(f"    Max drawdown   : ${max_drawdown:,.0f}  (biggest losing streak)")
    print()
    print(f"  How trades exited:")
    for reason, count in exits.items():
        print(f"    {reason:<15}: {count} trades")

    print()
    print("  Last 10 trades:")
    display_cols = ["Date","Direction","Entry_Price","Exit_Price",
                    "Exit_Reason","Points_PnL","Dollar_PnL"]
    print(results_df[display_cols].tail(10).to_string(index=False))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    signals, prices = load_files()

    print(f"\nSimulating trades: stop={STOP_LOSS_POINTS}pts, target={PROFIT_TARGET_PTS}pts, hold={HOLD_HOURS}h ...")

    trades = []
    for _, row in signals.iterrows():
        trade = simulate_trade(row, prices,
                               STOP_LOSS_POINTS, PROFIT_TARGET_PTS, HOLD_HOURS)
        trades.append(trade)

    results = pd.DataFrame(trades)

    # Save full trade log
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path  = os.path.join(script_dir, RESULTS_FILE)
    results.to_csv(save_path, index=False)
    print(f"Full trade log saved to: {save_path}")

    print_summary(results)
