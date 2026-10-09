"""
Stock Analysis & Backtesting Dashboard
--------------------------------------
A Streamlit application for:
- Moving Average (SMA) crossover backtesting
- Exponential Moving Average (EMA) crossover backtesting
- MACD crossover backtesting
- Stochastic RSI backtesting
- Fundamental analysis using Yahoo Finance
- News lookup using Yahoo Finance / optional NewsAPI
- Optional LSTM stock-price prediction
- Wikipedia company information

Run:
    streamlit run c.py

Optional environment / Streamlit secrets:
    NEWS_API_KEY = "your_newsapi_key"

Optional ML model:
    my_model.h5

The backtesting engine is intentionally kept separate from Streamlit UI code.
It uses a long-only portfolio model: Buy = 100% invested, Sell = 100% in cash.
Transaction costs and slippage are applied to actual trade turnover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_TITLE = "Stock Analysis & Trading Backtesting Dashboard"
DEFAULT_TICKER = "TSLA"
DEFAULT_START_DATE = "2021-01-01"
DEFAULT_INITIAL_CAPITAL = 10_000.0
DEFAULT_TRANSACTION_COST = 0.01       # 1%
DEFAULT_SLIPPAGE = 0.005              # 0.5%


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    data: pd.DataFrame
    final_equity: float
    total_return: float
    buy_and_hold_return: float
    total_transaction_costs: float
    total_slippage: float
    total_trading_costs: float
    number_of_trades: int
    winning_trades: int
    losing_trades: int
    win_loss_ratio: float
    profit_factor: float
    average_trade_return: float
    max_drawdown: float
    sharpe_ratio: float
    current_signal: str


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio without raising division-by-zero errors."""
    if denominator == 0:
        return float("inf") if numerator > 0 else 0.0
    return numerator / denominator


def format_metric(value: float, percent: bool = False) -> str:
    """Format a metric for Streamlit."""
    if not np.isfinite(value):
        return "∞"
    return f"{value:.2%}" if percent else f"{value:,.2f}"


def validate_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean historical OHLCV data."""
    if df is None or df.empty:
        raise ValueError("No historical price data was returned.")

    data = df.copy()

    # yfinance may return MultiIndex columns for some requests.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = data[list(required)].copy()
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.dropna(subset=["Close"])

    if len(data) < 2:
        raise ValueError("At least two historical observations are required.")

    return data


@st.cache_data(ttl=900, show_spinner=False)
def download_history(ticker: str, start_date: str) -> pd.DataFrame:
    """Download historical data from Yahoo Finance."""
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    data = yf.download(
        ticker,
        start=start_date,
        end=pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )

    return validate_price_data(data)


@st.cache_data(ttl=900, show_spinner=False)
def get_ticker_info(ticker: str) -> dict:
    """Retrieve Yahoo Finance company information."""
    return yf.Ticker(ticker).info


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def add_sma(data: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Add simple moving averages."""
    result = data.copy()
    result["Short_MA"] = result["Close"].rolling(short_window).mean()
    result["Long_MA"] = result["Close"].rolling(long_window).mean()
    return result


def add_ema(data: pd.DataFrame, short_span: int, long_span: int) -> pd.DataFrame:
    """Add exponential moving averages."""
    result = data.copy()
    result["Short_EMA"] = result["Close"].ewm(span=short_span, adjust=False).mean()
    result["Long_EMA"] = result["Close"].ewm(span=long_span, adjust=False).mean()
    return result


def add_macd(
    data: pd.DataFrame,
    short_span: int = 12,
    long_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """Calculate MACD and its signal line."""
    result = data.copy()
    short_ema = result["Close"].ewm(span=short_span, adjust=False).mean()
    long_ema = result["Close"].ewm(span=long_span, adjust=False).mean()

    result["MACD"] = short_ema - long_ema
    result["Signal_Line"] = result["MACD"].ewm(
        span=signal_span, adjust=False
    ).mean()
    result["MACD_Histogram"] = result["MACD"] - result["Signal_Line"]
    return result


def add_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate RSI using Wilder-style exponential smoothing."""
    result = data.copy()

    delta = result["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = average_gain / average_loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + rs))
    result["RSI"] = result["RSI"].fillna(50)

    return result


def add_stochastic_rsi(
    data: pd.DataFrame,
    rsi_window: int = 14,
    stoch_window: int = 14,
) -> pd.DataFrame:
    """Calculate Stochastic RSI in the 0-1 range."""
    result = add_rsi(data, rsi_window)

    rolling_min = result["RSI"].rolling(stoch_window).min()
    rolling_max = result["RSI"].rolling(stoch_window).max()

    denominator = (rolling_max - rolling_min).replace(0, np.nan)
    result["Stoch_RSI"] = (
        (result["RSI"] - rolling_min) / denominator
    ).clip(0, 1)

    return result


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def crossover_signals(
    indicator: pd.Series,
    reference: pd.Series,
) -> pd.Series:
    """
    Return portfolio signals:
    1 = long
    0 = cash

    A long position starts after a bullish crossover and ends after a
    bearish crossover.
    """
    bullish = (indicator > reference) & (
        indicator.shift(1) <= reference.shift(1)
    )
    bearish = (indicator < reference) & (
        indicator.shift(1) >= reference.shift(1)
    )

    signal = pd.Series(np.nan, index=indicator.index)
    signal.loc[bullish] = 1
    signal.loc[bearish] = 0

    return signal.ffill().fillna(0).astype(int)


def stochastic_rsi_signals(
    stoch_rsi: pd.Series,
    oversold: float,
    overbought: float,
) -> pd.Series:
    """
    Long-only Stochastic RSI strategy.

    Buy when Stoch RSI crosses upward through the oversold level.
    Sell when Stoch RSI crosses downward through the overbought level.
    """
    buy = (stoch_rsi > oversold) & (
        stoch_rsi.shift(1) <= oversold
    )
    sell = (stoch_rsi < overbought) & (
        stoch_rsi.shift(1) >= overbought
    )

    signal = pd.Series(np.nan, index=stoch_rsi.index)
    signal.loc[buy] = 1
    signal.loc[sell] = 0

    return signal.ffill().fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

def run_backtest(
    data: pd.DataFrame,
    target_position: pd.Series,
    initial_capital: float,
    transaction_cost_rate: float = DEFAULT_TRANSACTION_COST,
    slippage_rate: float = DEFAULT_SLIPPAGE,
) -> BacktestResult:
    """
    Run a long-only portfolio backtest.

    The portfolio is either:
        0% invested / 100% cash
        100% invested / 0% cash

    Trading costs are charged on the dollar value traded.
    Slippage changes the execution price in the unfavorable direction.
    """
    if initial_capital <= 0:
        raise ValueError("Initial capital must be greater than zero.")

    if transaction_cost_rate < 0 or slippage_rate < 0:
        raise ValueError("Transaction cost and slippage cannot be negative.")

    df = data.copy()
    position = target_position.reindex(df.index).fillna(0).astype(int)

    # Shift the strategy one day so today's signal is acted upon at the
    # following session. This avoids look-ahead bias.
    held_position = position.shift(1).fillna(0).astype(int)

    df["Position"] = held_position
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)

    # Position return before trading costs.
    df["Strategy_Return_Before_Costs"] = (
        df["Daily_Return"] * df["Position"]
    )

    # Turnover is 1 for a buy/sell and 0 when holding.
    df["Turnover"] = df["Position"].diff().abs().fillna(
        df["Position"].abs()
    )

    # Dollar portfolio value BEFORE each day's return.
    gross_equity = initial_capital * (
        1 + df["Strategy_Return_Before_Costs"]
    ).cumprod()

    # Cost model based on portfolio value traded.
    df["Transaction_Cost"] = (
        gross_equity.shift(1).fillna(initial_capital)
        * df["Turnover"]
        * transaction_cost_rate
    )

    df["Slippage"] = (
        gross_equity.shift(1).fillna(initial_capital)
        * df["Turnover"]
        * slippage_rate
    )

    df["Trading_Cost"] = df["Transaction_Cost"] + df["Slippage"]

    # Convert costs to portfolio returns.
    equity_before_cost = gross_equity
    df["Equity"] = equity_before_cost - df["Trading_Cost"].cumsum()

    # Ensure equity does not become negative due to extreme cost inputs.
    df["Equity"] = df["Equity"].clip(lower=0)

    df["Strategy_Return"] = df["Equity"].pct_change().replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    df["Cumulative_Strategy_Return"] = (
        df["Equity"] / initial_capital
    ) - 1

    df["Buy_Hold_Equity"] = initial_capital * (
        df["Close"] / df["Close"].iloc[0]
    )
    df["Cumulative_Return"] = (
        df["Buy_Hold_Equity"] / initial_capital
    ) - 1

    # Drawdown.
    rolling_peak = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] / rolling_peak - 1

    # Trade returns are calculated from changes in position.
    trade_entries = df.index[
        (df["Position"] == 1) & (df["Position"].shift(1).fillna(0) == 0)
    ]
    trade_exits = df.index[
        (df["Position"] == 0) & (df["Position"].shift(1).fillna(0) == 1)
    ]

    trade_returns = []
    entry_equity = None

    for date in df.index:
        if date in trade_entries:
            entry_equity = float(df.loc[date, "Equity"])

        if date in trade_exits and entry_equity is not None:
            exit_equity = float(df.loc[date, "Equity"])
            trade_returns.append(
                (exit_equity / entry_equity) - 1
                if entry_equity != 0 else 0
            )
            entry_equity = None

    # If a trade is still open, mark it to market at the end.
    if entry_equity is not None:
        final_equity = float(df["Equity"].iloc[-1])
        trade_returns.append(
            (final_equity / entry_equity) - 1
            if entry_equity != 0 else 0
        )

    wins = [x for x in trade_returns if x > 0]
    losses = [x for x in trade_returns if x < 0]

    winning_trades = len(wins)
    losing_trades = len(losses)

    profit_factor = (
        sum(wins) / abs(sum(losses))
        if losses
        else (float("inf") if wins else 0.0)
    )

    win_loss_ratio = safe_ratio(winning_trades, losing_trades)

    daily_strategy_returns = df["Strategy_Return"]
    std = daily_strategy_returns.std()

    sharpe_ratio = (
        (daily_strategy_returns.mean() / std) * np.sqrt(252)
        if std and np.isfinite(std) and std > 0
        else 0.0
    )

    # Signal description based on the last actual signal transition.
    current_position = int(position.iloc[-1])
    previous_position = int(position.iloc[-2]) if len(position) > 1 else 0

    if current_position == 1 and previous_position == 0:
        current_signal = "BUY"
    elif current_position == 0 and previous_position == 1:
        current_signal = "SELL"
    elif current_position == 1:
        current_signal = "HOLD / LONG"
    else:
        current_signal = "HOLD / CASH"

    total_costs = float(df["Transaction_Cost"].sum())
    total_slippage = float(df["Slippage"].sum())

    return BacktestResult(
        data=df,
        final_equity=float(df["Equity"].iloc[-1]),
        total_return=float(df["Equity"].iloc[-1] / initial_capital - 1),
        buy_and_hold_return=float(
            df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
        ),
        total_transaction_costs=total_costs,
        total_slippage=total_slippage,
        total_trading_costs=total_costs + total_slippage,
        number_of_trades=len(trade_returns),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_loss_ratio=win_loss_ratio,
        profit_factor=profit_factor,
        average_trade_return=(
            float(np.mean(trade_returns)) if trade_returns else 0.0
        ),
        max_drawdown=float(df["Drawdown"].min()),
        sharpe_ratio=float(sharpe_ratio),
        current_signal=current_signal,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def show_price_strategy_chart(
    df: pd.DataFrame,
    title: str,
    indicator_columns: list[Tuple[str, str]],
    buy_mask: Optional[pd.Series] = None,
    sell_mask: Optional[pd.Series] = None,
) -> None:
    """Display price, indicators and optional trade markers."""
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(df.index, df["Close"], label="Close Price", linewidth=1.8)

    for column, label in indicator_columns:
        if column in df.columns:
            ax.plot(df.index, df[column], label=label, linestyle="--")

    if buy_mask is not None and buy_mask.any():
        ax.scatter(
            df.index[buy_mask],
            df.loc[buy_mask, "Close"],
            marker="^",
            s=70,
            label="Buy",
        )

    if sell_mask is not None and sell_mask.any():
        ax.scatter(
            df.index[sell_mask],
            df.loc[sell_mask, "Close"],
            marker="v",
            s=70,
            label="Sell",
        )

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_equity_chart(result: BacktestResult, title: str) -> None:
    """Display strategy equity against buy-and-hold equity."""
    df = result.data

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df.index, df["Equity"], label="Strategy Equity", linewidth=2)
    ax.plot(
        df.index,
        df["Buy_Hold_Equity"],
        label="Buy & Hold",
        linestyle="--",
    )
    ax.set_title(title)
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Backtest UI
# ---------------------------------------------------------------------------

def display_backtest_metrics(result: BacktestResult) -> None:
    """Display common backtest metrics."""
    cols = st.columns(4)

    cols[0].metric(
        "Final Portfolio",
        f"${result.final_equity:,.2f}",
        f"{result.total_return:.2%}",
    )
    cols[1].metric(
        "Buy & Hold",
        f"{result.buy_and_hold_return:.2%}",
    )
    cols[2].metric(
        "Max Drawdown",
        f"{result.max_drawdown:.2%}",
    )
    cols[3].metric(
        "Sharpe Ratio",
        f"{result.sharpe_ratio:.2f}",
    )

    cols = st.columns(4)
    cols[0].metric("Trades", result.number_of_trades)
    cols[1].metric("Winning Trades", result.winning_trades)
    cols[2].metric("Losing Trades", result.losing_trades)
    cols[3].metric(
        "Win/Loss Ratio",
        format_metric(result.win_loss_ratio),
    )

    cols = st.columns(3)
    cols[0].metric(
        "Profit Factor",
        format_metric(result.profit_factor),
    )
    cols[1].metric(
        "Avg. Trade Return",
        f"{result.average_trade_return:.2%}",
    )
    cols[2].metric(
        "Trading Costs",
        f"${result.total_trading_costs:,.2f}",
    )

    st.write("### Trading Costs")
    cost_df = pd.DataFrame(
        {
            "Cost": [
                "Transaction Costs",
                "Slippage",
                "Total Trading Costs",
            ],
            "Amount": [
                result.total_transaction_costs,
                result.total_slippage,
                result.total_trading_costs,
            ],
        }
    )
    st.dataframe(
        cost_df.style.format({"Amount": "${:,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    if result.current_signal.startswith("BUY"):
        st.success("Current strategy signal: BUY")
    elif result.current_signal.startswith("SELL"):
        st.warning("Current strategy signal: SELL")
    elif result.current_signal == "HOLD / LONG":
        st.info("Current strategy state: HOLD / LONG")
    else:
        st.info("Current strategy state: HOLD / CASH")


def run_strategy_backtest(
    data: pd.DataFrame,
    strategy_name: str,
    indicator_data: pd.DataFrame,
    target_position: pd.Series,
    initial_capital: float,
    transaction_cost: float,
    slippage: float,
    indicator_columns: list[Tuple[str, str]],
    buy_mask: Optional[pd.Series] = None,
    sell_mask: Optional[pd.Series] = None,
) -> None:
    """Run and render a strategy backtest."""
    result = run_backtest(
        indicator_data,
        target_position,
        initial_capital,
        transaction_cost,
        slippage,
    )

    st.subheader(f"{strategy_name} Results")

    show_price_strategy_chart(
        indicator_data,
        f"{strategy_name} — Price & Signals",
        indicator_columns,
        buy_mask,
        sell_mask,
    )

    show_equity_chart(
        result,
        f"{strategy_name} — Portfolio Performance",
    )

    display_backtest_metrics(result)

    with st.expander("View Backtest Data"):
        st.dataframe(
            result.data.tail(500),
            use_container_width=True,
        )


def sma_dashboard(
    data: pd.DataFrame,
    initial_capital: float,
    transaction_cost: float,
    slippage: float,
) -> None:
    st.subheader("Simple Moving Average Crossover")

    col1, col2 = st.columns(2)
    short_window = col1.number_input(
        "Short SMA Window",
        min_value=2,
        value=12,
        step=1,
    )
    long_window = col2.number_input(
        "Long SMA Window",
        min_value=3,
        value=26,
        step=1,
    )

    if short_window >= long_window:
        st.error("Short SMA window must be smaller than long SMA window.")
        return

    df = add_sma(data, short_window, long_window)
    position = crossover_signals(df["Short_MA"], df["Long_MA"])

    buy = position.eq(1) & position.shift(1).fillna(0).eq(0)
    sell = position.eq(0) & position.shift(1).fillna(0).eq(1)

    run_strategy_backtest(
        data,
        "SMA Crossover",
        df,
        position,
        initial_capital,
        transaction_cost,
        slippage,
        [
            ("Short_MA", f"{short_window}-day SMA"),
            ("Long_MA", f"{long_window}-day SMA"),
        ],
        buy,
        sell,
    )


def ema_dashboard(
    data: pd.DataFrame,
    initial_capital: float,
    transaction_cost: float,
    slippage: float,
) -> None:
    st.subheader("Exponential Moving Average Crossover")

    col1, col2 = st.columns(2)
    short_span = col1.number_input(
        "Short EMA Span",
        min_value=2,
        value=12,
        step=1,
    )
    long_span = col2.number_input(
        "Long EMA Span",
        min_value=3,
        value=26,
        step=1,
    )

    if short_span >= long_span:
        st.error("Short EMA span must be smaller than long EMA span.")
        return

    df = add_ema(data, short_span, long_span)
    position = crossover_signals(df["Short_EMA"], df["Long_EMA"])

    buy = position.eq(1) & position.shift(1).fillna(0).eq(0)
    sell = position.eq(0) & position.shift(1).fillna(0).eq(1)

    run_strategy_backtest(
        data,
        "EMA Crossover",
        df,
        position,
        initial_capital,
        transaction_cost,
        slippage,
        [
            ("Short_EMA", f"{short_span}-day EMA"),
            ("Long_EMA", f"{long_span}-day EMA"),
        ],
        buy,
        sell,
    )


def macd_dashboard(
    data: pd.DataFrame,
    initial_capital: float,
    transaction_cost: float,
    slippage: float,
) -> None:
    st.subheader("MACD Crossover")

    col1, col2, col3 = st.columns(3)
    short_span = col1.number_input(
        "MACD Fast EMA",
        min_value=2,
        value=12,
        step=1,
    )
    long_span = col2.number_input(
        "MACD Slow EMA",
        min_value=3,
        value=26,
        step=1,
    )
    signal_span = col3.number_input(
        "MACD Signal",
        min_value=2,
        value=9,
        step=1,
    )

    if short_span >= long_span:
        st.error("Fast EMA must be smaller than slow EMA.")
        return

    df = add_macd(data, short_span, long_span, signal_span)
    position = crossover_signals(df["MACD"], df["Signal_Line"])

    buy = position.eq(1) & position.shift(1).fillna(0).eq(0)
    sell = position.eq(0) & position.shift(1).fillna(0).eq(1)

    run_strategy_backtest(
        data,
        "MACD",
        df,
        position,
        initial_capital,
        transaction_cost,
        slippage,
        [
            ("MACD", f"MACD ({short_span}, {long_span})"),
            ("Signal_Line", f"Signal ({signal_span})"),
        ],
        buy,
        sell,
    )

    st.subheader("MACD Histogram")
    st.bar_chart(df["MACD_Histogram"])


def stochastic_rsi_dashboard(
    data: pd.DataFrame,
    initial_capital: float,
    transaction_cost: float,
    slippage: float,
) -> None:
    st.subheader("Stochastic RSI")

    col1, col2, col3, col4 = st.columns(4)

    rsi_window = col1.number_input(
        "RSI Window",
        min_value=2,
        value=14,
        step=1,
    )
    stoch_window = col2.number_input(
        "Stochastic Window",
        min_value=2,
        value=14,
        step=1,
    )
    oversold = col3.number_input(
        "Oversold",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
    )
    overbought = col4.number_input(
        "Overbought",
        min_value=0.0,
        max_value=1.0,
        value=0.80,
        step=0.05,
    )

    if oversold >= overbought:
        st.error("Oversold must be lower than overbought.")
        return

    df = add_stochastic_rsi(data, rsi_window, stoch_window)

    position = stochastic_rsi_signals(
        df["Stoch_RSI"],
        oversold,
        overbought,
    )

    buy = position.eq(1) & position.shift(1).fillna(0).eq(0)
    sell = position.eq(0) & position.shift(1).fillna(0).eq(1)

    run_strategy_backtest(
        data,
        "Stochastic RSI",
        df,
        position,
        initial_capital,
        transaction_cost,
        slippage,
        [("Stoch_RSI", "Stochastic RSI")],
        buy,
        sell,
    )

    st.subheader("Stochastic RSI Indicator")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df.index, df["Stoch_RSI"], label="Stoch RSI")
    ax.axhline(oversold, linestyle="--", label="Oversold")
    ax.axhline(overbought, linestyle="--", label="Overbought")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fundamental Analysis
# ---------------------------------------------------------------------------

def fundamental_dashboard(ticker: str) -> None:
    st.subheader("Fundamental Analysis")

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception as exc:
        st.error(f"Unable to retrieve company information: {exc}")
        return

    company = info.get("longName") or info.get("shortName") or ticker

    st.markdown(f"### {company} ({ticker})")

    metrics = {
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Country": info.get("country"),
        "City": info.get("city"),
        "Employees": info.get("fullTimeEmployees"),
        "Current Price": info.get("currentPrice"),
        "Market Cap": info.get("marketCap"),
        "Revenue": info.get("totalRevenue"),
        "Net Income": info.get("netIncomeToCommon"),
        "EPS": info.get("trailingEps"),
        "Forward EPS": info.get("forwardEps"),
        "P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Dividend Yield": info.get("dividendYield"),
    }

    display = pd.DataFrame(
        {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
    )

    st.dataframe(display, use_container_width=True, hide_index=True)

    tabs = st.tabs(
        [
            "Income Statement",
            "Balance Sheet",
            "Cash Flow",
            "Holders",
            "Recommendations",
            "Splits",
        ]
    )

    with tabs[0]:
        try:
            st.dataframe(stock.financials, use_container_width=True)
        except Exception as exc:
            st.warning(f"Income statement unavailable: {exc}")

    with tabs[1]:
        try:
            st.dataframe(stock.balance_sheet, use_container_width=True)
            st.write("### Quarterly Balance Sheet")
            st.dataframe(
                stock.quarterly_balance_sheet,
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Balance sheet unavailable: {exc}")

    with tabs[2]:
        try:
            st.dataframe(stock.cashflow, use_container_width=True)
        except Exception as exc:
            st.warning(f"Cash-flow data unavailable: {exc}")

    with tabs[3]:
        try:
            st.write("### Institutional Holders")
            st.dataframe(
                stock.institutional_holders,
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Institutional holder data unavailable: {exc}")

        try:
            st.write("### Major Holders")
            st.dataframe(stock.major_holders, use_container_width=True)
        except Exception as exc:
            st.warning(f"Major holder data unavailable: {exc}")

        try:
            st.write("### Mutual Fund Holders")
            st.dataframe(
                stock.mutualfund_holders,
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Mutual fund holder data unavailable: {exc}")

    with tabs[4]:
        try:
            st.dataframe(stock.recommendations, use_container_width=True)
        except Exception as exc:
            st.warning(f"Recommendation data unavailable: {exc}")

    with tabs[5]:
        try:
            st.dataframe(stock.splits, use_container_width=True)
        except Exception as exc:
            st.warning(f"Split data unavailable: {exc}")


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def news_dashboard(ticker: str, data: pd.DataFrame) -> None:
    st.subheader("Market News")

    try:
        info = get_ticker_info(ticker)
        company = info.get("longName", ticker)

        st.write(
            f"**{company}** | "
            f"Sector: {info.get('sector', 'N/A')} | "
            f"Industry: {info.get('industry', 'N/A')}"
        )

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        cols = st.columns(4)
        cols[0].metric("Last Close", f"${latest['Close']:,.2f}")
        cols[1].metric(
            "Daily Change",
            f"{latest['Close'] - previous['Close']:+,.2f}",
        )
        cols[2].metric("High", f"${data['High'].iloc[-1]:,.2f}")
        cols[3].metric(
            "Volume",
            f"{int(data['Volume'].iloc[-1]):,}",
        )
    except Exception as exc:
        st.warning(f"Company information unavailable: {exc}")

    # Yahoo Finance news does not require exposing an API key.
    try:
        news_items = yf.Ticker(ticker).news or []
    except Exception as exc:
        st.error(f"Unable to retrieve news: {exc}")
        news_items = []

    if not news_items:
        st.info("No Yahoo Finance news articles were returned.")
        return

    for index, article in enumerate(news_items[:20], start=1):
        content = article.get("content", article)

        title = (
            content.get("title")
            or article.get("title")
            or "Untitled article"
        )
        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else article.get("publisher", "Unknown")
        )

        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else article.get("link")
        )

        with st.expander(f"{index}. {title}"):
            st.write(f"**Publisher:** {publisher or 'Unknown'}")

            if link:
                st.link_button("Read Article", link)


# ---------------------------------------------------------------------------
# Stock prediction dashboard
# ---------------------------------------------------------------------------

def stock_prediction_dashboard(
    data: pd.DataFrame,
    ticker: str,
) -> None:
    st.subheader("Stock Price Prediction")

    st.warning(
        "This section is for experimentation only. "
        "A machine-learning prediction is not a guarantee of future prices."
    )

    st.write("### Historical Price")
    st.line_chart(data["Close"])

    st.write("### Historical Volume")
    st.line_chart(data["Volume"])

    ma30 = data["Close"].rolling(30).mean()
    ma100 = data["Close"].rolling(100).mean()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(data.index, data["Close"], label="Close")
    ax.plot(data.index, ma30, label="30-day MA")
    ax.plot(data.index, ma100, label="100-day MA")
    ax.set_title(f"{ticker} — Moving Averages")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    model_file = st.text_input(
        "Model file",
        value="my_model.h5",
        help="Path to a trained Keras model compatible with this data.",
    )

    if not st.button("Run Prediction", type="primary"):
        return

    try:
        from keras.models import load_model
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
    except ImportError as exc:
        st.error(
            "Prediction dependencies are not installed. "
            "Install TensorFlow/Keras and scikit-learn first."
        )
        st.code(str(exc))
        return

    if len(data) < 100:
        st.error("At least 100 historical observations are required.")
        return

    try:
        model = load_model(model_file)
    except Exception as exc:
        st.error(f"Could not load model '{model_file}': {exc}")
        return

    close_values = data[["Close"]].values.astype(float)

    # Train/test split.
    train_len = math.ceil(len(close_values) * 0.8)

    if train_len < 60 or len(close_values) - train_len < 1:
        st.error("Not enough data for a train/test split.")
        return

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(close_values[:train_len])

    # Scale test values using only the training scaler to avoid leakage.
    test_scaled = scaler.transform(close_values[train_len - 60:])

    sequence_length = 60
    x_test = []
    y_test = []

    for i in range(sequence_length, len(test_scaled)):
        x_test.append(test_scaled[i - sequence_length:i, 0])
        y_test.append(test_scaled[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    if len(x_test) == 0:
        st.error("Unable to create test sequences.")
        return

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Some old models were trained on 100 timesteps. Support them explicitly.
    expected_shape = model.input_shape
    if isinstance(expected_shape, list):
        expected_shape = expected_shape[0]

    expected_steps = (
        int(expected_shape[1])
        if len(expected_shape) > 1 and expected_shape[1] is not None
        else sequence_length
    )

    if expected_steps != sequence_length:
        if expected_steps > len(test_scaled):
            st.error(
                f"Model expects {expected_steps} timesteps, "
                f"but only {len(test_scaled)} are available."
            )
            return

        x_test = []
        y_test = []

        for i in range(expected_steps, len(test_scaled)):
            x_test.append(test_scaled[i - expected_steps:i, 0])
            y_test.append(test_scaled[i, 0])

        x_test = np.array(x_test).reshape(-1, expected_steps, 1)
        y_test = np.array(y_test)

    predictions_scaled = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(
        np.asarray(predictions_scaled).reshape(-1, 1)
    ).ravel()

    actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).ravel()

    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mae = float(mean_absolute_error(actual, predictions))
    mse = float(mean_squared_error(actual, predictions))
    r2 = float(r2_score(actual, predictions))

    metrics = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "MSE", "R²"],
            "Value": [rmse, mae, mse, r2],
        }
    )

    st.write("### Model Evaluation")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    prediction_index = data.index[-len(predictions):]

    prediction_df = pd.DataFrame(
        {
            "Actual": actual,
            "Predicted": predictions,
        },
        index=prediction_index,
    )

    st.write("### Actual vs Predicted")
    st.line_chart(prediction_df)

    # Next-day prediction.
    recent_scaled = scaler.transform(
        close_values[-expected_steps:]
    )
    next_input = recent_scaled.reshape(1, expected_steps, 1)

    next_scaled = model.predict(next_input, verbose=0)
    next_price = float(
        scaler.inverse_transform(
            np.asarray(next_scaled).reshape(-1, 1)
        )[0, 0]
    )

    last_close = float(data["Close"].iloc[-1])
    change = (next_price / last_close) - 1 if last_close else 0

    st.write("### Next-Day Prediction")
    st.metric(
        "Predicted Price",
        f"${next_price:,.2f}",
        f"{change:+.2%} vs last close",
    )


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_wikipedia_page(query: str):
    """Find a Wikipedia page."""
    import wikipedia

    try:
        return wikipedia.page(query, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as exc:
        if not exc.options:
            return None
        return wikipedia.page(exc.options[0], auto_suggest=False)
    except wikipedia.exceptions.PageError:
        return None


def wikipedia_dashboard(ticker: str) -> None:
    st.subheader("Wikipedia")

    try:
        info = get_ticker_info(ticker)
        company = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        company = ticker

    try:
        page = get_wikipedia_page(company)
    except Exception as exc:
        st.error(f"Wikipedia lookup failed: {exc}")
        return

    if page is None:
        st.info(f"No Wikipedia page was found for {company}.")
        return

    st.write(f"### {page.title}")
    st.write(page.summary)

    if page.url:
        st.link_button("Open Wikipedia Page", page.url)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(APP_TITLE)
    st.caption(
        "Historical market analysis and strategy backtesting using Yahoo Finance."
    )

    # Sidebar controls.
    with st.sidebar:
        st.header("Market Settings")

        ticker = st.text_input(
            "Ticker Symbol",
            value=DEFAULT_TICKER,
        ).strip().upper()

        start_date = st.date_input(
            "Start Date",
            value=pd.Timestamp(DEFAULT_START_DATE).date(),
        )

        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=100.0,
            value=DEFAULT_INITIAL_CAPITAL,
            step=100.0,
        )

        transaction_cost_percent = st.number_input(
            "Transaction Cost (%)",
            min_value=0.0,
            value=DEFAULT_TRANSACTION_COST * 100,
            step=0.1,
        )

        slippage_percent = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            value=DEFAULT_SLIPPAGE * 100,
            step=0.1,
        )

        transaction_cost = transaction_cost_percent / 100
        slippage = slippage_percent / 100

        st.divider()

        dashboard = st.selectbox(
            "Dashboard",
            [
                "Overview",
                "SMA Backtest",
                "EMA Backtest",
                "MACD Backtest",
                "Stochastic RSI Backtest",
                "Fundamental Analysis",
                "News",
                "Stock Predictions",
                "Wikipedia",
            ],
        )

    if not ticker:
        st.warning("Enter a ticker symbol.")
        return

    # Data is required by most dashboards.
    data_required = dashboard not in {
        "Fundamental Analysis",
        "Wikipedia",
    }

    data = None

    if data_required:
        with st.spinner(f"Downloading {ticker} historical data..."):
            try:
                data = download_history(ticker, str(start_date))
            except Exception as exc:
                st.error(f"Could not load {ticker}: {exc}")
                st.info(
                    "Check the ticker symbol, start date, and internet connection."
                )
                return

    # Dashboard routing.
    if dashboard == "Overview":
        st.header(f"{ticker} Market Overview")

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        cols = st.columns(5)
        cols[0].metric("Last Close", f"${latest['Close']:,.2f}")
        cols[1].metric(
            "Daily Change",
            f"${latest['Close'] - previous['Close']:+,.2f}",
        )
        cols[2].metric("Day High", f"${latest['High']:,.2f}")
        cols[3].metric("Day Low", f"${latest['Low']:,.2f}")
        cols[4].metric(
            "Volume",
            f"{int(latest['Volume']):,}",
        )

        st.write("### Price History")
        st.line_chart(data["Close"])

        st.write("### Trading Volume")
        st.bar_chart(data["Volume"])

        with st.expander("Historical Data"):
            st.dataframe(
                data.tail(500),
                use_container_width=True,
            )

    elif dashboard == "SMA Backtest":
        sma_dashboard(
            data,
            initial_capital,
            transaction_cost,
            slippage,
        )

    elif dashboard == "EMA Backtest":
        ema_dashboard(
            data,
            initial_capital,
            transaction_cost,
            slippage,
        )

    elif dashboard == "MACD Backtest":
        macd_dashboard(
            data,
            initial_capital,
            transaction_cost,
            slippage,
        )

    elif dashboard == "Stochastic RSI Backtest":
        stochastic_rsi_dashboard(
            data,
            initial_capital,
            transaction_cost,
            slippage,
        )

    elif dashboard == "Fundamental Analysis":
        fundamental_dashboard(ticker)

    elif dashboard == "News":
        news_dashboard(ticker, data)

    elif dashboard == "Stock Predictions":
        stock_prediction_dashboard(data, ticker)

    elif dashboard == "Wikipedia":
        wikipedia_dashboard(ticker)


if __name__ == "__main__":
    main()
