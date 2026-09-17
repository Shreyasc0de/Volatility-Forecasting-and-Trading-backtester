"""Price loading, with a committed cache so results are reproducible.

yfinance is a live, unversioned source: the same call can return different
history a month later, after splits, dividend adjustments and vendor
backfills. A backtest whose numbers cannot be reproduced is not a backtest, so
every fetch is cached to CSV and the cache is what the pipeline reads. The
cache is committed and the README quotes its date range.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DataError(RuntimeError):
    pass


def cache_path(data_dir: str | Path, symbol: str) -> Path:
    return Path(data_dir) / f"{symbol.replace('/', '-')}.csv"


def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV. Requires yfinance and network access."""
    try:
        import yfinance
    except ImportError as exc:
        raise DataError(
            "yfinance is not installed, so no new symbol can be downloaded. "
            "Existing cached CSVs under data/ still work."
        ) from exc

    frame = yfinance.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if frame is None or frame.empty:
        raise DataError(f"yfinance returned nothing for {symbol} between {start} and {end}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename_axis("date").reset_index()
    keep = [c for c in ("date", "Open", "High", "Low", "Close", "Volume") if c in frame.columns]
    return frame[keep]


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a price CSV in either this project's format or yfinance's own.

    yfinance writes a three-row header when given a single ticker
    (``Price,Close,...`` / ``Ticker,SPY,...`` / ``Date,,,,``), which pandas
    reads as three columns of garbage unless it is told. Both shapes are
    accepted so the CSVs already sitting in this repo can be used directly
    rather than re-downloaded, which also means the numbers stay reproducible
    if the vendor revises its history.
    """
    path = Path(path)
    head = pd.read_csv(path, nrows=3, header=None)
    looks_like_yfinance = str(head.iloc[0, 0]).strip() in {"Price", "Unnamed: 0"} and \
        str(head.iloc[1, 0]).strip() == "Ticker"
    if looks_like_yfinance:
        frame = pd.read_csv(path, skiprows=3, header=None,
                            names=["date", "Close", "High", "Low", "Open", "Volume"])
    else:
        frame = pd.read_csv(path)
        lower = {c.lower(): c for c in frame.columns}
        if "date" in lower and lower["date"] != "date":
            frame = frame.rename(columns={lower["date"]: "date"})
    if "date" not in frame.columns or "Close" not in frame.columns:
        raise DataError(f"{path} has no date/Close columns; got {list(frame.columns)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["date", "Close"])
    if frame.empty:
        raise DataError(f"{path} produced no usable rows")
    return frame.sort_values("date").reset_index(drop=True)


def load(symbol: str, data_dir: str | Path, start: str = "2015-01-01",
         end: str = "2025-12-31", allow_fetch: bool = True) -> pd.DataFrame:
    """Read the cached CSV for ``symbol``, downloading and caching it if absent."""
    path = cache_path(data_dir, symbol)
    if not path.exists():
        if not allow_fetch:
            raise DataError(f"{path} is missing and fetching is disabled")
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        fetch(symbol, start, end).to_csv(path, index=False)
    return read_csv(path)


def simple_returns(prices: pd.DataFrame) -> pd.Series:
    """Daily simple returns, indexed by date.

    Simple, not log, because every downstream calculation compounds them. The
    previous version produced log returns and then compounded them as if they
    were simple, which understated growth by more the higher the volatility.
    """
    series = prices.set_index("date")["Close"].astype(float)
    if series.le(0).any():
        raise DataError("non-positive close prices in the series")
    return series.pct_change().dropna()


def summarise(returns: pd.Series) -> dict:
    """Series facts the README is allowed to quote."""
    r = returns.dropna()
    return {
        "n_days": int(len(r)),
        "first_day": r.index.min().date().isoformat(),
        "last_day": r.index.max().date().isoformat(),
        "annualised_vol": float(r.std(ddof=1) * np.sqrt(252)),
        "annualised_return": float((1 + r).prod() ** (252 / len(r)) - 1),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
        "worst_day": float(r.min()),
        "worst_day_date": r.idxmin().date().isoformat(),
        "best_day": float(r.max()),
        "best_day_date": r.idxmax().date().isoformat(),
    }
