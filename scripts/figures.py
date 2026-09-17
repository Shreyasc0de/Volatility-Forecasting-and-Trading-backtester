"""Render the three figures the README argues with.

    python scripts/figures.py

Each figure carries one finding. Nothing here recomputes anything: every value
comes from reports/<SYMBOL>/metrics.json, so a figure cannot disagree with the
prose or with the verifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SYMBOLS = ("SPY", "AAPL", "BTC-USD")
MODELS = ("constant", "rolling_20d", "rolling_60d", "ewma_0.94", "garch", "egarch", "gjr")
GARCH = {"garch", "egarch", "gjr"}

INK, MUTED, HAIRLINE = "#2B2B2B", "#5C5C5C", "#D8D8D8"
ACCENT, ALERT, QUIET = "#0B6E4F", "#C1443C", "#B6BDBA"
LABEL = {"rolling_20d": "rolling 20d", "rolling_60d": "rolling 60d",
         "ewma_0.94": "EWMA 0.94", "garch": "GARCH", "egarch": "EGARCH",
         "gjr": "GJR", "constant": "constant"}


def load() -> dict:
    out = {}
    for symbol in SYMBOLS:
        path = ROOT / "reports" / symbol / "metrics.json"
        if not path.exists():
            sys.exit(f"missing {path}. Run scripts/backtest.py --symbol {symbol} first.")
        out[symbol] = json.loads(path.read_text())
    return out


def _tidy(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set(linewidth=0.8, color=HAIRLINE)
    ax.tick_params(length=3, colors=MUTED, labelsize=9)


def ranks(scores: dict, target: str) -> dict:
    """Model -> 1-based rank under ``target``, best QLIKE first."""
    order = sorted(MODELS, key=lambda m: scores[m][target]["qlike"])
    return {m: i + 1 for i, m in enumerate(order)}


def figure_rank_inversion(data: dict, path: Path) -> None:
    """The methodological finding: the proxy decides the winner.

    A bump chart with everything grey except the two things that swap ends.
    Seven categorical hues would spend the whole colour channel on identity
    the labels already carry.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    for ax, symbol in zip(axes, SYMBOLS):
        scores = data[symbol]["vol_forecast_scores"]
        left, right = ranks(scores, "overlapping_20d"), ranks(scores, "one_bar")
        for model in MODELS:
            if model in GARCH:
                colour, width, z = ACCENT, 2.2, 4
            elif model == "rolling_20d":
                colour, width, z = ALERT, 2.2, 4
            else:
                colour, width, z = QUIET, 1.2, 2
            ax.plot([0, 1], [left[model], right[model]], color=colour, lw=width, zorder=z)
            ax.plot([0, 1], [left[model], right[model]], "o", color=colour, ms=5, zorder=z)
            if symbol == "SPY":
                ax.annotate(LABEL[model], xy=(0, left[model]), xytext=(-8, 0),
                            textcoords="offset points", ha="right", va="center",
                            fontsize=8.5, color=INK if colour != QUIET else MUTED)
            if symbol == "BTC-USD":
                ax.annotate(LABEL[model], xy=(1, right[model]), xytext=(8, 0),
                            textcoords="offset points", ha="left", va="center",
                            fontsize=8.5, color=INK if colour != QUIET else MUTED)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["overlapping\n20d proxy", "one-bar\nproxy"], fontsize=9)
        ax.set_title(symbol, fontsize=10, color=INK, pad=8)
        _tidy(ax)
    axes[0].set_ylim(7.6, 0.4)
    # The model names sit in the left and right margins, so numeric rank ticks
    # and an axis label would collide with them. Rank is positional and the
    # subtitle states the direction.
    axes[0].set_yticks([])
    for ax in axes:
        ax.spines["left"].set_visible(False)
    fig.suptitle("Swapping the realised-volatility proxy inverts the ranking",
                 fontsize=12.5, color=INK, y=0.985)
    fig.text(0.5, 0.905, "ranked by QLIKE, best at the top",
             ha="center", fontsize=9, color=MUTED)
    fig.text(0.5, 0.02,
             "Green: the three GARCH variants.  Red: the 20-day rolling window, whose "
             "own input window the overlapping proxy almost duplicates.",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0.10, 0.06, 0.91, 0.89))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_matched_drawdown(data: dict, path: Path) -> None:
    """The exposure finding: most of the drawdown gain is being less invested."""
    strategies = ("vol_target", "regime_only", "vol_and_regime")
    names = {"vol_target": "vol target", "regime_only": "regime only",
             "vol_and_regime": "vol + regime"}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for ax, symbol in zip(axes, SYMBOLS):
        table = data[symbol]["strategies"]
        bench = abs(table["buy_and_hold"]["max_drawdown"]) * 100
        ys = np.arange(len(strategies))
        for y, key in zip(ys, strategies):
            raw = abs(table[key]["max_drawdown"]) * 100
            matched = abs(table[key]["matched_max_drawdown"]) * 100
            ax.plot([raw, matched], [y, y], color=HAIRLINE, lw=2.4, zorder=1,
                    solid_capstyle="round")
            ax.plot(raw, y, "o", color=ACCENT, ms=8, zorder=3,
                    label="as reported" if y == 0 and symbol == "SPY" else None)
            ax.plot(matched, y, "o", color=ALERT, ms=8, zorder=3,
                    label="at matched risk" if y == 0 and symbol == "SPY" else None)
            ax.annotate(f"{table[key]['mean_exposure']:.0%} invested",
                        xy=(raw, y), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8, color=MUTED)
        ax.axvline(bench, color=INK, ls="--", lw=1.2, zorder=2)
        ax.annotate("buy and hold", xy=(bench, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, 6), textcoords="offset points", ha="left", va="bottom",
                    fontsize=8.5, color=INK, annotation_clip=False)
        ax.set_yticks(ys)
        ax.set_yticklabels([names[s] for s in strategies])
        ax.set_ylim(len(strategies) - 0.4, -0.7)
        ax.set_xlabel("Max drawdown (%)")
        ax.set_title(symbol, fontsize=10, color=INK, pad=14)
        _tidy(ax)
    # A per-axes legend landed on top of the bottom row's exposure label, so it
    # goes in the figure margin instead.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Lever each overlay to the benchmark's volatility and its drawdown advantage goes",
                 fontsize=12.5, color=INK, y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_lookahead(data: dict, path: Path) -> None:
    """What the original version's lookahead was worth, in Sharpe."""
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(SYMBOLS))
    width = 0.26
    bench = [data[s]["strategies"]["buy_and_hold"]["sharpe"] for s in SYMBOLS]
    honest = [data[s]["strategies"]["vol_and_regime"]["sharpe"] for s in SYMBOLS]
    peek = [data[s]["strategies"]["vol_and_regime_LOOKAHEAD"]["sharpe"] for s in SYMBOLS]

    ax.bar(x - width, bench, width, color=INK, label="buy and hold")
    ax.bar(x, honest, width, color=ACCENT, label="overlay, filtered states")
    ax.bar(x + width, peek, width, color=ALERT, label="overlay, smoothed states (lookahead)")
    for xi, (b, h, p) in zip(x, zip(bench, honest, peek)):
        for offset, value in ((-width, b), (0, h), (width, p)):
            ax.annotate(f"{value:.2f}", xy=(xi + offset, value), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
        gain = data[SYMBOLS[xi]]["lookahead_sharpe_gain_pct"]
        ax.annotate(f"+{gain:.0f}%", xy=(xi + width, p), xytext=(0, 18),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=ALERT, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SYMBOLS)
    ax.set_ylabel("Sharpe ratio")
    ax.set_ylim(0, max(peek) * 1.28)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Smoothed regime probabilities use the future, and it shows",
                 fontsize=12, color=INK, pad=10)
    ax.grid(axis="y", color=HAIRLINE, lw=0.7)
    ax.set_axisbelow(True)
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    data = load()
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figure_rank_inversion(data, out / "rank_inversion.png")
    figure_matched_drawdown(data, out / "matched_drawdown.png")
    figure_lookahead(data, out / "lookahead.png")
    print(f"[figures] wrote 3 figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
