import React, { useState, useMemo } from 'react';
import {
  AssetSymbol,
  BacktestConfig,
  DataPoint,
  MarkovRegimeResult,
  VolModelResult
} from '../types/quant';
import { runWalkForwardBacktest } from '../math/backtest';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
  ReferenceLine
} from 'recharts';
import {
  SlidersHorizontal,
  TrendingUp,
  ShieldCheck,
  Percent,
  RefreshCw,
  Zap,
  ArrowUpRight,
  TrendingDown,
  Activity,
  Layers
} from 'lucide-react';

interface Props {
  symbol: AssetSymbol;
  data: DataPoint[];
  volModel: VolModelResult;
  regimeModel: MarkovRegimeResult;
}

export const Phase4BacktestEngine: React.FC<Props> = ({
  symbol,
  data,
  volModel,
  regimeModel
}) => {
  // Strategy Overlay Configuration State
  const [config, setConfig] = useState<BacktestConfig>({
    symbol,
    baseStrategy: 'BuyAndHold',
    volOverlay: 'Combined',
    targetVolatility: 0.15, // 15% annual target vol
    maxLeverage: 1.5,
    regimeThreshold: 0.5,
    transactionCostBps: 5.0, // 5 bps slippage/commission
    rebalanceFrequency: 'Daily',
    initialCapital: 100000
  });

  const [activeChartTab, setActiveChartTab] = useState<'equity' | 'drawdown' | 'weight'>('equity');

  // Run Walk-Forward simulation dynamically on parameter changes
  const backtestResult = useMemo(() => {
    return runWalkForwardBacktest(data, volModel, regimeModel, config);
  }, [data, volModel, regimeModel, config]);

  const { benchmarkMetrics, baseMetrics, overlayMetrics, points, monthlyReturns } = backtestResult;

  // Filter sample points for high performance charting
  const chartPoints = points.filter((_, idx) => idx % 2 === 0);

  // Sharpe Delta & Drawdown Reduction
  const sharpeDelta = overlayMetrics.sharpeRatio - baseMetrics.sharpeRatio;
  const mddReduction = ((Math.abs(baseMetrics.maxDrawdown) - Math.abs(overlayMetrics.maxDrawdown)) / Math.max(0.01, Math.abs(baseMetrics.maxDrawdown))) * 100;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 relative overflow-hidden shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 font-mono text-xs font-semibold border border-emerald-500/20">
                PHASE 4: STRATEGY & VOLATILITY OVERLAY ENGINE
              </span>
              <span className="text-xs text-slate-400 font-mono">Walk-Forward Simulation • Zero Lookahead Bias</span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Dynamic Volatility Sizing & Regime-Gated Backtest
            </h1>
            <p className="text-sm text-slate-400 max-w-3xl">
              Overlaying econometric GARCH forecasts (position weight proportional to target vol / forecast vol) and Hamilton regime filters onto base strategies to compress maximum drawdowns and maximize risk-adjusted Sharpe ratios.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="px-4 py-2 rounded-xl bg-slate-950 border border-slate-800 text-right">
              <div className="text-[10px] text-slate-400 font-mono uppercase">Sharpe Ratio Alpha</div>
              <div className={`text-lg font-bold font-mono ${sharpeDelta >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {sharpeDelta >= 0 ? `+${sharpeDelta.toFixed(2)}` : sharpeDelta.toFixed(2)}
              </div>
            </div>
            <div className="px-4 py-2 rounded-xl bg-slate-950 border border-slate-800 text-right">
              <div className="text-[10px] text-slate-400 font-mono uppercase">Max Drawdown Reduction</div>
              <div className="text-lg font-bold text-blue-400 font-mono">
                {mddReduction.toFixed(0)}% Compressed
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Parameter Control Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-blue-400" />
            <h2 className="text-sm font-semibold text-white">Strategy & Volatility Overlay Parameters</h2>
          </div>
          <span className="text-xs text-slate-400 font-mono">Real-Time Recomputation</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
          {/* Volatility Overlay Type */}
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-col justify-between gap-1.5">
            <label className="text-slate-400 text-[11px]">Volatility Overlay Mode</label>
            <select
              value={config.volOverlay}
              onChange={(e) => setConfig({ ...config, volOverlay: e.target.value as any })}
              className="bg-slate-900 text-white rounded-lg p-2 border border-slate-700 text-xs font-mono focus:outline-hidden focus:border-blue-500"
            >
              <option value="Combined">Combined (Sizing + Regime Gate)</option>
              <option value="InverseVol">Inverse-Vol Sizing Only</option>
              <option value="RegimeGating">Regime Gating Only</option>
              <option value="None">None (Unmanaged Base)</option>
            </select>
            <div className="text-[10px] text-slate-400">
              {config.volOverlay === 'Combined'
                ? 'Target vol sizing + 75% de-risk in turbulent state'
                : config.volOverlay === 'InverseVol'
                ? 'Scale weight by Target Vol / GARCH forecast'
                : config.volOverlay === 'RegimeGating'
                ? 'Scale down exposure when P(Crisis) > threshold'
                : 'Raw underlying base strategy'}
            </div>
          </div>

          {/* Base Strategy */}
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-col justify-between gap-1.5">
            <label className="text-slate-400 text-[11px]">Base Foundation Strategy</label>
            <select
              value={config.baseStrategy}
              onChange={(e) => setConfig({ ...config, baseStrategy: e.target.value as any })}
              className="bg-slate-900 text-white rounded-lg p-2 border border-slate-700 text-xs font-mono focus:outline-hidden focus:border-blue-500"
            >
              <option value="BuyAndHold">Buy & Hold (100% Long Benchmark)</option>
              <option value="TrendSMA20_50">Trend Following (SMA 20/50 Crossover)</option>
              <option value="Momentum12M">Time-Series Momentum (12-Month / 252d)</option>
            </select>
            <div className="text-[10px] text-slate-400">Isolates the exact alpha from the volatility overlay</div>
          </div>

          {/* Target Volatility Slider */}
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-col justify-between gap-1.5">
            <div className="flex justify-between items-center">
              <label className="text-slate-400 text-[11px]">Annual Target Volatility (sigma_target)</label>
              <span className="text-emerald-400 font-bold">{(config.targetVolatility * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.08"
              max="0.30"
              step="0.01"
              value={config.targetVolatility}
              onChange={(e) => setConfig({ ...config, targetVolatility: parseFloat(e.target.value) })}
              className="accent-emerald-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>Conservative (8%)</span>
              <span>Aggressive (30%)</span>
            </div>
          </div>

          {/* Max Leverage Cap */}
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex flex-col justify-between gap-1.5">
            <div className="flex justify-between items-center">
              <label className="text-slate-400 text-[11px]">Max Leverage / Position Cap</label>
              <span className="text-blue-400 font-bold">{config.maxLeverage.toFixed(1)}x</span>
            </div>
            <input
              type="range"
              min="1.0"
              max="2.5"
              step="0.1"
              value={config.maxLeverage}
              onChange={(e) => setConfig({ ...config, maxLeverage: parseFloat(e.target.value) })}
              className="accent-blue-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>Unlevered (1.0x)</span>
              <span>Max Cap (2.5x)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Institutional Performance Scorecard Comparison */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-semibold text-white">Institutional Backtest Performance Comparison (2020 - 2025)</h2>
          </div>
          <span className="text-xs font-mono text-slate-400">Friction: {config.transactionCostBps} bps per trade</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider text-[11px] bg-slate-950/60">
                <th className="py-3 px-4">Performance Metric</th>
                <th className="py-3 px-4 text-slate-300">1. Benchmark (Buy & Hold)</th>
                <th className="py-3 px-4 text-slate-300">2. Base Strategy ({config.baseStrategy})</th>
                <th className="py-3 px-4 text-emerald-400 font-bold bg-emerald-500/10">3. Base + Vol Overlay ({config.volOverlay})</th>
                <th className="py-3 px-4 text-right">Vol Overlay Alpha Δ</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              <tr>
                <td className="py-3 px-4 font-semibold text-slate-300">Total Return (%)</td>
                <td className="py-3 px-4 text-slate-400">{(benchmarkMetrics.totalReturn * 100).toFixed(1)}%</td>
                <td className="py-3 px-4 text-slate-400">{(baseMetrics.totalReturn * 100).toFixed(1)}%</td>
                <td className="py-3 px-4 font-bold text-emerald-400 bg-emerald-500/5">
                  +{(overlayMetrics.totalReturn * 100).toFixed(1)}%
                </td>
                <td className="py-3 px-4 text-right font-bold text-emerald-400">
                  +{((overlayMetrics.totalReturn - baseMetrics.totalReturn) * 100).toFixed(1)}%
                </td>
              </tr>

              <tr>
                <td className="py-3 px-4 font-semibold text-slate-300">CAGR (Annualized Return)</td>
                <td className="py-3 px-4 text-slate-400">{(benchmarkMetrics.cagr * 100).toFixed(1)}%</td>
                <td className="py-3 px-4 text-slate-400">{(baseMetrics.cagr * 100).toFixed(1)}%</td>
                <td className="py-3 px-4 font-bold text-emerald-400 bg-emerald-500/5">
                  {(overlayMetrics.cagr * 100).toFixed(1)}%
                </td>
                <td className="py-3 px-4 text-right text-emerald-400 font-bold">
                  +{((overlayMetrics.cagr - baseMetrics.cagr) * 100).toFixed(1)}%
                </td>
              </tr>

              <tr>
                <td className="py-3 px-4 font-semibold text-slate-300">Sharpe Ratio (rf=3%)</td>
                <td className="py-3 px-4 text-slate-400">{benchmarkMetrics.sharpeRatio}</td>
                <td className="py-3 px-4 text-slate-400">{baseMetrics.sharpeRatio}</td>
                <td className="py-3 px-4 font-bold text-emerald-400 bg-emerald-500/5">
                  {overlayMetrics.sharpeRatio}
                </td>
                <td className="py-3 px-4 text-right text-emerald-400 font-bold">
                  +{sharpeDelta >= 0 ? `+${sharpeDelta.toFixed(2)}` : sharpeDelta.toFixed(2)}
                </td>
              </tr>

              <tr>
                <td className="py-3 px-4 font-semibold text-slate-300">Sortino Ratio (Downside Risk)</td>
                <td className="py-3 px-4 text-slate-400">{benchmarkMetrics.sortinoRatio}</td>
                <td className="py-3 px-4 text-slate-400">{baseMetrics.sortinoRatio}</td>
                <td className="py-3 px-4 font-bold text-emerald-400 bg-emerald-500/5">
                  {overlayMetrics.sortinoRatio}
                </td>
                <td className="py-3 px-4 text-right text-emerald-400 font-bold">
                  +{(overlayMetrics.sortinoRatio - baseMetrics.sortinoRatio).toFixed(2)}
                </td>
              </tr>

              <tr>
                <td className="py-3 px-4 font-semibold text-slate-300">Maximum Drawdown (MDD)</td>
                <td className="py-3 px-4 text-rose-400">{(benchmarkMetrics.maxDrawdown * 100).toFixed(1)}%</td>
                <td className="py-3 px-4 text-rose-400">{(baseMetrics.maxDrawdown * 100).toFixed(1)}%</td>
                <td className="py-3 px-4 font-bold text-emerald-300 bg-emerald-500/5">
                  {(overlayMetrics.maxDrawdown * 100).toFixed(1)}%
                </td>
                <td className="py-3 px-4 text-right text-emerald-400 font-bold">
                  +{mddReduction.toFixed(0)}% Lower Risk
                </td>
              </tr>

              <tr>
                <td className="py-3 px-4 font-semibold text-slate-300">Annualized Realized Volatility</td>
                <td className="py-3 px-4 text-slate-400">{(benchmarkMetrics.annualizedVol * 100).toFixed(1)}%</td>
                <td className="py-3 px-4 text-slate-400">{(baseMetrics.annualizedVol * 100).toFixed(1)}%</td>
                <td className="py-3 px-4 font-bold text-blue-300 bg-emerald-500/5">
                  {(overlayMetrics.annualizedVol * 100).toFixed(1)}%
                </td>
                <td className="py-3 px-4 text-right text-blue-400">Target: {(config.targetVolatility * 100).toFixed(0)}%</td>
              </tr>

              <tr>
                <td className="py-3 px-4 font-semibold text-slate-300">Calmar Ratio (CAGR / |MDD|)</td>
                <td className="py-3 px-4 text-slate-400">{benchmarkMetrics.calmarRatio}</td>
                <td className="py-3 px-4 text-slate-400">{baseMetrics.calmarRatio}</td>
                <td className="py-3 px-4 font-bold text-emerald-400 bg-emerald-500/5">
                  {overlayMetrics.calmarRatio}
                </td>
                <td className="py-3 px-4 text-right text-emerald-400 font-bold">
                  +{(overlayMetrics.calmarRatio - baseMetrics.calmarRatio).toFixed(2)}
                </td>
              </tr>

              <tr>
                <td className="py-3 px-4 font-semibold text-slate-300">Annualized Turnover</td>
                <td className="py-3 px-4 text-slate-400">0.0x</td>
                <td className="py-3 px-4 text-slate-400">{baseMetrics.turnoverPerYear.toFixed(1)}x / yr</td>
                <td className="py-3 px-4 text-slate-300 bg-emerald-500/5">{overlayMetrics.turnoverPerYear.toFixed(1)}x / yr</td>
                <td className="py-3 px-4 text-right text-slate-400">{overlayMetrics.totalTrades} rebalances</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Main Interactive Performance Charts */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        {/* Chart View Selector */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-semibold text-white">Walk-Forward Portfolio Dynamics</h2>
          </div>

          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-mono">
            <button
              onClick={() => setActiveChartTab('equity')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                activeChartTab === 'equity' ? 'bg-emerald-600 text-white font-semibold shadow-xs' : 'text-slate-400 hover:text-white'
              }`}
            >
              Equity Curves ($100k)
            </button>
            <button
              onClick={() => setActiveChartTab('drawdown')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                activeChartTab === 'drawdown' ? 'bg-rose-600 text-white font-semibold shadow-xs' : 'text-slate-400 hover:text-white'
              }`}
            >
              Underwater Drawdowns
            </button>
            <button
              onClick={() => setActiveChartTab('weight')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                activeChartTab === 'weight' ? 'bg-blue-600 text-white font-semibold shadow-xs' : 'text-slate-400 hover:text-white'
              }`}
            >
              Dynamic Position Sizing & Leverage
            </button>
          </div>
        </div>

        {/* Tab 1: Equity Curves */}
        {activeChartTab === 'equity' && (
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartPoints} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} minTickGap={40} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  labelStyle={{ color: '#94a3b8' }}
                  formatter={(val: any, name: string) => [`$${Number(val).toLocaleString()}`, name]}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Line type="monotone" dataKey="benchmarkEquity" name={`Benchmark (${symbol})`} stroke="#64748b" dot={false} strokeWidth={1.5} opacity={0.6} />
                <Line type="monotone" dataKey="baseStrategyEquity" name={`Base Strategy (${config.baseStrategy})`} stroke="#3b82f6" dot={false} strokeWidth={1.8} />
                <Line type="monotone" dataKey="overlayEquity" name={`Base + Vol Overlay (${config.volOverlay})`} stroke="#10b981" dot={false} strokeWidth={2.4} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Tab 2: Underwater Drawdown */}
        {activeChartTab === 'drawdown' && (
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartPoints} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <defs>
                  <linearGradient id="overlayDdGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="benchmarkDdGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} minTickGap={40} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} domain={['auto', 0]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  labelStyle={{ color: '#94a3b8' }}
                  formatter={(val: any, name: string) => [`${(Number(val) * 100).toFixed(2)}%`, name]}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Area type="monotone" dataKey="benchmarkDrawdown" name="Benchmark Drawdown" stroke="#f43f5e" strokeWidth={1.5} fillOpacity={1} fill="url(#benchmarkDdGrad)" />
                <Area type="monotone" dataKey="overlayDrawdown" name="Overlay Strategy Drawdown" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#overlayDdGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Tab 3: Position Weight & Leverage */}
        {activeChartTab === 'weight' && (
          <div className="h-[340px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartPoints} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} minTickGap={40} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={[0, config.maxLeverage + 0.2]} tickFormatter={(v) => `${v.toFixed(1)}x`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  labelStyle={{ color: '#94a3b8' }}
                  formatter={(val: any, name: string) => [`${Number(val).toFixed(2)}x`, name]}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <ReferenceLine y={1.0} stroke="#64748b" strokeDasharray="3 3" label={{ value: '1.0x (100% Cash Equiv)', fill: '#64748b', fontSize: 10 }} />
                <Line type="stepAfter" dataKey="positionWeight" name="Dynamic Position Weight w_t" stroke="#38bdf8" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Monthly Returns Heatmap Grid */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Percent className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-white">Monthly Returns Heatmap (Strategy + Vol Overlay)</h3>
          </div>
          <span className="text-xs font-mono text-slate-400">Net of Friction (5 bps)</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-center text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                <th className="py-2 px-3 text-left">Year</th>
                {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map((m) => (
                  <th key={m} className="py-2 px-1">{m}</th>
                ))}
                <th className="py-2 px-3 text-right">YTD Total</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {monthlyReturns.map((row) => {
                const ytd = row.months.reduce((a, b) => a + b, 0);
                return (
                  <tr key={row.year} className="hover:bg-slate-800/30">
                    <td className="py-2 px-3 text-left font-bold text-white">{row.year}</td>
                    {row.months.map((ret, mIdx) => {
                      const isPos = ret > 0;
                      const isZero = ret === 0;
                      const bgStyle = isZero
                        ? 'text-slate-500'
                        : isPos
                        ? ret > 5
                          ? 'bg-emerald-500/25 text-emerald-300 font-bold'
                          : 'bg-emerald-500/10 text-emerald-400'
                        : ret < -5
                        ? 'bg-rose-500/25 text-rose-300 font-bold'
                        : 'bg-rose-500/10 text-rose-400';
                      return (
                        <td key={mIdx} className={`py-1 px-1 rounded-sm text-[11px] ${bgStyle}`}>
                          {isZero ? '—' : `${isPos ? '+' : ''}${ret.toFixed(1)}%`}
                        </td>
                      );
                    })}
                    <td className={`py-2 px-3 text-right font-bold ${ytd >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {ytd >= 0 ? `+${ytd.toFixed(1)}%` : `${ytd.toFixed(1)}%`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
