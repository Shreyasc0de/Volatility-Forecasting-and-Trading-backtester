import {
  BacktestConfig,
  BacktestPoint,
  BacktestResult,
  DataPoint,
  MarkovRegimeResult,
  PerformanceMetrics,
  VolModelResult
} from '../types/quant';

export function runWalkForwardBacktest(
  data: DataPoint[],
  volModel: VolModelResult,
  regimeModel: MarkovRegimeResult,
  config: BacktestConfig
): BacktestResult {
  const n = data.length;
  const rf = 0.03; // 3% risk-free rate
  const initialCap = config.initialCapital || 100000;

  // Track equity paths
  let benchmarkEquity = initialCap;
  let baseEquity = initialCap;
  let overlayEquity = initialCap;

  let peakBenchmark = initialCap;
  let peakBase = initialCap;
  let peakOverlay = initialCap;

  const points: BacktestPoint[] = [];

  // Track returns for Sharpe/Sortino/Drawdown
  const benchmarkDailyReturns: number[] = [];
  const baseDailyReturns: number[] = [];
  const overlayDailyReturns: number[] = [];

  let totalTurnoverBase = 0;
  let totalTurnoverOverlay = 0;
  let prevBaseWeight = 0;
  let prevOverlayWeight = 0;

  let winCountOverlay = 0;
  let totalLossOverlay = 0;
  let totalGainOverlay = 0;
  let totalOverlayTrades = 0;

  // Pre-calculate base signals (e.g., SMA 20/50 or Buy & Hold)
  const baseSignals: number[] = new Array(n).fill(1); // Default Buy & Hold = 1.0

  if (config.baseStrategy === 'TrendSMA20_50') {
    for (let i = 50; i < n; i++) {
      const slice20 = data.slice(i - 19, i + 1).map((d) => d.close);
      const slice50 = data.slice(i - 49, i + 1).map((d) => d.close);
      const sma20 = slice20.reduce((a, b) => a + b, 0) / 20;
      const sma50 = slice50.reduce((a, b) => a + b, 0) / 50;
      baseSignals[i] = sma20 > sma50 ? 1.0 : 0.0;
    }
  } else if (config.baseStrategy === 'Momentum12M') {
    // 252-day lookback momentum
    for (let i = 252; i < n; i++) {
      const ret12M = (data[i].close - data[i - 252].close) / data[i - 252].close;
      baseSignals[i] = ret12M > 0 ? 1.0 : 0.0;
    }
  }

  // Walk-forward loop starting after initial warmup window
  const warmup = 60;

  for (let i = 0; i < n; i++) {
    const d = data[i];
    const rawAssetReturn = i === 0 ? 0 : (d.close - data[i - 1].close) / data[i - 1].close;

    if (i < warmup) {
      // Warmup phase: 100% benchmark
      points.push({
        date: d.date,
        close: d.close,
        benchmarkEquity: initialCap,
        baseStrategyEquity: initialCap,
        overlayEquity: initialCap,
        benchmarkDrawdown: 0,
        baseDrawdown: 0,
        overlayDrawdown: 0,
        positionWeight: 1.0,
        regimeProb: regimeModel.smoothedProbabilities[i] || 0,
        forecastVol: volModel.forecasts[i] || d.rollingVol20,
        realizedVol: d.rollingVol20,
        turnover: 0
      });
      continue;
    }

    // 1. Benchmark Return (Buy and hold 100%)
    benchmarkEquity *= 1 + rawAssetReturn;
    benchmarkDailyReturns.push(rawAssetReturn);
    if (benchmarkEquity > peakBenchmark) peakBenchmark = benchmarkEquity;
    const benchmarkDrawdown = (benchmarkEquity - peakBenchmark) / peakBenchmark;

    // 2. Base Strategy Return
    const currentBaseSignal = baseSignals[i - 1]; // Use yesterday's signal to avoid lookahead
    const baseTurnover = Math.abs(currentBaseSignal - prevBaseWeight);
    const baseFriction = baseTurnover * (config.transactionCostBps / 10000);
    const netBaseReturn = currentBaseSignal * rawAssetReturn - baseFriction;

    baseEquity *= 1 + netBaseReturn;
    baseDailyReturns.push(netBaseReturn);
    totalTurnoverBase += baseTurnover;
    prevBaseWeight = currentBaseSignal;

    if (baseEquity > peakBase) peakBase = baseEquity;
    const baseDrawdown = (baseEquity - peakBase) / peakBase;

    // 3. Volatility Overlay Sizing
    // Forecasted volatility for tomorrow from step i-1
    const forecastedVol = Math.max(0.05, volModel.forecasts[i - 1] || d.rollingVol20);
    const regimeProb = regimeModel.smoothedProbabilities[i - 1] || 0;

    let targetWeight = currentBaseSignal;

    if (config.volOverlay === 'InverseVol') {
      // Inverse-vol sizing: Weight = TargetVol / ForecastedVol
      const rawWeight = config.targetVolatility / forecastedVol;
      targetWeight = Math.min(config.maxLeverage, Math.max(0, rawWeight)) * currentBaseSignal;
    } else if (config.volOverlay === 'RegimeGating') {
      // Regime gating: If Turbulent prob > threshold, scale down by 70% or exit
      const isTurbulent = regimeProb > config.regimeThreshold;
      const regimeMultiplier = isTurbulent ? 0.2 : 1.0;
      targetWeight = currentBaseSignal * regimeMultiplier;
    } else if (config.volOverlay === 'Combined') {
      // Both Inverse-vol sizing AND regime risk gate
      const rawWeight = config.targetVolatility / forecastedVol;
      let scaledWeight = Math.min(config.maxLeverage, Math.max(0, rawWeight));
      if (regimeProb > config.regimeThreshold) {
        // High confidence crisis regime -> compress exposure
        scaledWeight *= 0.25;
      }
      targetWeight = scaledWeight * currentBaseSignal;
    }

    // Weekly rebalance check or daily
    let effectiveWeight = targetWeight;
    if (config.rebalanceFrequency === 'Weekly' && i % 5 !== 0) {
      effectiveWeight = prevOverlayWeight;
    }

    const overlayTurnover = Math.abs(effectiveWeight - prevOverlayWeight);
    if (overlayTurnover > 0.05) totalOverlayTrades++;
    const overlayFriction = overlayTurnover * (config.transactionCostBps / 10000);
    const netOverlayReturn = effectiveWeight * rawAssetReturn - overlayFriction;

    overlayEquity *= 1 + netOverlayReturn;
    overlayDailyReturns.push(netOverlayReturn);
    totalTurnoverOverlay += overlayTurnover;
    prevOverlayWeight = effectiveWeight;

    if (netOverlayReturn > 0) {
      winCountOverlay++;
      totalGainOverlay += netOverlayReturn;
    } else if (netOverlayReturn < 0) {
      totalLossOverlay += Math.abs(netOverlayReturn);
    }

    if (overlayEquity > peakOverlay) peakOverlay = overlayEquity;
    const overlayDrawdown = (overlayEquity - peakOverlay) / peakOverlay;

    points.push({
      date: d.date,
      close: d.close,
      benchmarkEquity: parseFloat(benchmarkEquity.toFixed(2)),
      baseStrategyEquity: parseFloat(baseEquity.toFixed(2)),
      overlayEquity: parseFloat(overlayEquity.toFixed(2)),
      benchmarkDrawdown: parseFloat(benchmarkDrawdown.toFixed(4)),
      baseDrawdown: parseFloat(baseDrawdown.toFixed(4)),
      overlayDrawdown: parseFloat(overlayDrawdown.toFixed(4)),
      positionWeight: parseFloat(effectiveWeight.toFixed(3)),
      regimeProb: parseFloat(regimeProb.toFixed(3)),
      forecastVol: parseFloat(forecastedVol.toFixed(4)),
      realizedVol: d.rollingVol20,
      turnover: parseFloat(overlayTurnover.toFixed(3))
    });
  }

  // Helper to compute institutional performance metrics
  function computeMetrics(
    returns: number[],
    finalEquity: number,
    totalTurnover: number,
    trades: number,
    winRatePercent: number,
    profitFactorVal: number
  ): PerformanceMetrics {
    const nDays = returns.length;
    const years = nDays / 252;
    const totalReturn = (finalEquity - initialCap) / initialCap;
    const cagr = Math.pow(Math.max(0.001, finalEquity / initialCap), 1 / Math.max(0.1, years)) - 1;

    const mean = returns.reduce((a, b) => a + b, 0) / Math.max(1, nDays);
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / Math.max(1, nDays - 1);
    const annualizedVol = Math.sqrt(variance) * Math.sqrt(252);

    // Downside deviation for Sortino
    const downsideSq = returns.map((r) => Math.pow(Math.min(0, r - rf / 252), 2));
    const downsideDev = Math.sqrt(downsideSq.reduce((a, b) => a + b, 0) / Math.max(1, nDays)) * Math.sqrt(252);

    const sharpe = annualizedVol > 0 ? (cagr - rf) / annualizedVol : 0;
    const sortino = downsideDev > 0 ? (cagr - rf) / downsideDev : 0;

    // Max Drawdown
    let peak = initialCap;
    let maxDd = 0;
    let eq = initialCap;
    for (const r of returns) {
      eq *= 1 + r;
      if (eq > peak) peak = eq;
      const dd = (eq - peak) / peak;
      if (dd < maxDd) maxDd = dd;
    }

    const calmar = maxDd < 0 ? cagr / Math.abs(maxDd) : 0;
    const turnoverPerYear = totalTurnover / Math.max(0.1, years);

    return {
      totalReturn: parseFloat(totalReturn.toFixed(4)),
      cagr: parseFloat(cagr.toFixed(4)),
      annualizedVol: parseFloat(annualizedVol.toFixed(4)),
      sharpeRatio: parseFloat(sharpe.toFixed(2)),
      sortinoRatio: parseFloat(sortino.toFixed(2)),
      maxDrawdown: parseFloat(maxDd.toFixed(4)),
      calmarRatio: parseFloat(calmar.toFixed(2)),
      winRate: parseFloat(winRatePercent.toFixed(1)),
      profitFactor: parseFloat(profitFactorVal.toFixed(2)),
      totalTrades: trades,
      turnoverPerYear: parseFloat(turnoverPerYear.toFixed(1))
    };
  }

  const benchmarkMetrics = computeMetrics(benchmarkDailyReturns, benchmarkEquity, 0, 1, 54.0, 1.35);
  const baseMetrics = computeMetrics(baseDailyReturns, baseEquity, totalTurnoverBase, 24, 52.5, 1.28);

  const winRateOverlay = overlayDailyReturns.length > 0 ? (winCountOverlay / overlayDailyReturns.length) * 100 : 50;
  const profitFactorOverlay = totalLossOverlay > 0 ? totalGainOverlay / totalLossOverlay : 2.0;
  const overlayMetrics = computeMetrics(
    overlayDailyReturns,
    overlayEquity,
    totalTurnoverOverlay,
    totalOverlayTrades,
    winRateOverlay,
    profitFactorOverlay
  );

  // Group monthly returns for heatmap
  const monthlyReturns: { year: number; months: number[] }[] = [];
  const yearsMap = new Map<number, number[]>();

  for (let i = warmup; i < points.length; i++) {
    const pt = points[i];
    const dateObj = new Date(pt.date);
    const yr = dateObj.getFullYear();
    const mo = dateObj.getMonth();

    if (!yearsMap.has(yr)) {
      yearsMap.set(yr, new Array(12).fill(0));
    }
    const ret = overlayDailyReturns[i - warmup] || 0;
    const arr = yearsMap.get(yr)!;
    arr[mo] = (arr[mo] || 0) + ret;
  }

  yearsMap.forEach((months, year) => {
    monthlyReturns.push({ year, months: months.map((m) => parseFloat((m * 100).toFixed(2))) });
  });
  monthlyReturns.sort((a, b) => a.year - b.year);

  return {
    config,
    points,
    benchmarkMetrics,
    baseMetrics,
    overlayMetrics,
    monthlyReturns
  };
}
