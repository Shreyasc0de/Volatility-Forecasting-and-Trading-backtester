export type AssetSymbol = 'SPY' | 'AAPL' | 'BTC-USD';

export interface DataPoint {
  date: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  logReturn: number;
  rollingVol20: number;
  rollingVol60: number;
  ewmaVol: number;
  intradayRealizedVol: number;
  event?: string;
}

export interface AssetSummary {
  symbol: AssetSymbol;
  name: string;
  assetClass: 'Equities (Index)' | 'Equities (Single Stock)' | 'Crypto';
  period: string;
  totalDays: number;
  meanDailyReturn: number;
  annualizedReturn: number;
  annualizedVol: number;
  skewness: number;
  kurtosis: number;
  excessKurtosis: number;
  archLmStat: number;
  archLmPValue: number;
  hasArchEffects: boolean;
  jarqueBeraStat: number;
  jarqueBeraPValue: number;
}

export interface ModelParameters {
  omega: number;
  alpha: number;
  beta: number;
  gamma?: number; // For GJR-GARCH
  theta?: number; // For EGARCH
  persistence: number;
  halfLife: number;
  unconditionalVol: number;
}

export interface VolModelResult {
  id: string;
  name: string;
  type: 'Baseline' | 'GARCH' | 'GJR-GARCH' | 'EGARCH' | 'Regime-GARCH';
  description: string;
  params: ModelParameters;
  aic: number;
  bic: number;
  logLikelihood: number;
  inSampleRmse: number;
  outSampleRmse: number;
  outSampleQlike: number;
  outSampleMae: number;
  forecasts: number[]; // forecasted annualized volatility series
}

export interface MarkovRegimeResult {
  smoothedProbabilities: number[]; // Prob of Regime 1 (Turbulent)
  filteredProbabilities: number[];
  regimeAssignments: number[]; // 0 = Calm, 1 = Turbulent
  p00: number; // Prob of remaining in Calm
  p01: number; // Prob of switching Calm -> Turbulent
  p10: number; // Prob of switching Turbulent -> Calm
  p11: number; // Prob of remaining in Turbulent
  expectedDurationCalm: number; // days
  expectedDurationTurbulent: number; // days
  calmMeanReturn: number;
  calmVol: number;
  turbulentMeanReturn: number;
  turbulentVol: number;
  leadLagCorrelations: { lag: number; correlation: number }[];
}

export interface BacktestConfig {
  symbol: AssetSymbol;
  baseStrategy: 'BuyAndHold' | 'TrendSMA20_50' | 'Momentum12M';
  volOverlay: 'None' | 'InverseVol' | 'RegimeGating' | 'Combined';
  targetVolatility: number; // e.g. 0.15 (15%)
  maxLeverage: number; // e.g. 1.5
  regimeThreshold: number; // e.g. 0.50
  transactionCostBps: number; // e.g. 5 bps = 0.0005
  rebalanceFrequency: 'Daily' | 'Weekly';
  initialCapital: number;
}

export interface BacktestPoint {
  date: string;
  close: number;
  benchmarkEquity: number;
  baseStrategyEquity: number;
  overlayEquity: number;
  benchmarkDrawdown: number;
  baseDrawdown: number;
  overlayDrawdown: number;
  positionWeight: number;
  regimeProb: number;
  forecastVol: number;
  realizedVol: number;
  turnover: number;
}

export interface PerformanceMetrics {
  totalReturn: number;
  cagr: number;
  annualizedVol: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  calmarRatio: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  turnoverPerYear: number;
}

export interface BacktestResult {
  config: BacktestConfig;
  points: BacktestPoint[];
  benchmarkMetrics: PerformanceMetrics;
  baseMetrics: PerformanceMetrics;
  overlayMetrics: PerformanceMetrics;
  monthlyReturns: { year: number; months: number[] }[];
}
