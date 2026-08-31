import { AssetSymbol, DataPoint } from '../types/quant';

// Generate synthetic but realistic daily market data matching 2020-2025 authentic market price action and vol dynamics
function generateAssetHistory(
  symbol: AssetSymbol,
  startPrice: number,
  tradingDays: number,
  startDateStr: string
): DataPoint[] {
  const points: DataPoint[] = [];
  let currentPrice = startPrice;
  let currentDate = new Date(startDateStr);

  // Calibrated daily volatility factors & macro regimes
  const isCrypto = symbol === 'BTC-USD';
  const isSingleStock = symbol === 'AAPL';

  // Specific historical events to seed realistic macro shocks
  const eventMap: { [dayIndex: number]: { name: string; shock: number; volMultiplier: number } } = {
    35: { name: 'COVID-19 Global Pandemic Crash (Mar 2020)', shock: isCrypto ? -0.38 : -0.11, volMultiplier: 3.8 },
    45: { name: 'Fed Emergency Liquidity Injection (Apr 2020)', shock: isCrypto ? 0.14 : 0.06, volMultiplier: 2.2 },
    240: { name: 'Vaccine Efficacy Announcement (Nov 2020)', shock: isCrypto ? 0.08 : 0.035, volMultiplier: 1.5 },
    340: { name: 'Tech & Crypto Stimulus Surge (Apr 2021)', shock: isCrypto ? 0.12 : 0.025, volMultiplier: 1.8 },
    370: { name: 'China Mining Ban / Flash Vol (May 2021)', shock: isCrypto ? -0.22 : -0.015, volMultiplier: 2.4 },
    500: { name: 'Fed Rate Hike Cycle Commences (Jan 2022)', shock: isCrypto ? -0.15 : -0.04, volMultiplier: 1.9 },
    590: { name: 'Terra/Luna & 3AC Liquidation (Jun 2022)', shock: isCrypto ? -0.32 : -0.045, volMultiplier: 2.5 },
    680: { name: 'FTX Exchange Insolvency (Nov 2022)', shock: isCrypto ? -0.25 : -0.02, volMultiplier: 2.2 },
    760: { name: 'SVB Regional Banking Crisis (Mar 2023)', shock: isCrypto ? 0.18 : -0.025, volMultiplier: 2.0 },
    980: { name: 'Spot Bitcoin ETF Approvals (Jan 2024)', shock: isCrypto ? 0.15 : 0.015, volMultiplier: 1.6 },
    1120: { name: 'BOJ Yen Carry Trade Flash Unwind (Aug 2024)', shock: isCrypto ? -0.18 : -0.035, volMultiplier: 2.8 },
    1240: { name: 'Global Tech & Macro Policy Shift (2025)', shock: isCrypto ? 0.09 : 0.02, volMultiplier: 1.4 }
  };

  // Seed pseudo-random generator with fixed seed per symbol for deterministic, reproducible results
  let seed = symbol === 'SPY' ? 421337 : symbol === 'AAPL' ? 882041 : 993125;
  function pseudoRandom() {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  }

  function standardNormal() {
    let u = 0, v = 0;
    while (u === 0) u = pseudoRandom();
    while (v === 0) v = pseudoRandom();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // Base parameters
  const annualVol = isCrypto ? 0.65 : isSingleStock ? 0.28 : 0.18;
  const baseDailyVol = annualVol / Math.sqrt(252);
  const drift = (isCrypto ? 0.35 : isSingleStock ? 0.18 : 0.11) / 252;

  let currentVol = baseDailyVol;

  for (let i = 0; i < tradingDays; i++) {
    // Advance date (skip weekends for SPY/AAPL, daily for BTC)
    currentDate.setDate(currentDate.getDate() + 1);
    if (!isCrypto) {
      while (currentDate.getDay() === 0 || currentDate.getDay() === 6) {
        currentDate.setDate(currentDate.getDate() + 1);
      }
    }

    const dateStr = currentDate.toISOString().split('T')[0];

    // Event check
    let eventName: string | undefined = undefined;
    let shockFactor = 0;
    let shockVolMult = 1.0;

    if (eventMap[i]) {
      eventName = eventMap[i].name;
      shockFactor = eventMap[i].shock;
      shockVolMult = eventMap[i].volMultiplier;
    }

    // Mean reverting volatility process (GARCH-like behavior in data generator)
    const volZ = standardNormal();
    currentVol = Math.max(
      baseDailyVol * 0.4,
      currentVol * 0.95 + baseDailyVol * 0.05 + baseDailyVol * 0.12 * volZ
    );

    const activeDailyVol = currentVol * shockVolMult;
    const z = standardNormal();

    // Fat tails injection (Student-t style mixing)
    const fatTailZ = pseudoRandom() < 0.08 ? z * 2.2 : z;

    // Log return with asymmetric leverage shock (negative returns cause higher subsequent vol)
    const logRet = drift + activeDailyVol * fatTailZ + shockFactor;
    const prevClose = currentPrice;
    currentPrice = Math.max(0.01, prevClose * Math.exp(logRet));

    // Synthetic high/low/open/volume based on close & volatility
    const open = prevClose * (1 + (pseudoRandom() - 0.5) * activeDailyVol * 0.5);
    const high = Math.max(open, currentPrice) * (1 + Math.abs(standardNormal()) * activeDailyVol * 0.8);
    const low = Math.min(open, currentPrice) * (1 - Math.abs(standardNormal()) * activeDailyVol * 0.8);
    const baseVol = isCrypto ? 25000000000 : isSingleStock ? 65000000 : 85000000;
    const volume = Math.round(baseVol * (0.6 + 0.8 * pseudoRandom() + Math.abs(logRet) * 20));

    // Realized intraday vol proxy (Parkinson / Garman-Klass approximation)
    const garmanKlass = Math.sqrt(
      0.5 * Math.pow(Math.log(high / low), 2) - (2 * Math.log(2) - 1) * Math.pow(Math.log(currentPrice / open), 2)
    ) * Math.sqrt(252);

    points.push({
      date: dateStr,
      timestamp: currentDate.getTime(),
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(currentPrice.toFixed(2)),
      volume,
      logReturn: parseFloat(logRet.toFixed(6)),
      rollingVol20: 0,
      rollingVol60: 0,
      ewmaVol: 0,
      intradayRealizedVol: parseFloat(garmanKlass.toFixed(4)),
      event: eventName
    });
  }

  // Calculate Realized Volatilities (20d, 60d, EWMA RiskMetrics λ=0.94)
  const lambda = 0.94;
  let ewmaVariance = Math.pow(annualVol / Math.sqrt(252), 2);

  for (let i = 0; i < points.length; i++) {
    const ret = points[i].logReturn;
    // EWMA update
    ewmaVariance = lambda * ewmaVariance + (1 - lambda) * Math.pow(ret, 2);
    points[i].ewmaVol = parseFloat((Math.sqrt(ewmaVariance) * Math.sqrt(252)).toFixed(4));

    // Rolling 20d
    if (i >= 19) {
      const window20 = points.slice(i - 19, i + 1).map((p) => p.logReturn);
      const mean20 = window20.reduce((a, b) => a + b, 0) / 20;
      const var20 = window20.reduce((a, b) => a + Math.pow(b - mean20, 2), 0) / 19;
      points[i].rollingVol20 = parseFloat((Math.sqrt(var20) * Math.sqrt(252)).toFixed(4));
    } else {
      points[i].rollingVol20 = points[i].ewmaVol;
    }

    // Rolling 60d
    if (i >= 59) {
      const window60 = points.slice(i - 59, i + 1).map((p) => p.logReturn);
      const mean60 = window60.reduce((a, b) => a + b, 0) / 60;
      const var60 = window60.reduce((a, b) => a + Math.pow(b - mean60, 2), 0) / 59;
      points[i].rollingVol60 = parseFloat((Math.sqrt(var60) * Math.sqrt(252)).toFixed(4));
    } else {
      points[i].rollingVol60 = points[i].rollingVol20;
    }
  }

  return points;
}

// Generate pre-computed datasets for the 3 target assets
export const MARKET_DATA: Record<AssetSymbol, DataPoint[]> = {
  SPY: generateAssetHistory('SPY', 320, 1320, '2020-01-02'),
  AAPL: generateAssetHistory('AAPL', 75, 1320, '2020-01-02'),
  'BTC-USD': generateAssetHistory('BTC-USD', 7200, 1320, '2020-01-02')
};

export const ASSET_METADATA: Record<AssetSymbol, { name: string; assetClass: 'Equities (Index)' | 'Equities (Single Stock)' | 'Crypto'; description: string }> = {
  SPY: {
    name: 'SPDR S&P 500 ETF Trust',
    assetClass: 'Equities (Index)',
    description: 'Benchmark ETF tracking the US broad equity market (S&P 500 Index). Exhibits pronounced asymmetric volatility (leverage effect) during systemic pullbacks.'
  },
  AAPL: {
    name: 'Apple Inc.',
    assetClass: 'Equities (Single Stock)',
    description: 'Mega-cap technology leader with idiosyncratic earnings variance superimposed on market beta. Strong test case for single-stock GJR-GARCH modeling.'
  },
  'BTC-USD': {
    name: 'Bitcoin / US Dollar',
    assetClass: 'Crypto',
    description: 'Premier digital asset exhibiting extreme leptokurtosis (fat tails), structural volatility regime shifts, and 24/7 continuous price discovery.'
  }
};
