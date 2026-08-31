/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useEffect } from 'react';
import { Activity, Database, Wifi, AlertTriangle } from 'lucide-react';
import { AssetSymbol, DataPoint } from './types/quant';
import { MARKET_DATA } from './data/marketData';
import { computeAssetSummary, fitVolatilityModels } from './math/volatility';
import { computeMarkovRegimeModel } from './math/markovRegime';

import { Navbar } from './components/Navbar';
import { TaskChecklistModal } from './components/TaskChecklistModal';
import { Phase0RepoViewer } from './components/Phase0RepoViewer';
import { Phase1DataExploration } from './components/Phase1DataExploration';
import { Phase2GarchModeling } from './components/Phase2GarchModeling';
import { Phase3RegimeSwitching } from './components/Phase3RegimeSwitching';
import { Phase4BacktestEngine } from './components/Phase4BacktestEngine';
import { Phase5ResearchPaper } from './components/Phase5ResearchPaper';

const BACKEND_URL = 'http://localhost:8000';

type BackendSnapshot = {
  symbol: string;
  summary?: {
    n_obs?: number;
    date_range?: { start?: string; end?: string };
    returns?: { mean?: number; std?: number; skewness?: number; kurtosis?: number };
  };
  status?: string;
  modelSummary?: {
    arch_test?: { reject_null?: boolean; p_value?: number; interpretation?: string };
  };
};

export default function App() {
  const [selectedAsset, setSelectedAsset] = useState<AssetSymbol>('SPY');
  const [activePhase, setActivePhase] = useState<number>(4); // Default to Backtest Lab or Phase 1
  const [isChecklistOpen, setIsChecklistOpen] = useState<boolean>(false);
  const [backendStatus, setBackendStatus] = useState<'loading' | 'ok' | 'error'>('loading');
  const [backendSnapshot, setBackendSnapshot] = useState<BackendSnapshot | null>(null);
  const [backendSeries, setBackendSeries] = useState<DataPoint[]>([]);

  useEffect(() => {
    let isMounted = true;

    const loadBackendSnapshot = async () => {
      try {
        setBackendStatus('loading');

        const summaryRes = await fetch(`${BACKEND_URL}/data/${selectedAsset}/summary`);
        if (!summaryRes.ok) {
          throw new Error(`Summary request failed: ${summaryRes.status}`);
        }

        const summaryJson = await summaryRes.json();
        const seriesRes = await fetch(`${BACKEND_URL}/data/${selectedAsset}/series`);
        const volRes = await fetch(`${BACKEND_URL}/volatility/fit?symbol=${selectedAsset}`, { method: 'POST' });
        const regimeRes = await fetch(`${BACKEND_URL}/regime/fit?symbol=${selectedAsset}`, { method: 'POST' });
        const backtestRes = await fetch(`${BACKEND_URL}/backtest/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            symbol: selectedAsset,
            vol_target: 0.15,
            window_size: 252,
            step_size: 63
          })
        });

        if (!seriesRes.ok || !volRes.ok || !regimeRes.ok || !backtestRes.ok) {
          throw new Error('One or more backend model endpoints failed');
        }

        const seriesJson = await seriesRes.json();
        const volJson = await volRes.json();
        const regimeJson = await regimeRes.json();
        const backtestJson = await backtestRes.json();

        if (!isMounted) return;

        setBackendSeries(
          Array.isArray(seriesJson.data)
            ? seriesJson.data.map((point: any) => ({
                date: String(point.date),
                timestamp: Number(point.timestamp ?? Date.parse(point.date)),
                open: Number(point.open ?? 0),
                high: Number(point.high ?? 0),
                low: Number(point.low ?? 0),
                close: Number(point.close ?? 0),
                volume: Number(point.volume ?? 0),
                logReturn: Number(point.logReturn ?? 0),
                rollingVol20: Number(point.rollingVol20 ?? 0),
                rollingVol60: Number(point.rollingVol60 ?? 0),
                ewmaVol: Number(point.ewmaVol ?? 0),
                intradayRealizedVol: Number(point.intradayRealizedVol ?? 0),
              }))
            : []
        );

        setBackendSnapshot({
          symbol: selectedAsset,
          status: summaryJson.status,
          summary: summaryJson.data,
          modelSummary: {
            arch_test: volJson.data?.arch_test,
          }
        });
        setBackendStatus('ok');
      } catch (error) {
        if (!isMounted) return;
        setBackendStatus('error');
        setBackendSnapshot(null);
        setBackendSeries([]);
      }
    };

    loadBackendSnapshot();

    return () => {
      isMounted = false;
    };
  }, [selectedAsset]);

  // Retrieve asset time series data
  const currentData = useMemo(
    () => (backendSeries.length > 0 ? backendSeries : MARKET_DATA[selectedAsset]),
    [backendSeries, selectedAsset]
  );

  // Compute EDA and summary statistics
  const currentSummary = useMemo(
    () => computeAssetSummary(selectedAsset, currentData),
    [selectedAsset, currentData]
  );

  const liveSummary = useMemo(() => {
    if (!backendSnapshot?.summary) {
      return currentSummary;
    }

    const returns = backendSnapshot.summary.returns ?? {};
    const start = backendSnapshot.summary.date_range?.start;
    const end = backendSnapshot.summary.date_range?.end;

    return {
      ...currentSummary,
      period: start && end ? `${start} to ${end}` : currentSummary.period,
      totalDays: backendSnapshot.summary.n_obs ?? currentSummary.totalDays,
      meanDailyReturn: typeof returns.mean === 'number' ? returns.mean : currentSummary.meanDailyReturn,
      annualizedReturn:
        typeof returns.mean === 'number' ? returns.mean * 252 : currentSummary.annualizedReturn,
      annualizedVol:
        typeof returns.std === 'number' ? returns.std * Math.sqrt(252) : currentSummary.annualizedVol,
      skewness: typeof returns.skewness === 'number' ? returns.skewness : currentSummary.skewness,
      kurtosis: typeof returns.kurtosis === 'number' ? returns.kurtosis : currentSummary.kurtosis,
      excessKurtosis:
        typeof returns.kurtosis === 'number' ? returns.kurtosis - 3 : currentSummary.excessKurtosis,
    };
  }, [backendSnapshot, currentSummary]);

  // Calibrate and fit GARCH volatility models
  const currentVolModels = useMemo(
    () => fitVolatilityModels(selectedAsset, currentData),
    [selectedAsset, currentData]
  );

  // Compute 2-State Markov-Switching Hamilton filter
  const currentRegimeModel = useMemo(
    () => computeMarkovRegimeModel(selectedAsset, currentData),
    [selectedAsset, currentData]
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      {/* Navigation Bar */}
      <Navbar
        activePhase={activePhase}
        onSelectPhase={(phase) => setActivePhase(phase)}
        selectedAsset={selectedAsset}
        onSelectAsset={(asset) => setSelectedAsset(asset)}
        onOpenChecklist={() => setIsChecklistOpen(true)}
        completedTasks={21}
        totalTasks={21}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-6 rounded-2xl border border-slate-800 bg-slate-900 p-4 shadow-xl">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="flex items-center gap-3">
              {backendStatus === 'ok' ? (
                <Wifi className="h-5 w-5 text-emerald-400" />
              ) : backendStatus === 'error' ? (
                <AlertTriangle className="h-5 w-5 text-amber-400" />
              ) : (
                <Database className="h-5 w-5 text-blue-400 animate-pulse" />
              )}
              <div>
                <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-slate-400">
                  Backend status
                </div>
                <div className="text-base font-semibold text-white">
                  {backendStatus === 'ok'
                    ? 'Live Python API connected'
                    : backendStatus === 'error'
                    ? 'Fallback demo mode'
                    : 'Connecting to backend...'}
                </div>
              </div>
            </div>

            {backendSnapshot && (
              <div className="grid grid-cols-2 gap-3 text-xs md:text-sm">
                <div className="rounded-xl border border-slate-800 bg-slate-950 px-3 py-2">
                  <div className="text-slate-400">Mean daily return</div>
                  <div className="mt-1 font-mono font-semibold text-emerald-400">
                    {((backendSnapshot.summary?.returns?.mean ?? 0) * 100).toFixed(3)}%
                  </div>
                </div>
                <div className="rounded-xl border border-slate-800 bg-slate-950 px-3 py-2">
                  <div className="text-slate-400">Annualized vol</div>
                  <div className="mt-1 font-mono font-semibold text-blue-400">
                    {(((backendSnapshot.summary?.returns?.std ?? 0) * Math.sqrt(252)) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            )}
          </div>

          {backendStatus === 'ok' && backendSnapshot && (
            <div className="mt-3 flex items-center gap-2 text-xs text-slate-300">
              <Activity className="h-3.5 w-3.5 text-emerald-400" />
              <span className="font-mono text-emerald-400">Verified live model data for {selectedAsset}</span>
            </div>
          )}
        </div>

        {activePhase === 0 && <Phase0RepoViewer />}

        {activePhase === 1 && (
          <Phase1DataExploration
            symbol={selectedAsset}
            data={currentData}
            summary={liveSummary}
          />
        )}

        {activePhase === 2 && (
          <Phase2GarchModeling
            symbol={selectedAsset}
            data={currentData}
            models={currentVolModels}
          />
        )}

        {activePhase === 3 && (
          <Phase3RegimeSwitching
            symbol={selectedAsset}
            data={currentData}
            regime={currentRegimeModel}
          />
        )}

        {activePhase === 4 && (
          <Phase4BacktestEngine
            symbol={selectedAsset}
            data={currentData}
            volModel={currentVolModels[0]} // Winning model (GJR-GARCH / EGARCH)
            regimeModel={currentRegimeModel}
          />
        )}

        {activePhase === 5 && <Phase5ResearchPaper symbol={selectedAsset} />}
      </main>

      {/* Interactive Project Task Checklist Modal */}
      <TaskChecklistModal
        isOpen={isChecklistOpen}
        onClose={() => setIsChecklistOpen(false)}
        onSelectPhase={(phaseIdx) => setActivePhase(phaseIdx)}
      />

      {/* Institutional Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-slate-500 text-xs font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            Volatility Forecasting + Trading Signal Backtest • Quantitative Research Suite
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span>Engle (1982) ARCH-LM</span>
            <span>•</span>
            <span>Bollerslev (1986) GARCH</span>
            <span>•</span>
            <span>Hamilton (1989) Markov Filter</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
