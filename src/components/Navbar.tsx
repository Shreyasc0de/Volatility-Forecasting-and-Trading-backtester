import React from 'react';
import { AssetSymbol } from '../types/quant';
import {
  FolderGit2,
  LineChart,
  BrainCircuit,
  Layers,
  SlidersHorizontal,
  FileText,
  CheckSquare,
  TrendingUp,
  Activity
} from 'lucide-react';

interface Props {
  activePhase: number;
  onSelectPhase: (phase: number) => void;
  selectedAsset: AssetSymbol;
  onSelectAsset: (asset: AssetSymbol) => void;
  onOpenChecklist: () => void;
  completedTasks: number;
  totalTasks: number;
}

export const Navbar: React.FC<Props> = ({
  activePhase,
  onSelectPhase,
  selectedAsset,
  onSelectAsset,
  onOpenChecklist,
  completedTasks,
  totalTasks
}) => {
  const phases = [
    { id: 0, label: 'Phase 0: Scaffold', icon: FolderGit2, subtitle: 'Repo & Pipeline' },
    { id: 1, label: 'Phase 1: Data & EDA', icon: LineChart, subtitle: 'Realized Vol & QQ' },
    { id: 2, label: 'Phase 2: GARCH Suite', icon: BrainCircuit, subtitle: 'GJR, EGARCH, Loss' },
    { id: 3, label: 'Phase 3: Regimes', icon: Layers, subtitle: 'Hamilton 2-State' },
    { id: 4, label: 'Phase 4: Backtest', icon: SlidersHorizontal, subtitle: 'Walk-Forward Alpha' },
    { id: 5, label: 'Phase 5: Paper', icon: FileText, subtitle: 'Research Writeup' }
  ];

  return (
    <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-slate-950/85 backdrop-blur-xl">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/30 bg-cyan-500/10 text-cyan-300 shadow-[0_0_24px_rgba(34,211,238,0.2)]">
              <Activity className="h-6 w-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-base font-bold tracking-[0.16em] text-white">
                  QUANT_VOL::LAB
                </span>
                <span className="rounded-md border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-emerald-300">
                  v2.4 LIVE
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Volatility Forecasting & Regime-Aware Trading Backtester
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 rounded-xl border border-slate-800 bg-slate-900/80 p-1 shadow-inner shadow-slate-950/50">
              {(['SPY', 'AAPL', 'BTC-USD'] as AssetSymbol[]).map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => onSelectAsset(symbol)}
                  className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold font-mono transition-all ${
                    selectedAsset === symbol
                      ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-slate-950 shadow-lg shadow-cyan-500/20'
                      : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                  }`}
                >
                  <TrendingUp className="h-3.5 w-3.5" />
                  <span>{symbol}</span>
                </button>
              ))}
            </div>

            <button
              onClick={onOpenChecklist}
              className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/80 px-3.5 py-1.5 text-xs font-medium text-slate-200 transition-all hover:border-emerald-500/40 hover:bg-slate-800"
            >
              <CheckSquare className="h-4 w-4 text-emerald-400" />
              <span>Checklist</span>
              <span className="rounded-md bg-emerald-500/15 px-1.5 py-0.5 font-mono text-[11px] font-bold text-emerald-300">
                {completedTasks}/{totalTasks}
              </span>
            </button>
          </div>
        </div>

        <nav className="flex items-center gap-1 overflow-x-auto py-2.5 scrollbar-none">
          {phases.map((phase) => {
            const Icon = phase.icon;
            const isActive = activePhase === phase.id;
            return (
              <button
                key={phase.id}
                onClick={() => onSelectPhase(phase.id)}
                className={`flex flex-shrink-0 items-center gap-2.5 rounded-xl px-3.5 py-2 text-xs font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? 'border border-cyan-400/30 bg-cyan-500/10 text-cyan-300 shadow-[0_0_18px_rgba(34,211,238,0.12)]'
                    : 'border border-transparent text-slate-400 hover:bg-slate-800/70 hover:text-slate-200'
                }`}
              >
                <Icon className={`h-4 w-4 ${isActive ? 'text-cyan-300' : 'text-slate-400'}`} />
                <div className="text-left">
                  <div className={`font-semibold ${isActive ? 'text-white' : 'text-slate-300'}`}>
                    {phase.label}
                  </div>
                  <div className="text-[10px] text-slate-400 font-normal">{phase.subtitle}</div>
                </div>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
