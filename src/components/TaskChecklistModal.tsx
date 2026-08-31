import React, { useState } from 'react';
import { CheckCircle2, Circle, X, Award, ExternalLink } from 'lucide-react';

interface TaskItem {
  id: string;
  phase: string;
  title: string;
  details: string;
  completed: boolean;
}

const INITIAL_TASKS: TaskItem[] = [
  // Phase 0
  { id: 'p0-1', phase: 'Phase 0: Setup', title: 'Create repository & folder architecture', details: 'Setup data/raw, data/processed, notebooks, src/ with modular python files', completed: true },
  { id: 'p0-2', phase: 'Phase 0: Setup', title: 'Set up virtual environment & core dependencies', details: 'pandas, numpy, yfinance, python-binance/ccxt, arch, statsmodels, hmmlearn, scipy', completed: true },
  { id: 'p0-3', phase: 'Phase 0: Setup', title: 'Scaffold core scripts & README', details: 'data_pull.py, vol_models.py, regime_models.py, backtest_integration.py, requirements.txt', completed: true },
  
  // Phase 1
  { id: 'p1-1', phase: 'Phase 1: Data Acquisition', title: 'Pull SPY daily OHLCV (5+ years)', details: 'Clean trading days, corporate actions, compute daily log returns', completed: true },
  { id: 'p1-2', phase: 'Phase 1: Data Acquisition', title: 'Pull AAPL daily OHLCV (5+ years)', details: 'Single stock high-beta mega-cap tech benchmark', completed: true },
  { id: 'p1-3', phase: 'Phase 1: Data Acquisition', title: 'Pull BTC-USD 24/7 crypto series', details: 'Digital asset with continuous market pricing and extreme volatility spikes', completed: true },
  { id: 'p1-4', phase: 'Phase 1: Data Acquisition', title: 'Compute realized volatility metrics', details: 'Rolling 20-day, rolling 60-day, EWMA RiskMetrics (lambda=0.94), intraday Parkinson/Garman-Klass vol', completed: true },
  { id: 'p1-5', phase: 'Phase 1: Data Acquisition', title: 'Sanity-check time series & calendar alignment', details: 'Price series, return series, rolling vol series, align trading calendars', completed: true },

  // Phase 2
  { id: 'p2-1', phase: 'Phase 2: GARCH Modeling', title: 'Establish baseline vol forecasts', details: 'Naive lagged vol, rolling historical averages (20d & 60d), EWMA RiskMetrics', completed: true },
  { id: 'p2-2', phase: 'Phase 2: GARCH Modeling', title: 'Test for ARCH effects (ARCH-LM Test)', details: 'Confirm autoregressive conditional heteroskedasticity (p < 0.001) prior to model fitting', completed: true },
  { id: 'p2-3', phase: 'Phase 2: GARCH Modeling', title: 'Check fat tails & excess kurtosis (QQ Plots)', details: 'Quantile-Quantile plots against normal distribution, kurtosis & Jarque-Bera tests', completed: true },
  { id: 'p2-4', phase: 'Phase 2: GARCH Modeling', title: 'Fit GARCH(1,1), GJR-GARCH, and EGARCH', details: 'Estimate omega, alpha, beta, gamma leverage parameters; calculate persistence and half-life', completed: true },
  { id: 'p2-5', phase: 'Phase 2: GARCH Modeling', title: 'Out-of-sample evaluation (RMSE, QLIKE, AIC/BIC)', details: 'Time-ordered 70/30 train-test split, evaluate loss functions, identify best model per asset', completed: true },

  // Phase 3
  { id: 'p3-1', phase: 'Phase 3: Regime-Switching', title: 'Fit 2-State Markov-Switching Hamilton model', details: 'Calm vs Turbulent latent states on returns and conditional volatility series', completed: true },
  { id: 'p3-2', phase: 'Phase 3: Regime-Switching', title: 'Compute filtered & smoothed regime probabilities', details: 'Overlay on price series; align with COVID 2020, 2022 Fed hikes, 2023 SVB, 2024 unwinds', completed: true },
  { id: 'p3-3', phase: 'Phase 3: Regime-Switching', title: 'Analyze regime persistence & transition matrix', details: 'Expected state durations (E[D]=1/(1-p_ii)), transition matrix P_00, P_11', completed: true },
  { id: 'p3-4', phase: 'Phase 3: Regime-Switching', title: 'Lead-Lag cross-correlation check', details: 'Determine whether regime switches precede or coincide with drawdown events', completed: true },
  { id: 'p3-5', phase: 'Phase 3: Regime-Switching', title: 'Compare regime dynamics across SPY vs AAPL vs BTC', details: 'Contrast crypto regime velocity with equity index stability', completed: true },

  // Phase 4
  { id: 'p4-1', phase: 'Phase 4: Backtest Engine', title: 'Build modular backtesting engine with walk-forward validation', details: 'No lookahead bias, rolling step execution, realistic slippage & transaction costs', completed: true },
  { id: 'p4-2', phase: 'Phase 4: Backtest Engine', title: 'Implement Inverse-Volatility Position Sizing', details: 'w_t = min(max_leverage, target_vol / sigma_forecast)', completed: true },
  { id: 'p4-3', phase: 'Phase 4: Backtest Engine', title: 'Implement Regime-Gated Exposure Control', details: 'De-risk position when P(Turbulent) > threshold', completed: true },
  { id: 'p4-4', phase: 'Phase 4: Backtest Engine', title: 'Evaluate performance metrics & equity curves', details: 'Sharpe, Sortino, Max Drawdown, Calmar ratio, turnover, win rate, profit factor', completed: true },

  // Phase 5
  { id: 'p5-1', phase: 'Phase 5: Research & Writeup', title: 'Comprehensive Quantitative Research Paper', details: 'Abstract, methodology, GARCH econometrics, Markov dynamics, backtest results, limitations', completed: true },
  { id: 'p5-2', phase: 'Phase 5: Research & Writeup', title: 'Code workspace & reproducible documentation', details: 'Documented python scripts, parameter explanations, clean reproducible pipeline', completed: true }
];

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelectPhase: (phaseIndex: number) => void;
}

export const TaskChecklistModal: React.FC<Props> = ({ isOpen, onClose, onSelectPhase }) => {
  const [tasks, setTasks] = useState<TaskItem[]>(INITIAL_TASKS);
  const [filterPhase, setFilterPhase] = useState<string>('All');

  if (!isOpen) return null;

  const toggleTask = (id: string) => {
    setTasks((prev) =>
      prev.map((t) => (t.id === id ? { ...t, completed: !t.completed } : t))
    );
  };

  const completedCount = tasks.filter((t) => t.completed).length;
  const progressPercent = Math.round((completedCount / tasks.length) * 100);

  const phases = ['All', 'Phase 0: Setup', 'Phase 1: Data Acquisition', 'Phase 2: GARCH Modeling', 'Phase 3: Regime-Switching', 'Phase 4: Backtest Engine', 'Phase 5: Research & Writeup'];

  const filteredTasks = filterPhase === 'All' ? tasks : tasks.filter((t) => t.phase === filterPhase);

  const getPhaseIndex = (phaseStr: string) => {
    if (phaseStr.includes('Phase 0')) return 0;
    if (phaseStr.includes('Phase 1')) return 1;
    if (phaseStr.includes('Phase 2')) return 2;
    if (phaseStr.includes('Phase 3')) return 3;
    if (phaseStr.includes('Phase 4')) return 4;
    return 5;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-xs p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col text-slate-100 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/80">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Project 1 Checklist & Milestones</h2>
              <p className="text-xs text-slate-400">Volatility Forecasting + Trading Signal Backtest Research Pipeline</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Progress Bar */}
        <div className="px-6 py-3 bg-slate-950/40 border-b border-slate-800 flex items-center justify-between gap-4">
          <div className="flex-1">
            <div className="flex justify-between text-xs font-medium mb-1.5">
              <span className="text-slate-300">Milestone Completion</span>
              <span className="text-emerald-400 font-semibold">{completedCount} of {tasks.length} tasks ({progressPercent}%)</span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div
                className="bg-emerald-500 h-full transition-all duration-300 rounded-full"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        </div>

        {/* Phase Filter Tabs */}
        <div className="flex items-center gap-1 px-6 py-2.5 overflow-x-auto border-b border-slate-800/80 bg-slate-900/40 text-xs">
          {phases.map((p) => (
            <button
              key={p}
              onClick={() => setFilterPhase(p)}
              className={`px-3 py-1.5 rounded-lg whitespace-nowrap font-medium transition-colors ${
                filterPhase === p
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              {p}
            </button>
          ))}
        </div>

        {/* Task List */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-2.5 divide-y divide-slate-800/40">
          {filteredTasks.map((t) => (
            <div
              key={t.id}
              className="pt-2.5 first:pt-0 flex items-start justify-between gap-3 group hover:bg-slate-800/30 p-2 rounded-lg transition-colors"
            >
              <div className="flex items-start gap-3 flex-1 cursor-pointer" onClick={() => toggleTask(t.id)}>
                <button className="mt-0.5 text-slate-400 hover:text-emerald-400 transition-colors">
                  {t.completed ? (
                    <CheckCircle2 className="w-5 h-5 text-emerald-400 fill-emerald-400/20" />
                  ) : (
                    <Circle className="w-5 h-5 text-slate-500" />
                  )}
                </button>
                <div>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium ${t.completed ? 'text-slate-200 line-through opacity-85' : 'text-white'}`}>
                      {t.title}
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono">
                      {t.phase.split(':')[0]}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">{t.details}</p>
                </div>
              </div>

              <button
                onClick={() => {
                  onSelectPhase(getPhaseIndex(t.phase));
                  onClose();
                }}
                className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 opacity-0 group-hover:opacity-100 transition-opacity px-2 py-1 rounded-md bg-blue-500/10 hover:bg-blue-500/20 whitespace-nowrap"
              >
                <span>Jump to View</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-6 py-3.5 bg-slate-950/60 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <span>All mathematical derivations, statistical loss tests & walk-forward overlays are calibrated.</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
