import { useEffect, useState } from 'react'
import { getStrategies, toggleStrategy } from '../api/client'
import type { Strategy } from '../api/client'
import { useTradeStore } from '../stores/useTradeStore'
import StrategyCard from '../components/strategies/StrategyCard'
import StrategyConfig, { type StrategyParam, type ParamValue } from '../components/strategies/StrategyConfig'
import { Loader2 } from 'lucide-react'

/**
 * Tunable parameters surfaced for a strategy, derived from its type/name.
 * Mirrors the backend `config/strategies.yaml` parameter sets so the editor is
 * meaningful even before a dedicated params endpoint exists.
 */
function paramsForStrategy(strategy: Strategy): StrategyParam[] {
  const key = `${strategy.type} ${strategy.name}`.toLowerCase()
  if (key.includes('momentum')) {
    return [
      { key: 'fast_ma_period', value: 10, min: 2, max: 100, step: 1, description: 'Fast moving-average window' },
      { key: 'slow_ma_period', value: 50, min: 5, max: 400, step: 1, description: 'Slow moving-average window' },
      { key: 'adx_threshold', value: 25, min: 0, max: 60, step: 1, description: 'Min ADX to confirm trend strength' },
      { key: 'breakout_lookback', value: 20, min: 2, max: 200, step: 1, description: 'Bars for breakout detection' },
    ]
  }
  if (key.includes('reversion') || key.includes('arbitrage')) {
    return [
      { key: 'bb_period', value: 20, min: 5, max: 100, step: 1, description: 'Bollinger Band lookback' },
      { key: 'bb_std', value: 2.0, min: 0.5, max: 4, step: 0.1, description: 'Bollinger Band standard deviations' },
      { key: 'zscore_entry', value: 2.0, min: 0.5, max: 5, step: 0.1, description: 'Z-score to open a position' },
      { key: 'zscore_exit', value: 0.5, min: 0, max: 3, step: 0.1, description: 'Z-score to close a position' },
    ]
  }
  if (key.includes('ml') || key.includes('lstm') || key.includes('dl')) {
    return [
      { key: 'min_confidence', value: 0.55, min: 0.5, max: 0.95, step: 0.01, description: 'Minimum predicted-probability threshold' },
      { key: 'use_trend_filter', value: true, description: 'Require a confirming trend regime' },
    ]
  }
  // Generic risk/exit controls applicable to any rule-based strategy.
  return [
    { key: 'atr_stop_mult', value: 2.5, min: 0.5, max: 6, step: 0.1, description: 'Protective stop in ATR units' },
    { key: 'max_hold_days', value: 20, min: 1, max: 120, step: 1, description: 'Force-close after N trading days' },
    { key: 'allow_short', value: true, description: 'Permit short positions' },
  ]
}

export default function Strategies() {
  const { strategies, setStrategies, updateStrategy } = useTradeStore()
  const [loading, setLoading] = useState(true)
  const [configuringId, setConfiguringId] = useState<string | null>(null)
  const [savedParams, setSavedParams] = useState<Record<string, Record<string, ParamValue>>>({})

  useEffect(() => {
    async function load() {
      try {
        const data = await getStrategies()
        setStrategies(data)
      } catch {
        // API not available
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [setStrategies])

  // Close the config modal on Escape for keyboard users.
  useEffect(() => {
    if (!configuringId) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setConfiguringId(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [configuringId])

  async function handleToggle(id: string, enabled: boolean) {
    updateStrategy(id, { enabled })
    try {
      const updated = await toggleStrategy(id, enabled)
      updateStrategy(id, updated)
    } catch {
      updateStrategy(id, { enabled: !enabled })
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  const displayStrategies: Strategy[] =
    strategies.length > 0
      ? strategies
      : [
          {
            id: '1',
            name: 'Mean Reversion',
            description: 'Statistical arbitrage strategy using z-score for mean reversion signals on equity pairs.',
            type: 'Statistical Arbitrage',
            enabled: true,
            pnl: 12450.0,
            winRate: 0.62,
            tradesCount: 234,
            sharpeRatio: 1.85,
            maxDrawdown: 0.045,
            status: 'running' as const,
            lastSignal: 'BUY AAPL',
            lastSignalTime: new Date().toISOString(),
          },
          {
            id: '2',
            name: 'Momentum Alpha',
            description: 'Cross-sectional momentum strategy with ML-based signal generation and dynamic position sizing.',
            type: 'Momentum',
            enabled: true,
            pnl: 8920.0,
            winRate: 0.55,
            tradesCount: 156,
            sharpeRatio: 1.42,
            maxDrawdown: 0.067,
            status: 'running' as const,
          },
          {
            id: '3',
            name: 'LSTM Predictor',
            description: 'Deep learning based price prediction using LSTM networks with attention mechanism.',
            type: 'ML/DL',
            enabled: false,
            pnl: -1240.0,
            winRate: 0.48,
            tradesCount: 89,
            sharpeRatio: 0.65,
            maxDrawdown: 0.092,
            status: 'stopped' as const,
          },
        ]

  const configuring = displayStrategies.find((s) => s.id === configuringId) ?? null

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Strategies</h1>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>
            {displayStrategies.filter((s) => s.enabled).length} active /{' '}
            {displayStrategies.length} total
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {displayStrategies.map((strategy) => (
          <StrategyCard
            key={strategy.id}
            strategy={strategy}
            onToggle={handleToggle}
            onConfigure={setConfiguringId}
          />
        ))}
      </div>

      {configuring && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          onClick={() => setConfiguringId(null)}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-label={`Configure ${configuring.name}`}
            onClick={(e) => e.stopPropagation()}
          >
            <StrategyConfig
              strategyName={configuring.name}
              params={(paramsForStrategy(configuring)).map((p) => ({
                ...p,
                value: savedParams[configuring.id]?.[p.key] ?? p.value,
              }))}
              onClose={() => setConfiguringId(null)}
              onSave={(values) => {
                setSavedParams((prev) => ({ ...prev, [configuring.id]: values }))
                setConfiguringId(null)
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
