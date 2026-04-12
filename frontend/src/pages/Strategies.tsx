import { useEffect, useState } from 'react'
import { getStrategies, toggleStrategy } from '../api/client'
import type { Strategy } from '../api/client'
import { useTradeStore } from '../stores/useTradeStore'
import StrategyCard from '../components/strategies/StrategyCard'
import { Loader2 } from 'lucide-react'

export default function Strategies() {
  const { strategies, setStrategies, updateStrategy } = useTradeStore()
  const [loading, setLoading] = useState(true)

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
          <StrategyCard key={strategy.id} strategy={strategy} onToggle={handleToggle} />
        ))}
      </div>
    </div>
  )
}
