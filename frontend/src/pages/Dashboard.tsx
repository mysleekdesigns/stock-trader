import { useEffect, useState } from 'react'
import { getDashboard, getRiskMetrics } from '../api/client'
import type { DashboardData, RiskMetrics } from '../api/client'
import { useTradeStore } from '../stores/useTradeStore'
import EquityCurve from '../components/charts/EquityCurve'
import PnLSummary from '../components/portfolio/PnLSummary'
import PositionsTable from '../components/portfolio/PositionsTable'
import AllocationDonut from '../components/portfolio/AllocationDonut'
import HeatMap from '../components/charts/HeatMap'
import RiskGauge from '../components/risk/RiskGauge'
import { ExposureSummary } from '../components/risk/ExposureBar'
import { Loader2 } from 'lucide-react'

const ALLOC_COLORS = [
  '#e8ae49', '#4fb6c4', '#2fcf8e', '#f4615a', '#c9923c',
  '#7aa2c4', '#d9b86a', '#8a8275', '#5ec5a8', '#e0905a',
]

const DEFAULT_UNIVERSE = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA']

/**
 * Build a symmetric correlation matrix for the given symbols. Uses a stable
 * hash of each symbol pair so the view is deterministic until a real
 * correlation endpoint is wired in (mirrors the dashboard's API-fallback data).
 */
function correlationMatrix(symbols: string[]): number[][] {
  const hash = (s: string): number => {
    let h = 0
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0
    return h
  }
  return symbols.map((a, i) =>
    symbols.map((b, j) => {
      if (i === j) return 1
      const pair = i < j ? a + b : b + a
      // Map a stable hash into roughly [-0.4, 0.95] — markets skew positively correlated.
      return Math.round((((hash(pair) % 1000) / 1000) * 1.35 - 0.4) * 100) / 100
    }),
  )
}

export default function Dashboard() {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null)
  const [risk, setRisk] = useState<RiskMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const { updatePortfolio, setPositions, setEquityCurve, portfolio } = useTradeStore()

  useEffect(() => {
    async function load() {
      try {
        const [dashData, riskData] = await Promise.all([getDashboard(), getRiskMetrics()])
        setDashboard(dashData)
        setRisk(riskData)

        updatePortfolio({
          totalValue: dashData.portfolio.totalValue,
          cash: dashData.portfolio.cash,
          unrealizedPnl: dashData.portfolio.unrealizedPnl,
          drawdown: dashData.portfolio.drawdown,
          dailyPnl: dashData.portfolio.dailyPnl,
        })
        setPositions(dashData.positions)
        setEquityCurve(dashData.equityCurve)
      } catch {
        // API not available, use empty state
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [updatePortfolio, setPositions, setEquityCurve])

  if (loading) {
    return (
      <div className="panel flex h-96 flex-col items-center justify-center gap-3">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="eyebrow">Loading Terminal</span>
      </div>
    )
  }

  const positions = dashboard?.positions ?? []
  const equityCurve = dashboard?.equityCurve ?? []

  const allocationSlices = positions.map((p, i) => ({
    label: p.symbol,
    value: p.marketValue,
    color: ALLOC_COLORS[i % ALLOC_COLORS.length],
  }))

  // Risk gauge: convert drawdown to 0-100 scale (5% drawdown = 50 risk)
  const riskValue = risk ? Math.min(risk.currentDrawdown * 20 * 100, 100) : 0

  const heatmapSymbols = (positions.length > 0 ? positions.map((p) => p.symbol) : DEFAULT_UNIVERSE).slice(0, 8)

  return (
    <div className="space-y-8">
      <div className="animate-rise space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="space-y-1.5">
            <div className="eyebrow">01 — Overview</div>
            <h1 className="font-display text-3xl font-semibold tracking-tight">Dashboard</h1>
            <p className="text-sm text-muted-foreground">
              Live portfolio, risk, and correlation readout.
            </p>
          </div>
          <div className="flex gap-3">
            <span className="font-mono text-xs tabular-nums text-muted-foreground">
              {new Date().toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </span>
          </div>
        </div>
        <div className="rule" />
      </div>

      <div className="animate-rise" style={{ animationDelay: '80ms' }}>
        <PnLSummary
          dailyPnl={portfolio.dailyPnl}
          weeklyPnl={portfolio.weeklyPnl}
          monthlyPnl={portfolio.monthlyPnl}
        />
      </div>

      <div className="animate-rise grid grid-cols-1 gap-6 lg:grid-cols-3" style={{ animationDelay: '160ms' }}>
        <div className="lg:col-span-2">
          <EquityCurve data={equityCurve} />
        </div>
        <div className="space-y-6">
          <RiskGauge value={riskValue} />
          {risk && (
            <ExposureSummary
              longExposure={risk.exposure.long}
              shortExposure={risk.exposure.short}
              netExposure={risk.exposure.net}
              grossExposure={risk.exposure.gross}
              totalValue={portfolio.totalValue || 1}
            />
          )}
        </div>
      </div>

      <div className="animate-rise grid grid-cols-1 gap-6 lg:grid-cols-3" style={{ animationDelay: '240ms' }}>
        <div className="lg:col-span-2">
          <PositionsTable positions={positions} />
        </div>
        <AllocationDonut slices={allocationSlices} />
      </div>

      <div className="animate-rise grid grid-cols-1 gap-6 lg:grid-cols-2" style={{ animationDelay: '320ms' }}>
        <HeatMap labels={heatmapSymbols} matrix={correlationMatrix(heatmapSymbols)} title="Position Correlation" />
      </div>
    </div>
  )
}
