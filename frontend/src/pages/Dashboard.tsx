import { useEffect, useState } from 'react'
import { getDashboard, getRiskMetrics } from '../api/client'
import type { DashboardData, RiskMetrics } from '../api/client'
import { useTradeStore } from '../stores/useTradeStore'
import EquityCurve from '../components/charts/EquityCurve'
import PnLSummary from '../components/portfolio/PnLSummary'
import PositionsTable from '../components/portfolio/PositionsTable'
import AllocationDonut from '../components/portfolio/AllocationDonut'
import RiskGauge from '../components/risk/RiskGauge'
import { ExposureSummary } from '../components/risk/ExposureBar'
import { Loader2 } from 'lucide-react'

const ALLOC_COLORS = [
  '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6',
  '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#6366f1',
]

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
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Dashboard</h1>
        <span className="text-xs text-muted-foreground font-mono">
          {new Date().toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          })}
        </span>
      </div>

      <PnLSummary
        dailyPnl={portfolio.dailyPnl}
        weeklyPnl={portfolio.weeklyPnl}
        monthlyPnl={portfolio.monthlyPnl}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <PositionsTable positions={positions} />
        </div>
        <AllocationDonut slices={allocationSlices} />
      </div>
    </div>
  )
}
