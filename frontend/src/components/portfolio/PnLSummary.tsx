import { TrendingUp, TrendingDown } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value)
}

interface PnLCardProps {
  label: string
  value: number
}

function PnLCard({ label, value }: PnLCardProps) {
  const isPositive = value >= 0

  return (
    <Card>
      <CardContent className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">{label}</span>
          {isPositive ? (
            <TrendingUp className="w-4 h-4 text-up" />
          ) : (
            <TrendingDown className="w-4 h-4 text-down" />
          )}
        </div>
        <div
          className={`text-2xl font-bold font-mono ${
            isPositive ? 'pnl-positive' : 'pnl-negative'
          }`}
        >
          {isPositive ? '+' : ''}
          {formatCurrency(value)}
        </div>
      </CardContent>
    </Card>
  )
}

interface PnLSummaryProps {
  dailyPnl: number
  weeklyPnl: number
  monthlyPnl: number
}

export default function PnLSummary({ dailyPnl, weeklyPnl, monthlyPnl }: PnLSummaryProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <PnLCard label="Daily P&L" value={dailyPnl} />
      <PnLCard label="Weekly P&L" value={weeklyPnl} />
      <PnLCard label="Monthly P&L" value={monthlyPnl} />
    </div>
  )
}
