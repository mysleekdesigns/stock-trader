import { Play, Pause, AlertTriangle, TrendingUp, BarChart3, Target } from 'lucide-react'
import type { Strategy } from '../../api/client'
import { Card, CardContent } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(value)
}

interface StrategyCardProps { strategy: Strategy; onToggle: (id: string, enabled: boolean) => void }

export default function StrategyCard({ strategy, onToggle }: StrategyCardProps) {
  const statusColors = { running: 'text-up', stopped: 'text-muted-foreground', error: 'text-down' }
  const statusIcons = {
    running: <Play className="w-3.5 h-3.5" />,
    stopped: <Pause className="w-3.5 h-3.5" />,
    error: <AlertTriangle className="w-3.5 h-3.5" />,
  }

  return (
    <Card className="hover:border-primary/30 transition-colors">
      <CardContent>
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold">{strategy.name}</h3>
            <p className="text-xs text-muted-foreground mt-0.5">{strategy.type}</p>
          </div>
          <div className="flex items-center gap-3">
            <span className={`flex items-center gap-1 text-xs ${statusColors[strategy.status]}`}>
              {statusIcons[strategy.status]}
              {strategy.status}
            </span>
            <Switch checked={strategy.enabled} onCheckedChange={(checked) => onToggle(strategy.id, checked)} />
          </div>
        </div>
        <p className="text-xs text-muted-foreground mb-4 line-clamp-2">{strategy.description}</p>
        <div className="grid grid-cols-2 gap-3">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-3.5 h-3.5 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">P&L</div>
              <div className={`text-sm font-mono font-semibold ${strategy.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`}>
                {formatCurrency(strategy.pnl)}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Target className="w-3.5 h-3.5 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Win Rate</div>
              <div className="text-sm font-mono font-semibold">{(strategy.winRate * 100).toFixed(1)}%</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-3.5 h-3.5 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Sharpe</div>
              <div className="text-sm font-mono font-semibold">{strategy.sharpeRatio.toFixed(2)}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-3.5 h-3.5 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Trades</div>
              <div className="text-sm font-mono font-semibold">{strategy.tradesCount}</div>
            </div>
          </div>
        </div>
        {strategy.lastSignal && (
          <>
            <Separator className="mt-4" />
            <div className="flex items-center justify-between text-xs pt-3">
              <span className="text-muted-foreground">Last Signal</span>
              <span className="font-mono">{strategy.lastSignal}</span>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
