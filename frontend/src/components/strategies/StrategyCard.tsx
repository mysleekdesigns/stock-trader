import { Play, Pause, AlertTriangle, TrendingUp, BarChart3, Target, Settings2 } from 'lucide-react'
import type { Strategy } from '../../api/client'
import { Card, CardContent } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(value)
}

interface StrategyCardProps {
  strategy: Strategy
  onToggle: (id: string, enabled: boolean) => void
  onConfigure?: (id: string) => void
}

export default function StrategyCard({ strategy, onToggle, onConfigure }: StrategyCardProps) {
  const statusColors = { running: 'text-up', stopped: 'text-muted-foreground', error: 'text-down' }
  const statusIcons = {
    running: <Play className="w-3.5 h-3.5" />,
    stopped: <Pause className="w-3.5 h-3.5" />,
    error: <AlertTriangle className="w-3.5 h-3.5" />,
  }

  return (
    <Card className="transition-all hover:border-primary/40">
      <CardContent>
        <div className="mb-4 flex items-start justify-between gap-3">
          <div className="space-y-1">
            <h3 className="font-display text-base font-semibold tracking-tight">{strategy.name}</h3>
            <p className="font-mono text-[0.7rem] uppercase tracking-wider text-muted-foreground">{strategy.type}</p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`flex items-center gap-1.5 rounded-[0.3rem] border border-border bg-muted px-2 py-1 font-mono text-[0.65rem] uppercase tracking-wider ${statusColors[strategy.status]}`}
            >
              {statusIcons[strategy.status]}
              {strategy.status}
            </span>
            {onConfigure && (
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={() => onConfigure(strategy.id)}
                aria-label="Configure strategy"
                title="Configure parameters"
              >
                <Settings2 />
              </Button>
            )}
            <Switch checked={strategy.enabled} onCheckedChange={(checked) => onToggle(strategy.id, checked)} />
          </div>
        </div>
        <p className="mb-5 line-clamp-2 text-xs leading-relaxed text-muted-foreground">{strategy.description}</p>
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-border bg-border">
          <div className="space-y-1.5 bg-card p-3">
            <div className="flex items-center gap-1.5">
              <TrendingUp className="h-3 w-3 text-muted-foreground" strokeWidth={1.75} />
              <span className="eyebrow !text-[0.6rem]">P&L</span>
            </div>
            <div className={`font-mono text-sm font-semibold tabular-nums ${strategy.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`}>
              {formatCurrency(strategy.pnl)}
            </div>
          </div>
          <div className="space-y-1.5 bg-card p-3">
            <div className="flex items-center gap-1.5">
              <Target className="h-3 w-3 text-muted-foreground" strokeWidth={1.75} />
              <span className="eyebrow !text-[0.6rem]">Win Rate</span>
            </div>
            <div className="font-mono text-sm font-semibold tabular-nums text-foreground">{(strategy.winRate * 100).toFixed(1)}%</div>
          </div>
          <div className="space-y-1.5 bg-card p-3">
            <div className="flex items-center gap-1.5">
              <BarChart3 className="h-3 w-3 text-muted-foreground" strokeWidth={1.75} />
              <span className="eyebrow !text-[0.6rem]">Sharpe</span>
            </div>
            <div className="font-mono text-sm font-semibold tabular-nums text-foreground">{strategy.sharpeRatio.toFixed(2)}</div>
          </div>
          <div className="space-y-1.5 bg-card p-3">
            <div className="flex items-center gap-1.5">
              <TrendingUp className="h-3 w-3 text-muted-foreground" strokeWidth={1.75} />
              <span className="eyebrow !text-[0.6rem]">Trades</span>
            </div>
            <div className="font-mono text-sm font-semibold tabular-nums text-foreground">{strategy.tradesCount}</div>
          </div>
        </div>
        {strategy.lastSignal && (
          <>
            <Separator className="mt-4" />
            <div className="flex items-center justify-between pt-3 text-xs">
              <span className="eyebrow">Last Signal</span>
              <span className="font-mono tabular-nums text-foreground">{strategy.lastSignal}</span>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
