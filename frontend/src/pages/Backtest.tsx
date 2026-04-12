import { useState } from 'react'
import { runBacktest } from '../api/client'
import type { BacktestResult } from '../api/client'
import EquityCurve from '../components/charts/EquityCurve'
import DrawdownChart from '../components/charts/DrawdownChart'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import {
  Loader2, Play, TrendingUp, TrendingDown, BarChart3,
  Activity, Target, DollarSign, ArrowUpDown, Timer,
} from 'lucide-react'

/* ── helpers ── */

function fmt(v: number, decimals = 2) {
  return v.toFixed(decimals)
}

function pct(v: number) {
  return `${(v * 100).toFixed(2)}%`
}

function money(v: number) {
  return v.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
}

/** Add i days to a YYYY-MM-DD string using UTC to avoid DST issues. */
function addDaysUTC(dateStr: string, days: number): string {
  const ms = Date.parse(dateStr) + days * 86_400_000
  return new Date(ms).toISOString().slice(0, 10)
}

/** Convert a flat equity_curve array to {time, value}[] for charts. */
function equityToTimeSeries(
  equityCurve: number[],
  startDate: string,
): { time: string; value: number }[] {
  return equityCurve.map((value, i) => ({
    time: addDaysUTC(startDate, i),
    value,
  }))
}

/** Derive drawdown curve from equity curve. */
function equityToDrawdown(
  equityCurve: number[],
  startDate: string,
): { time: string; value: number }[] {
  let peak = equityCurve[0] ?? 0
  return equityCurve.map((value, i) => {
    if (value > peak) peak = value
    const dd = peak > 0 ? -(peak - value) / peak : 0
    return { time: addDaysUTC(startDate, i), value: dd }
  })
}

/* ── stat card ── */

function StatCard({ icon: Icon, label, value, positive }: {
  icon: React.ElementType; label: string; value: string; positive?: boolean
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
          <Icon className="w-4 h-4 text-primary" />
        </div>
        <div className="min-w-0">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</p>
          <p className={`text-lg font-bold font-mono leading-tight ${
            positive === undefined ? '' : positive ? 'text-up' : 'text-down'
          }`}>
            {value}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

/* ── main page ── */

export default function Backtest() {
  const [strategy, setStrategy] = useState('')
  const [startDate, setStartDate] = useState('2024-01-01')
  const [endDate, setEndDate] = useState('2024-12-31')
  const [symbols, setSymbols] = useState('AAPL,MSFT,GOOGL')
  const [initialCapital, setInitialCapital] = useState('100000')
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState('')
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleRun() {
    setRunning(true)
    setError(null)
    setResult(null)
    setProgress('Launching backtest...')
    try {
      const data = await runBacktest(
        {
          strategy,
          start_date: startDate,
          end_date: endDate,
          symbols: symbols.split(',').map((s) => s.trim()).filter(Boolean),
          initial_capital: parseFloat(initialCapital),
        },
        setProgress,
      )
      if (data.status === 'failed') {
        setError(data.metrics?.error?.toString() ?? 'Backtest failed')
      } else {
        setResult(data)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Backtest failed')
    } finally {
      setRunning(false)
      setProgress('')
    }
  }

  const m = result?.metrics ?? {}
  const equityCurveData = result ? equityToTimeSeries(result.equity_curve, startDate) : []
  const drawdownData = result ? equityToDrawdown(result.equity_curve, startDate) : []
  const trades = result?.trades ?? []

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Backtest</h1>

      {/* ── configuration panel ── */}
      <Card>
        <CardContent className="space-y-4">
          <h3 className="text-sm font-semibold">Configuration</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Strategy</Label>
              <Select value={strategy} onValueChange={setStrategy}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select strategy..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mean_reversion">Mean Reversion</SelectItem>
                  <SelectItem value="momentum">Momentum Alpha</SelectItem>
                  <SelectItem value="lstm">LSTM Predictor</SelectItem>
                  <SelectItem value="ensemble">Ensemble</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Start Date</Label>
              <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">End Date</Label>
              <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Symbols</Label>
              <Input
                type="text"
                value={symbols}
                onChange={(e) => setSymbols(e.target.value)}
                placeholder="AAPL,MSFT,GOOGL"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Initial Capital</Label>
              <Input
                type="number"
                value={initialCapital}
                onChange={(e) => setInitialCapital(e.target.value)}
                min={1000}
                step={1000}
              />
            </div>
            <div className="flex items-end">
              <Button onClick={handleRun} disabled={running || !strategy} className="w-full gap-2">
                {running ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {progress || 'Running...'}
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Run Backtest
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── error state ── */}
      {error && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-destructive shrink-0" />
            <p className="text-sm text-destructive">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* ── results ── */}
      {result && result.status === 'completed' && (
        <>
          {/* key metrics */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            <StatCard
              icon={TrendingUp}
              label="Total Return"
              value={pct(m.total_return ?? 0)}
              positive={(m.total_return ?? 0) >= 0}
            />
            <StatCard
              icon={BarChart3}
              label="Sharpe Ratio"
              value={fmt(m.sharpe_ratio ?? 0)}
              positive={(m.sharpe_ratio ?? 0) >= 1}
            />
            <StatCard
              icon={Activity}
              label="Sortino Ratio"
              value={fmt(m.sortino_ratio ?? 0)}
              positive={(m.sortino_ratio ?? 0) >= 1}
            />
            <StatCard
              icon={TrendingDown}
              label="Max Drawdown"
              value={pct(m.max_drawdown ?? 0)}
              positive={false}
            />
            <StatCard
              icon={Target}
              label="Win Rate"
              value={pct(m.win_rate ?? 0)}
              positive={(m.win_rate ?? 0) >= 0.5}
            />
            <StatCard
              icon={ArrowUpDown}
              label="Total Trades"
              value={String(m.trades_count ?? 0)}
            />
            <StatCard
              icon={DollarSign}
              label="Profit Factor"
              value={fmt(m.profit_factor ?? 0)}
              positive={(m.profit_factor ?? 0) >= 1}
            />
            <StatCard
              icon={Timer}
              label="Final Value"
              value={money(m.final_value ?? 0)}
              positive={(m.final_value ?? 0) >= (m.initial_capital ?? 0)}
            />
          </div>

          {/* charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <EquityCurve data={equityCurveData} title="Backtest Equity Curve" />
            <DrawdownChart data={drawdownData} />
          </div>

          {/* trades table */}
          {trades.length > 0 && (
            <Card>
              <CardContent className="space-y-3">
                <h3 className="text-sm font-semibold">
                  Trade Log
                  <span className="text-muted-foreground font-normal ml-2">({trades.length} trades)</span>
                </h3>
                <div className="max-h-80 overflow-y-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Side</TableHead>
                        <TableHead className="text-right">Qty</TableHead>
                        <TableHead className="text-right">Entry</TableHead>
                        <TableHead className="text-right">Exit</TableHead>
                        <TableHead className="text-right">P&L</TableHead>
                        <TableHead>Date</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {trades.slice(0, 50).map((t, i) => {
                        const pnl = Number(t.pnl ?? 0)
                        return (
                          <TableRow key={i}>
                            <TableCell className="font-mono font-medium">{String(t.symbol)}</TableCell>
                            <TableCell>
                              <Badge variant={t.side === 'buy' ? 'up' : 'down'} className="text-[10px] uppercase">
                                {String(t.side)}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-right font-mono">{String(t.quantity)}</TableCell>
                            <TableCell className="text-right font-mono">${Number(t.entry_price).toFixed(2)}</TableCell>
                            <TableCell className="text-right font-mono">${Number(t.exit_price).toFixed(2)}</TableCell>
                            <TableCell className={`text-right font-mono font-medium ${pnl >= 0 ? 'text-up' : 'text-down'}`}>
                              {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                            </TableCell>
                            <TableCell className="text-xs text-muted-foreground">
                              {t.date ? new Date(String(t.date)).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : '—'}
                            </TableCell>
                          </TableRow>
                        )
                      })}
                    </TableBody>
                  </Table>
                </div>
                {trades.length > 50 && (
                  <p className="text-xs text-muted-foreground text-center">
                    Showing 50 of {trades.length} trades
                  </p>
                )}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
