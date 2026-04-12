import { useState } from 'react'
import { runBacktest } from '../api/client'
import type { BacktestResult } from '../api/client'
import EquityCurve from '../components/charts/EquityCurve'
import DrawdownChart from '../components/charts/DrawdownChart'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Loader2, Play } from 'lucide-react'

export default function Backtest() {
  const [strategyId, setStrategyId] = useState('')
  const [startDate, setStartDate] = useState('2024-01-01')
  const [endDate, setEndDate] = useState('2024-12-31')
  const [symbols, setSymbols] = useState('AAPL,MSFT,GOOGL')
  const [initialCapital, setInitialCapital] = useState('100000')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleRun() {
    setRunning(true); setError(null); setResult(null)
    try {
      const data = await runBacktest({
        strategyId, startDate, endDate,
        symbols: symbols.split(',').map((s) => s.trim()),
        initialCapital: parseFloat(initialCapital),
      })
      setResult(data)
    } catch (err) { setError(err instanceof Error ? err.message : 'Backtest failed') }
    finally { setRunning(false) }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Backtest</h1>
      <Card>
        <h3 className="text-sm font-semibold">Configuration</h3>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Strategy</Label>
              <Select value={strategyId} onValueChange={setStrategyId}>
                <SelectTrigger><SelectValue placeholder="Select strategy..." /></SelectTrigger>
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
              <Input type="text" value={symbols} onChange={(e) => setSymbols(e.target.value)} placeholder="AAPL,MSFT,GOOGL" />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Initial Capital</Label>
              <Input type="number" value={initialCapital} onChange={(e) => setInitialCapital(e.target.value)} />
            </div>
            <div className="flex items-end">
              <Button onClick={handleRun} disabled={running || !strategyId} className="w-full">
                {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {running ? 'Running...' : 'Run Backtest'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
      {error && <Card className="border-destructive/30 bg-destructive/10"><p className="text-sm text-destructive">{error}</p></Card>}
      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {[
              { label: 'Total Return', value: `${(result.totalReturn * 100).toFixed(2)}%`, positive: result.totalReturn >= 0 },
              { label: 'Sharpe Ratio', value: result.sharpeRatio.toFixed(2), positive: result.sharpeRatio >= 1 },
              { label: 'Max Drawdown', value: `${(result.maxDrawdown * 100).toFixed(2)}%`, positive: false },
              { label: 'Win Rate', value: `${(result.winRate * 100).toFixed(1)}%`, positive: result.winRate >= 0.5 },
              { label: 'Total Trades', value: result.tradesCount.toString(), positive: true },
              { label: 'Profit Factor', value: (result.metrics?.profitFactor ?? 0).toFixed(2), positive: (result.metrics?.profitFactor ?? 0) >= 1 },
            ].map(({ label, value, positive }) => (
              <Card key={label}><CardContent>
                <div className="text-xs text-muted-foreground">{label}</div>
                <div className={`text-lg font-bold font-mono mt-1 ${positive ? 'text-up' : 'text-down'}`}>{value}</div>
              </CardContent></Card>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <EquityCurve data={result.equityCurve} title="Backtest Equity" />
            <DrawdownChart data={result.drawdownCurve} />
          </div>
        </>
      )}
    </div>
  )
}
