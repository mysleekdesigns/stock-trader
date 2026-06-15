import { useState, useCallback } from 'react'
import {
  Search,
  Loader2,
  Activity,
  TrendingUp,
  BarChart3,
  Clock,
  Volume2,
  Settings2,
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import ORBChart from '../components/charts/ORBChart'
import { scanORB, getORBSignals, updateORBConfig } from '../api/client'
import type { ORBScanResult, ORBSignalResponse } from '../api/client'

const POPULAR_SYMBOLS = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'MSFT', 'AMZN', 'GOOGL']

export default function ORBScanner() {
  const [symbol, setSymbol] = useState('')
  const [loading, setLoading] = useState(false)
  const [scanResult, setScanResult] = useState<ORBScanResult | null>(null)
  const [signals, setSignals] = useState<ORBSignalResponse[]>([])
  const [error, setError] = useState<string | null>(null)
  const [showConfig, setShowConfig] = useState(false)
  const [volMult, setVolMult] = useState('1.5')
  const [cutoff, setCutoff] = useState('11:30')

  const handleScan = useCallback(
    async (sym?: string) => {
      const target = (sym || symbol).trim().toUpperCase()
      if (!target) return
      setLoading(true)
      setError(null)
      setSymbol(target)
      try {
        const result = await scanORB(target)
        setScanResult(result)
        const sigs = await getORBSignals()
        setSignals(sigs)
      } catch (err: any) {
        setError(err?.message || 'Scan failed')
      } finally {
        setLoading(false)
      }
    },
    [symbol],
  )

  const handleConfigSave = useCallback(async () => {
    try {
      await updateORBConfig({
        volume_multiplier: parseFloat(volMult),
        signal_cutoff: cutoff,
      })
      setShowConfig(false)
      if (scanResult) handleScan(scanResult.symbol)
    } catch {
      // ignore
    }
  }, [volMult, cutoff, scanResult, handleScan])

  const orbOverlay = scanResult
    ? {
        orHigh: scanResult.state.or_high,
        orLow: scanResult.state.or_low,
        vwap: scanResult.state.vwap,
        breakoutTime: scanResult.signal?.timestamp || null,
      }
    : { orHigh: null, orLow: null, vwap: null, breakoutTime: null }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="animate-rise space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="space-y-1.5">
            <div className="eyebrow">03 — ORB Scanner</div>
            <h1 className="font-display text-3xl font-semibold tracking-tight">ORB Scanner</h1>
            <p className="text-sm text-muted-foreground">
              Opening Range Breakout scanner with volume + VWAP confirmation
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowConfig(!showConfig)}
            >
              <Settings2 className="mr-1 h-4 w-4" />
              Config
            </Button>
          </div>
        </div>
        <div className="rule" />
      </div>

      {/* Config Panel */}
      {showConfig && (
        <Card className="animate-rise">
          <CardContent className="pt-4">
            <div className="flex items-end gap-4">
              <div>
                <label className="eyebrow mb-2 block">Volume Multiplier</label>
                <Input
                  value={volMult}
                  onChange={(e) => setVolMult(e.target.value)}
                  className="w-32 font-mono tabular-nums"
                  type="number"
                  step="0.1"
                  min="1"
                />
              </div>
              <div>
                <label className="eyebrow mb-2 block">Signal Cutoff (EST)</label>
                <Input
                  value={cutoff}
                  onChange={(e) => setCutoff(e.target.value)}
                  className="w-32 font-mono tabular-nums"
                  placeholder="11:30"
                />
              </div>
              <Button size="sm" onClick={handleConfigSave}>
                Save
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Search */}
      <Card className="animate-rise" style={{ animationDelay: '80ms' }}>
        <CardContent className="pt-4">
          <div className="flex items-center gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === 'Enter' && handleScan()}
                placeholder="Enter symbol (e.g. AAPL)"
                className="pl-9 font-mono tracking-wide"
              />
            </div>
            <Button onClick={() => handleScan()} disabled={loading || !symbol.trim()}>
              {loading ? (
                <Loader2 className="mr-1 h-4 w-4 animate-spin" />
              ) : (
                <Activity className="mr-1 h-4 w-4" />
              )}
              Scan
            </Button>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="eyebrow">Quick</span>
            {POPULAR_SYMBOLS.map((s) => (
              <button
                key={s}
                onClick={() => handleScan(s)}
                className="rounded-md border border-border bg-secondary/40 px-2 py-1 font-mono text-[0.7rem] tracking-wide text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
              >
                {s}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="animate-rise border-destructive/50 bg-destructive/5">
          <CardContent className="pt-4 font-mono text-sm text-destructive">{error}</CardContent>
        </Card>
      )}

      {/* Scan Result */}
      {scanResult && (
        <>
          {/* State Cards */}
          <div
            className="grid animate-rise grid-cols-2 gap-4 md:grid-cols-4"
            style={{ animationDelay: '160ms' }}
          >
            <Card className="transition-colors hover:border-primary/40">
              <CardContent className="pt-4">
                <div className="mb-2 flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-primary" />
                  <span className="eyebrow">OR High</span>
                </div>
                <div className="font-mono text-lg font-semibold tabular-nums text-foreground">
                  {scanResult.state.or_high?.toFixed(2) ?? '—'}
                </div>
              </CardContent>
            </Card>
            <Card className="transition-colors hover:border-primary/40">
              <CardContent className="pt-4">
                <div className="mb-2 flex items-center gap-2">
                  <Volume2 className="h-4 w-4" style={{ color: '#4fb6c4' }} />
                  <span className="eyebrow">OR Avg Volume</span>
                </div>
                <div className="font-mono text-lg font-semibold tabular-nums text-foreground">
                  {scanResult.state.or_avg_volume
                    ? Math.round(scanResult.state.or_avg_volume).toLocaleString()
                    : '—'}
                </div>
              </CardContent>
            </Card>
            <Card className="transition-colors hover:border-primary/40">
              <CardContent className="pt-4">
                <div className="mb-2 flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" style={{ color: '#4fb6c4' }} />
                  <span className="eyebrow">VWAP</span>
                </div>
                <div className="font-mono text-lg font-semibold tabular-nums text-foreground">
                  {scanResult.state.vwap?.toFixed(2) ?? '—'}
                </div>
              </CardContent>
            </Card>
            <Card className="transition-colors hover:border-primary/40">
              <CardContent className="pt-4">
                <div className="mb-2 flex items-center gap-2">
                  <Clock
                    className={`h-4 w-4 ${scanResult.state.breached ? 'text-up' : 'text-muted-foreground'}`}
                  />
                  <span className="eyebrow">Status</span>
                </div>
                <div className="font-mono text-lg font-semibold">
                  {scanResult.state.breached ? (
                    <span className="text-up">Breakout</span>
                  ) : scanResult.state.or_complete ? (
                    <span className="text-primary">Watching</span>
                  ) : (
                    <span className="text-muted-foreground">Building OR</span>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Signal Detail */}
          {scanResult.signal && (
            <Card
              className="animate-rise border-up/40 bg-up/8"
              style={{ animationDelay: '240ms' }}
            >
              <CardContent className="pt-4">
                <div className="mb-3 flex items-center gap-2">
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-up">
                    <span className="absolute inset-0 inline-flex rounded-full bg-up animate-pulse-ring" />
                  </span>
                  <span className="eyebrow !text-up">ORB Breakout Signal</span>
                </div>
                <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                  <div className="space-y-1.5">
                    <span className="eyebrow">Direction</span>
                    <div className="font-mono text-lg font-semibold tabular-nums text-up">
                      {scanResult.signal.direction.toUpperCase()}
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <span className="eyebrow">Strength</span>
                    <div className="font-mono text-lg font-semibold tabular-nums text-foreground">
                      {(scanResult.signal.strength * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <span className="eyebrow">Confidence</span>
                    <div className="font-mono text-lg font-semibold tabular-nums text-foreground">
                      {(scanResult.signal.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <span className="eyebrow">Volume Ratio</span>
                    <div className="font-mono text-lg font-semibold tabular-nums text-foreground">
                      {scanResult.signal.metadata?.volume_ratio ?? '—'}x
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Chart */}
          {scanResult.bars.length > 0 && (
            <div className="animate-rise" style={{ animationDelay: '320ms' }}>
              <ORBChart
                data={scanResult.bars}
                overlay={orbOverlay}
                symbol={scanResult.symbol}
                height={500}
              />
            </div>
          )}
        </>
      )}

      {/* Recent Signals Table */}
      {signals.length > 0 && (
        <Card className="animate-rise" style={{ animationDelay: '400ms' }}>
          <CardContent className="pt-4">
            <div className="eyebrow mb-4">Recent ORB Signals</div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="eyebrow py-2 pr-4 text-left font-normal">Symbol</th>
                    <th className="eyebrow py-2 pr-4 text-left font-normal">Direction</th>
                    <th className="eyebrow py-2 pr-4 text-right font-normal">Strength</th>
                    <th className="eyebrow py-2 pr-4 text-right font-normal">Confidence</th>
                    <th className="eyebrow py-2 pr-4 text-right font-normal">Vol Ratio</th>
                    <th className="eyebrow py-2 text-right font-normal">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {signals.map((sig, i) => (
                    <tr
                      key={`${sig.symbol}-${sig.timestamp}-${i}`}
                      className="cursor-pointer border-b border-border transition-colors last:border-0 hover:bg-muted/50"
                      onClick={() => handleScan(sig.symbol)}
                    >
                      <td className="py-2.5 pr-4 font-mono font-semibold tabular-nums text-foreground">
                        {sig.symbol}
                      </td>
                      <td className="py-2.5 pr-4">
                        <Badge variant={sig.direction.toLowerCase() === 'short' ? 'down' : 'up'}>
                          {sig.direction.toUpperCase()}
                        </Badge>
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                        {(sig.strength * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                        {(sig.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                        {sig.metadata?.volume_ratio ?? '—'}x
                      </td>
                      <td className="py-2.5 text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {new Date(sig.timestamp).toLocaleTimeString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
