import { useState, useCallback } from 'react'
import {
  Search,
  Loader2,
  Activity,
  TrendingUp,
  BarChart3,
  Clock,
  Crosshair,
  Volume2,
  Settings2,
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <Crosshair className="w-5 h-5 text-primary" />
            ORB Scanner
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Opening Range Breakout scanner with volume + VWAP confirmation
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowConfig(!showConfig)}
        >
          <Settings2 className="w-4 h-4 mr-1" />
          Config
        </Button>
      </div>

      {/* Config Panel */}
      {showConfig && (
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-end gap-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">
                  Volume Multiplier
                </label>
                <Input
                  value={volMult}
                  onChange={(e) => setVolMult(e.target.value)}
                  className="w-32"
                  type="number"
                  step="0.1"
                  min="1"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">
                  Signal Cutoff (EST)
                </label>
                <Input
                  value={cutoff}
                  onChange={(e) => setCutoff(e.target.value)}
                  className="w-32"
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
      <Card>
        <CardContent className="pt-4">
          <div className="flex items-center gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === 'Enter' && handleScan()}
                placeholder="Enter symbol (e.g. AAPL)"
                className="pl-9"
              />
            </div>
            <Button onClick={() => handleScan()} disabled={loading || !symbol.trim()}>
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin mr-1" />
              ) : (
                <Activity className="w-4 h-4 mr-1" />
              )}
              Scan
            </Button>
          </div>
          <div className="flex items-center gap-2 mt-3">
            <span className="text-xs text-muted-foreground">Quick:</span>
            {POPULAR_SYMBOLS.map((s) => (
              <Button
                key={s}
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={() => handleScan(s)}
              >
                {s}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-4 text-sm text-destructive">{error}</CardContent>
        </Card>
      )}

      {/* Scan Result */}
      {scanResult && (
        <>
          {/* State Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="w-4 h-4 text-orange-500" />
                  <span className="text-xs text-muted-foreground">OR High</span>
                </div>
                <div className="text-lg font-mono font-semibold">
                  {scanResult.state.or_high?.toFixed(2) ?? '—'}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-2 mb-1">
                  <Volume2 className="w-4 h-4 text-blue-500" />
                  <span className="text-xs text-muted-foreground">OR Avg Volume</span>
                </div>
                <div className="text-lg font-mono font-semibold">
                  {scanResult.state.or_avg_volume
                    ? Math.round(scanResult.state.or_avg_volume).toLocaleString()
                    : '—'}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-2 mb-1">
                  <BarChart3 className="w-4 h-4 text-indigo-500" />
                  <span className="text-xs text-muted-foreground">VWAP</span>
                </div>
                <div className="text-lg font-mono font-semibold">
                  {scanResult.state.vwap?.toFixed(2) ?? '—'}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="w-4 h-4 text-green-500" />
                  <span className="text-xs text-muted-foreground">Status</span>
                </div>
                <div className="text-lg font-semibold">
                  {scanResult.state.breached ? (
                    <span className="text-up">Breakout</span>
                  ) : scanResult.state.or_complete ? (
                    <span className="text-yellow-500">Watching</span>
                  ) : (
                    <span className="text-muted-foreground">Building OR</span>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Signal Detail */}
          {scanResult.signal && (
            <Card className="border-up/50 bg-up/5">
              <CardContent className="pt-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full bg-up animate-pulse" />
                  <span className="text-sm font-semibold text-up">ORB Breakout Signal</span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-xs text-muted-foreground">Direction</span>
                    <div className="font-mono font-semibold">
                      {scanResult.signal.direction.toUpperCase()}
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-muted-foreground">Strength</span>
                    <div className="font-mono font-semibold">
                      {(scanResult.signal.strength * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-muted-foreground">Confidence</span>
                    <div className="font-mono font-semibold">
                      {(scanResult.signal.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-muted-foreground">Volume Ratio</span>
                    <div className="font-mono font-semibold">
                      {scanResult.signal.metadata?.volume_ratio ?? '—'}x
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Chart */}
          {scanResult.bars.length > 0 && (
            <ORBChart
              data={scanResult.bars}
              overlay={orbOverlay}
              symbol={scanResult.symbol}
              height={500}
            />
          )}
        </>
      )}

      {/* Recent Signals Table */}
      {signals.length > 0 && (
        <Card>
          <CardContent className="pt-4">
            <h3 className="text-sm font-semibold mb-3">Recent ORB Signals</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-xs text-muted-foreground">
                    <th className="text-left py-2 pr-4">Symbol</th>
                    <th className="text-left py-2 pr-4">Direction</th>
                    <th className="text-right py-2 pr-4">Strength</th>
                    <th className="text-right py-2 pr-4">Confidence</th>
                    <th className="text-right py-2 pr-4">Vol Ratio</th>
                    <th className="text-right py-2">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {signals.map((sig, i) => (
                    <tr
                      key={`${sig.symbol}-${sig.timestamp}-${i}`}
                      className="border-b last:border-0 hover:bg-muted/50 cursor-pointer"
                      onClick={() => handleScan(sig.symbol)}
                    >
                      <td className="py-2 pr-4 font-mono font-semibold">{sig.symbol}</td>
                      <td className="py-2 pr-4">
                        <span className="text-up font-medium">
                          {sig.direction.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-2 pr-4 text-right font-mono">
                        {(sig.strength * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 pr-4 text-right font-mono">
                        {(sig.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 pr-4 text-right font-mono">
                        {sig.metadata?.volume_ratio ?? '—'}x
                      </td>
                      <td className="py-2 text-right text-muted-foreground text-xs">
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
