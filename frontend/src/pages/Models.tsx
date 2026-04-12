import { useEffect, useState } from 'react'
import { getModels } from '../api/client'
import type { ModelInfo } from '../api/client'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import {
  Brain, CheckCircle2, Loader2, XCircle, RefreshCw,
  BarChart3, Activity, TrendingUp, Clock, LayoutGrid, List,
} from 'lucide-react'

/* ── placeholder data (shown when API returns empty or errors) ── */
const PLACEHOLDER_MODELS: ModelInfo[] = [
  {
    id: '1', name: 'LightGBM Ensemble', type: 'LightGBM', version: 'v2.4.1',
    accuracy: 0.72, precision: 0.68, recall: 0.75, f1Score: 0.71,
    lastTrained: '2026-04-10T14:30:00Z', status: 'active',
    features: ['RSI', 'MACD', 'Volume', 'Bollinger', 'ATR'],
    predictions: [],
  },
  {
    id: '2', name: 'LSTM Price Predictor', type: 'PyTorch LSTM', version: 'v1.2.0',
    accuracy: 0.65, precision: 0.63, recall: 0.67, f1Score: 0.65,
    lastTrained: '2026-04-09T09:15:00Z', status: 'active',
    features: ['OHLCV', 'SMA_20', 'SMA_50', 'RSI', 'Volume_MA'],
    predictions: [],
  },
  {
    id: '3', name: 'XGBoost Classifier', type: 'XGBoost', version: 'v3.1.0',
    accuracy: 0.69, precision: 0.66, recall: 0.71, f1Score: 0.68,
    lastTrained: '2026-04-08T16:00:00Z', status: 'training',
    features: ['Momentum', 'Volatility', 'Trend', 'Sentiment'],
    predictions: [],
  },
  {
    id: '4', name: 'RL Position Sizer', type: 'Stable-Baselines3 PPO', version: 'v0.8.0',
    accuracy: 0.58, precision: 0.55, recall: 0.6, f1Score: 0.57,
    lastTrained: '2026-04-05T11:00:00Z', status: 'inactive',
    features: ['Portfolio_State', 'Market_Regime', 'Volatility'],
    predictions: [],
  },
  {
    id: '5', name: 'TFT Forecaster', type: 'PyTorch TFT', version: 'v1.0.3',
    accuracy: 0.67, precision: 0.64, recall: 0.69, f1Score: 0.66,
    lastTrained: '2026-04-11T08:45:00Z', status: 'active',
    features: ['OHLCV', 'Macro_Indicators', 'Sector_Momentum', 'VIX'],
    predictions: [],
  },
]

/* ── small helpers ── */

function MetricBar({ label, value, max = 1 }: { label: string; value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100)
  const color = pct >= 80 ? '#22c55e' : pct >= 60 ? '#f59e0b' : '#ef4444'
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'active':
      return <CheckCircle2 className="w-4 h-4 text-up" />
    case 'training':
      return <RefreshCw className="w-4 h-4 text-amber-400 animate-spin" />
    case 'inactive':
      return <XCircle className="w-4 h-4 text-muted-foreground" />
    default:
      return null
  }
}

function StatusBadge({ status }: { status: string }) {
  const variant = status === 'active' ? 'up' : status === 'training' ? 'neutral' : 'down'
  return (
    <Badge variant={variant} className="text-[10px] capitalize">
      <StatusIcon status={status} />
      {status}
    </Badge>
  )
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

/* ── summary stat card ── */

function StatCard({ icon: Icon, label, value, sub }: {
  icon: React.ElementType; label: string; value: string | number; sub?: string
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
          <Icon className="w-4.5 h-4.5 text-primary" />
        </div>
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground">{label}</p>
          <p className="text-lg font-semibold leading-tight">{value}</p>
          {sub && <p className="text-[10px] text-muted-foreground">{sub}</p>}
        </div>
      </CardContent>
    </Card>
  )
}

/* ── feature importance bar (horizontal) ── */

function FeatureChip({ name, count, maxCount }: { name: string; count: number; maxCount: number }) {
  const pct = (count / maxCount) * 100
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 truncate text-muted-foreground font-mono">{name}</span>
      <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-primary/70 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-muted-foreground w-5 text-right">{count}</span>
    </div>
  )
}

/* ── main page ── */

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [view, setView] = useState<'grid' | 'table'>('grid')

  useEffect(() => {
    async function load() {
      try {
        const data = await getModels()
        setModels(data)
      } catch {
        // API not available — leave empty so fallback kicks in below
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  // Use placeholder data when API returns nothing (same pattern as Strategies page)
  const displayModels = models.length > 0 ? models : PLACEHOLDER_MODELS

  /* ── derived stats ── */
  const activeCount = displayModels.filter((m) => m.status === 'active').length
  const trainingCount = displayModels.filter((m) => m.status === 'training').length
  const avgAccuracy = displayModels.reduce((s, m) => s + m.accuracy, 0) / displayModels.length
  const bestModel = [...displayModels].sort((a, b) => b.f1Score - a.f1Score)[0]

  // Feature frequency across all models
  const featureFreq: Record<string, number> = {}
  displayModels.forEach((m) => m.features.forEach((f) => {
    featureFreq[f] = (featureFreq[f] || 0) + 1
  }))
  const sortedFeatures = Object.entries(featureFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
  const maxFeatureCount = sortedFeatures[0]?.[1] ?? 1

  return (
    <div className="space-y-6">
      {/* ── header ── */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">ML Models</h1>
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground">
            {activeCount} active / {displayModels.length} total
          </span>
          <div className="flex rounded-md border border-border overflow-hidden">
            <button
              onClick={() => setView('grid')}
              className={`p-1.5 transition-colors ${view === 'grid' ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
            >
              <LayoutGrid className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setView('table')}
              className={`p-1.5 transition-colors ${view === 'table' ? 'bg-primary/10 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
            >
              <List className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* ── summary stats ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard icon={Brain} label="Total Models" value={displayModels.length} sub={`${trainingCount} training`} />
        <StatCard icon={Activity} label="Active" value={activeCount} sub="deployed" />
        <StatCard icon={BarChart3} label="Avg Accuracy" value={`${(avgAccuracy * 100).toFixed(1)}%`} />
        <StatCard icon={TrendingUp} label="Best F1" value={`${(bestModel.f1Score * 100).toFixed(1)}%`} sub={bestModel.name} />
      </div>

      {/* ── grid view ── */}
      {view === 'grid' && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {displayModels.map((model) => (
            <Card key={model.id} className="hover:border-primary/30 transition-colors">
              <CardContent className="space-y-4">
                {/* header row */}
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                      <Brain className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold">{model.name}</h3>
                      <p className="text-xs text-muted-foreground">
                        {model.type} &middot; {model.version}
                      </p>
                    </div>
                  </div>
                  <StatusBadge status={model.status} />
                </div>

                {/* metrics */}
                <div className="space-y-2.5">
                  <MetricBar label="Accuracy" value={model.accuracy} />
                  <MetricBar label="Precision" value={model.precision} />
                  <MetricBar label="Recall" value={model.recall} />
                  <MetricBar label="F1 Score" value={model.f1Score} />
                </div>

                {/* features */}
                <div className="flex flex-wrap gap-1.5">
                  {model.features.map((f) => (
                    <Badge key={f} variant="secondary" className="text-[10px] font-mono">{f}</Badge>
                  ))}
                </div>

                {/* footer */}
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Clock className="w-3 h-3" />
                  Last trained: {formatDate(model.lastTrained)}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* ── table view ── */}
      {view === 'table' && (
        <Card>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Version</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead className="text-right">Precision</TableHead>
                  <TableHead className="text-right">Recall</TableHead>
                  <TableHead className="text-right">F1</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Last Trained</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {displayModels.map((model) => (
                  <TableRow key={model.id}>
                    <TableCell className="font-medium">{model.name}</TableCell>
                    <TableCell className="text-muted-foreground text-xs">{model.type}</TableCell>
                    <TableCell className="font-mono text-xs">{model.version}</TableCell>
                    <TableCell className="text-right font-mono">{(model.accuracy * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-right font-mono">{(model.precision * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-right font-mono">{(model.recall * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-right font-mono">{(model.f1Score * 100).toFixed(1)}%</TableCell>
                    <TableCell><StatusBadge status={model.status} /></TableCell>
                    <TableCell className="text-xs text-muted-foreground">{formatDate(model.lastTrained)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* ── bottom panels: feature frequency + training timeline ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* feature frequency */}
        <Card>
          <CardContent className="space-y-3">
            <h3 className="text-sm font-semibold">Feature Usage Across Models</h3>
            <div className="space-y-2">
              {sortedFeatures.map(([name, count]) => (
                <FeatureChip key={name} name={name} count={count} maxCount={maxFeatureCount} />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* training timeline */}
        <Card>
          <CardContent className="space-y-3">
            <h3 className="text-sm font-semibold">Training Timeline</h3>
            <div className="space-y-3">
              {[...displayModels]
                .sort((a, b) => new Date(b.lastTrained).getTime() - new Date(a.lastTrained).getTime())
                .map((model) => (
                  <div key={model.id} className="flex items-center gap-3">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium truncate">{model.name}</span>
                        <StatusBadge status={model.status} />
                      </div>
                      <p className="text-[10px] text-muted-foreground font-mono">
                        {formatDate(model.lastTrained)} &middot; {model.version}
                      </p>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
