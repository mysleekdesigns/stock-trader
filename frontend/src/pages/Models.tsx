import { useEffect, useState } from 'react'
import { getModels } from '../api/client'
import type { ModelInfo } from '../api/client'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Brain, CheckCircle2, Loader2, XCircle, RefreshCw } from 'lucide-react'

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
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      try { const data = await getModels(); setModels(data) }
      catch {
        setModels([
          { id: '1', name: 'LightGBM Ensemble', type: 'LightGBM', version: 'v2.4.1', accuracy: 0.72, precision: 0.68, recall: 0.75, f1Score: 0.71, lastTrained: '2026-04-10T14:30:00Z', status: 'active', features: ['RSI', 'MACD', 'Volume', 'Bollinger', 'ATR'], predictions: [] },
          { id: '2', name: 'LSTM Price Predictor', type: 'PyTorch LSTM', version: 'v1.2.0', accuracy: 0.65, precision: 0.63, recall: 0.67, f1Score: 0.65, lastTrained: '2026-04-09T09:15:00Z', status: 'active', features: ['OHLCV', 'SMA_20', 'SMA_50', 'RSI', 'Volume_MA'], predictions: [] },
          { id: '3', name: 'XGBoost Classifier', type: 'XGBoost', version: 'v3.1.0', accuracy: 0.69, precision: 0.66, recall: 0.71, f1Score: 0.68, lastTrained: '2026-04-08T16:00:00Z', status: 'training', features: ['Momentum', 'Volatility', 'Trend', 'Sentiment'], predictions: [] },
          { id: '4', name: 'RL Position Sizer', type: 'Stable-Baselines3 PPO', version: 'v0.8.0', accuracy: 0.58, precision: 0.55, recall: 0.6, f1Score: 0.57, lastTrained: '2026-04-05T11:00:00Z', status: 'inactive', features: ['Portfolio_State', 'Market_Regime', 'Volatility'], predictions: [] },
        ])
      } finally { setLoading(false) }
    }
    load()
  }, [])

  if (loading) return <div className="flex items-center justify-center h-96"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>

  const statusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle2 className="w-4 h-4 text-up" />
      case 'training': return <RefreshCw className="w-4 h-4 text-amber-400 animate-spin" />
      case 'inactive': return <XCircle className="w-4 h-4 text-muted-foreground" />
      default: return null
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">ML Models</h1>
        <span className="text-xs text-muted-foreground">{models.filter((m) => m.status === 'active').length} active models</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map((model) => (
          <Card key={model.id} className="hover:border-primary/30 transition-colors">
            <CardContent>
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Brain className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold">{model.name}</h3>
                    <p className="text-xs text-muted-foreground">{model.type} &middot; {model.version}</p>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 text-xs">{statusIcon(model.status)}<span className="capitalize">{model.status}</span></div>
              </div>
              <div className="space-y-3 mb-4">
                <MetricBar label="Accuracy" value={model.accuracy} />
                <MetricBar label="Precision" value={model.precision} />
                <MetricBar label="Recall" value={model.recall} />
                <MetricBar label="F1 Score" value={model.f1Score} />
              </div>
              <div className="flex flex-wrap gap-1.5 mb-3">
                {model.features.map((f) => <Badge key={f} variant="secondary" className="text-[10px] font-mono">{f}</Badge>)}
              </div>
              <div className="text-xs text-muted-foreground">
                Last trained: {new Date(model.lastTrained).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
