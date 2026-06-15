import { Card, CardContent } from '@/components/ui/card'

interface ExposureBarProps {
  label: string
  value: number
  maxValue: number
  color?: string
  showPercent?: boolean
}

export default function ExposureBar({
  label, value, maxValue, color = '#4fb6c4', showPercent = true,
}: ExposureBarProps) {
  const pct = maxValue > 0 ? Math.min((Math.abs(value) / maxValue) * 100, 100) : 0

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono tabular-nums">
          {showPercent ? `${pct.toFixed(1)}%` : value.toLocaleString()}
        </span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

interface ExposureSummaryProps {
  longExposure: number
  shortExposure: number
  netExposure: number
  grossExposure: number
  totalValue: number
}

export function ExposureSummary({
  longExposure, shortExposure, netExposure, grossExposure, totalValue,
}: ExposureSummaryProps) {
  return (
    <Card>
      <div className="space-y-1">
        <div className="eyebrow">Risk</div>
        <h3 className="font-display text-sm font-semibold tracking-tight">Exposure</h3>
      </div>
      <CardContent className="space-y-4">
        <ExposureBar label="Long" value={longExposure} maxValue={totalValue} color="#2fcf8e" />
        <ExposureBar label="Short" value={shortExposure} maxValue={totalValue} color="#f4615a" />
        <ExposureBar label="Net" value={netExposure} maxValue={totalValue} color="#4fb6c4" />
        <ExposureBar label="Gross" value={grossExposure} maxValue={totalValue} color="#e8ae49" />
      </CardContent>
    </Card>
  )
}
