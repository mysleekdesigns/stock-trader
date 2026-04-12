import { Card, CardContent } from '@/components/ui/card'

interface ExposureBarProps {
  label: string
  value: number
  maxValue: number
  color?: string
  showPercent?: boolean
}

export default function ExposureBar({
  label, value, maxValue, color = '#3b82f6', showPercent = true,
}: ExposureBarProps) {
  const pct = maxValue > 0 ? Math.min((Math.abs(value) / maxValue) * 100, 100) : 0

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">
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
      <h3 className="text-sm font-semibold">Exposure</h3>
      <CardContent className="space-y-4">
        <ExposureBar label="Long" value={longExposure} maxValue={totalValue} color="#22c55e" />
        <ExposureBar label="Short" value={shortExposure} maxValue={totalValue} color="#ef4444" />
        <ExposureBar label="Net" value={netExposure} maxValue={totalValue} color="#3b82f6" />
        <ExposureBar label="Gross" value={grossExposure} maxValue={totalValue} color="#f59e0b" />
      </CardContent>
    </Card>
  )
}
