import { Card, CardContent } from '@/components/ui/card'

interface RiskGaugeProps {
  value: number
  label?: string
  size?: number
}

export default function RiskGauge({ value, label = 'Risk Level', size = 160 }: RiskGaugeProps) {
  const clampedValue = Math.max(0, Math.min(100, value))
  const center = size / 2
  const radius = center - 16
  const strokeWidth = 12

  const startAngle = -225
  const endAngle = 45
  const totalAngle = endAngle - startAngle
  const valueAngle = startAngle + (clampedValue / 100) * totalAngle

  const toRad = (deg: number) => (deg * Math.PI) / 180

  const arcPath = (start: number, end: number) => {
    const x1 = center + radius * Math.cos(toRad(start))
    const y1 = center + radius * Math.sin(toRad(start))
    const x2 = center + radius * Math.cos(toRad(end))
    const y2 = center + radius * Math.sin(toRad(end))
    const largeArc = end - start > 180 ? 1 : 0
    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`
  }

  let riskColor = '#2fcf8e'
  let riskLabel = 'Low'
  if (clampedValue > 70) {
    riskColor = '#f4615a'
    riskLabel = 'High'
  } else if (clampedValue > 40) {
    riskColor = '#e8ae49'
    riskLabel = 'Medium'
  }

  return (
    <Card className="items-center">
      <div className="space-y-1 self-start">
        <div className="eyebrow">Risk</div>
        <h3 className="font-display text-sm font-semibold tracking-tight">{label}</h3>
      </div>
      <CardContent>
        <svg width={size} height={size * 0.7} viewBox={`0 0 ${size} ${size * 0.75}`}>
          <path
            d={arcPath(startAngle, endAngle)}
            fill="none"
            stroke="var(--border)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {clampedValue > 0 && (
            <path
              d={arcPath(startAngle, valueAngle)}
              fill="none"
              stroke={riskColor}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
            />
          )}
          {(() => {
            const nx = center + (radius - 24) * Math.cos(toRad(valueAngle))
            const ny = center + (radius - 24) * Math.sin(toRad(valueAngle))
            return (
              <line x1={center} y1={center} x2={nx} y2={ny}
                stroke={riskColor} strokeWidth={2} strokeLinecap="round" />
            )
          })()}
          <circle cx={center} cy={center} r={4} fill={riskColor} />
          <text x={center} y={center + 24} textAnchor="middle"
            className="fill-foreground font-semibold" style={{ fontSize: '20px', fontFamily: 'IBM Plex Mono, monospace' }}>
            {clampedValue.toFixed(0)}
          </text>
          <text x={center} y={center + 40} textAnchor="middle"
            style={{ fontSize: '10px', fill: riskColor, fontFamily: 'IBM Plex Mono, monospace', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            {riskLabel}
          </text>
        </svg>
      </CardContent>
    </Card>
  )
}
