import { Card, CardContent } from '@/components/ui/card'

interface AllocationSlice { label: string; value: number; color: string }
interface AllocationDonutProps { slices: AllocationSlice[]; title?: string; size?: number }

const DEFAULT_COLORS = [
  '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6',
  '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#6366f1',
]

export default function AllocationDonut({ slices, title = 'Allocation', size = 180 }: AllocationDonutProps) {
  const total = slices.reduce((sum, s) => sum + s.value, 0)
  if (total === 0) {
    return (
      <Card>
        <h3 className="text-sm font-semibold">{title}</h3>
        <CardContent className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          No allocation data
        </CardContent>
      </Card>
    )
  }

  const radius = size / 2
  const innerRadius = radius * 0.6
  const center = radius
  let cumulativeAngle = -Math.PI / 2

  const paths = slices.map((slice, i) => {
    const angle = (slice.value / total) * 2 * Math.PI
    const startAngle = cumulativeAngle
    const endAngle = cumulativeAngle + angle
    cumulativeAngle = endAngle
    const x1 = center + radius * Math.cos(startAngle)
    const y1 = center + radius * Math.sin(startAngle)
    const x2 = center + radius * Math.cos(endAngle)
    const y2 = center + radius * Math.sin(endAngle)
    const ix1 = center + innerRadius * Math.cos(endAngle)
    const iy1 = center + innerRadius * Math.sin(endAngle)
    const ix2 = center + innerRadius * Math.cos(startAngle)
    const iy2 = center + innerRadius * Math.sin(startAngle)
    const largeArc = angle > Math.PI ? 1 : 0
    const color = slice.color || DEFAULT_COLORS[i % DEFAULT_COLORS.length]
    const d = `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} L ${ix1} ${iy1} A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${ix2} ${iy2} Z`
    return <path key={i} d={d} fill={color} className="hover:opacity-80 transition-opacity" />
  })

  return (
    <Card>
      <h3 className="text-sm font-semibold">{title}</h3>
      <CardContent className="flex items-center gap-6">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {paths}
          <text x={center} y={center - 8} textAnchor="middle"
            className="fill-foreground text-lg font-bold" style={{ fontSize: '16px' }}>
            ${(total / 1000).toFixed(0)}k
          </text>
          <text x={center} y={center + 12} textAnchor="middle"
            className="fill-muted-foreground" style={{ fontSize: '10px' }}>
            Total
          </text>
        </svg>
        <div className="flex flex-col gap-2 flex-1">
          {slices.map((slice, i) => (
            <div key={i} className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: slice.color || DEFAULT_COLORS[i % DEFAULT_COLORS.length] }} />
                <span className="text-muted-foreground">{slice.label}</span>
              </div>
              <span className="font-mono">{((slice.value / total) * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
