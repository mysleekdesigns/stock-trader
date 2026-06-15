import { Card, CardContent } from '@/components/ui/card'

interface HeatMapProps {
  /** Row/column labels (symbols or sectors). Square matrix is assumed. */
  labels: string[]
  /** `matrix[i][j]` is the value (e.g. correlation in [-1, 1]) for labels[i] vs labels[j]. */
  matrix: number[][]
  title?: string
  /** Lower/upper bounds of the colour scale. Defaults to a [-1, 1] correlation scale. */
  min?: number
  max?: number
}

/**
 * Correlation / sector heat map.
 *
 * Renders a labelled grid where each cell is colour-coded on a diverging scale:
 * green (`--color-up`) for high/positive values, red (`--color-down`) for
 * low/negative values, fading through the muted surface near the midpoint.
 */
export default function HeatMap({ labels, matrix, title = 'Correlation', min = -1, max = 1 }: HeatMapProps) {
  const n = labels.length
  const hasData = n > 0 && matrix.length === n && matrix.every((row) => row.length === n)

  if (!hasData) {
    return (
      <Card>
        <div className="space-y-1">
          <div className="eyebrow">Matrix</div>
          <h3 className="font-display text-sm font-semibold tracking-tight">{title}</h3>
        </div>
        <CardContent className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          No correlation data
        </CardContent>
      </Card>
    )
  }

  const mid = (min + max) / 2

  // Diverging colour scale: red → transparent → green, centred on `mid`.
  function cellStyle(value: number): { background: string; color: string } {
    const clamped = Math.max(min, Math.min(max, value))
    if (clamped >= mid) {
      const t = (clamped - mid) / (max - mid || 1) // 0..1
      return { background: `rgba(47, 207, 142, ${0.12 + t * 0.78})`, color: t > 0.55 ? '#fff' : 'var(--foreground)' }
    }
    const t = (mid - clamped) / (mid - min || 1) // 0..1
    return { background: `rgba(244, 97, 90, ${0.12 + t * 0.78})`, color: t > 0.55 ? '#fff' : 'var(--foreground)' }
  }

  return (
    <Card>
      <div className="space-y-1">
        <div className="eyebrow">Matrix</div>
        <h3 className="font-display text-sm font-semibold tracking-tight">{title}</h3>
      </div>
      <CardContent className="overflow-x-auto">
        <table className="border-separate border-spacing-1 text-xs font-mono">
          <thead>
            <tr>
              <th className="w-12" />
              {labels.map((label, j) => (
                <th key={`h-${label}-${j}`} className="px-1 pb-1 text-muted-foreground font-medium text-center min-w-[3rem]">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((rowLabel, i) => (
              <tr key={`r-${rowLabel}-${i}`}>
                <td className="pr-2 text-right text-muted-foreground font-medium whitespace-nowrap">{rowLabel}</td>
                {labels.map((colLabel, j) => {
                  const value = matrix[i][j]
                  const { background, color } = cellStyle(value)
                  return (
                    <td
                      key={`c-${i}-${j}`}
                      title={`${rowLabel} · ${colLabel}: ${value.toFixed(2)}`}
                      className="h-9 min-w-[3rem] rounded-sm text-center tabular-nums transition-opacity hover:opacity-80"
                      style={{ background, color }}
                    >
                      {value.toFixed(2)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="flex items-center gap-2 mt-3 font-mono text-[10px] tabular-nums text-muted-foreground">
          <span>{min.toFixed(1)}</span>
          <div
            className="h-2 flex-1 rounded-full"
            style={{ background: 'linear-gradient(to right, #f4615a, rgba(138,130,117,0.25), #2fcf8e)' }}
          />
          <span>{max.toFixed(1)}</span>
        </div>
      </CardContent>
    </Card>
  )
}
