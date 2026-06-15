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
        <h3 className="text-sm font-semibold">{title}</h3>
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
      return { background: `rgba(34, 197, 94, ${0.12 + t * 0.78})`, color: t > 0.55 ? '#fff' : 'var(--foreground)' }
    }
    const t = (mid - clamped) / (mid - min || 1) // 0..1
    return { background: `rgba(239, 68, 68, ${0.12 + t * 0.78})`, color: t > 0.55 ? '#fff' : 'var(--foreground)' }
  }

  return (
    <Card>
      <h3 className="text-sm font-semibold">{title}</h3>
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
        <div className="flex items-center gap-2 mt-3 text-[10px] text-muted-foreground">
          <span>{min.toFixed(1)}</span>
          <div
            className="h-2 flex-1 rounded-full"
            style={{ background: 'linear-gradient(to right, #ef4444, rgba(120,120,120,0.25), #22c55e)' }}
          />
          <span>{max.toFixed(1)}</span>
        </div>
      </CardContent>
    </Card>
  )
}
