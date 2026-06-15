import { useEffect, useRef, useMemo } from 'react'
import {
  createChart,
  type IChartApi,
  ColorType,
  LineStyle,
} from 'lightweight-charts'
import { Card, CardContent } from '@/components/ui/card'

interface CandlestickData {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface ORBOverlay {
  orHigh: number | null
  orLow: number | null
  vwap: number | null
  breakoutTime: string | null
}

interface ORBChartProps {
  data: CandlestickData[]
  overlay: ORBOverlay
  symbol?: string
  height?: number
}

/** Convert an ISO datetime string to a Unix timestamp (seconds). */
function toUnix(iso: string): number {
  return Math.floor(new Date(iso).getTime() / 1000)
}

export default function ORBChart({ data, overlay, symbol, height = 500 }: ORBChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  // Convert ISO time strings to Unix timestamps once
  const chartData = useMemo(
    () => data.map((d) => ({ ...d, time: toUnix(d.time) })),
    [data],
  )

  const breakoutUnix = useMemo(
    () => (overlay.breakoutTime ? toUnix(overlay.breakoutTime) : null),
    [overlay.breakoutTime],
  )

  useEffect(() => {
    if (!containerRef.current) return
    const isDark = document.documentElement.classList.contains('dark')
    const bg = isDark ? '#221e18' : '#fdfcfa'
    const textColor = isDark ? '#8a8275' : '#8c8678'
    const gridColor = isDark ? '#2c2720' : '#ece8e0'

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: bg },
        textColor,
        fontFamily: 'IBM Plex Mono, monospace',
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      crosshair: {
        vertLine: { color: '#e8ae49', width: 1, style: 2 },
        horzLine: { color: '#e8ae49', width: 1, style: 2 },
      },
      rightPriceScale: { borderColor: gridColor },
      timeScale: { borderColor: gridColor, timeVisible: true, secondsVisible: false },
    })

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#2fcf8e',
      downColor: '#f4615a',
      borderUpColor: '#2fcf8e',
      borderDownColor: '#f4615a',
      wickUpColor: '#2fcf8e',
      wickDownColor: '#f4615a',
    })

    if (chartData.length > 0) {
      candleSeries.setData(chartData as any)
    }

    // Volume series
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    })

    if (chartData.length > 0) {
      volumeSeries.setData(
        chartData.map((d) => ({
          time: d.time,
          value: d.volume || 0,
          color: d.close >= d.open ? 'rgba(47, 207, 142, 0.30)' : 'rgba(244, 97, 90, 0.30)',
        })) as any,
      )
    }

    // OR High line
    if (overlay.orHigh !== null && chartData.length > 0) {
      const orHighLine = chart.addLineSeries({
        color: '#fb923c',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        title: 'OR High',
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      })
      orHighLine.setData(
        chartData.map((d) => ({ time: d.time, value: overlay.orHigh! })) as any,
      )
    }

    // OR Low line
    if (overlay.orLow !== null && chartData.length > 0) {
      const orLowLine = chart.addLineSeries({
        color: '#fb923c',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        title: 'OR Low',
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      })
      orLowLine.setData(
        chartData.map((d) => ({ time: d.time, value: overlay.orLow! })) as any,
      )
    }

    // VWAP line
    if (overlay.vwap !== null && chartData.length > 0) {
      const vwapLine = chart.addLineSeries({
        color: '#4fb6c4',
        lineWidth: 1,
        lineStyle: LineStyle.Solid,
        title: 'VWAP',
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      })
      vwapLine.setData(
        chartData.map((d) => ({ time: d.time, value: overlay.vwap! })) as any,
      )
    }

    // Breakout marker
    if (breakoutUnix !== null && chartData.length > 0) {
      candleSeries.setMarkers([
        {
          time: breakoutUnix as any,
          position: 'aboveBar',
          color: '#2fcf8e',
          shape: 'arrowUp',
          text: 'ORB',
        },
      ])
    }

    chart.timeScale().fitContent()
    chartRef.current = chart

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [chartData, overlay.orHigh, overlay.orLow, overlay.vwap, breakoutUnix, height])

  return (
    <Card>
      {symbol && (
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="eyebrow">Opening Range Breakout</div>
            <h3 className="font-mono text-sm font-semibold tracking-tight tabular-nums">{symbol}</h3>
          </div>
          <div className="flex items-center gap-3 font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: '#fb923c' }} />OR
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: '#4fb6c4' }} />VWAP
            </span>
          </div>
        </div>
      )}
      <CardContent>
        <div ref={containerRef} />
      </CardContent>
    </Card>
  )
}
