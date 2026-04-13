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
    const bg = isDark ? '#1e293b' : '#ffffff'
    const textColor = isDark ? '#94a3b8' : '#64748b'
    const gridColor = isDark ? '#334155' : '#e2e8f0'

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: bg },
        textColor,
        fontFamily: 'Inter, system-ui, sans-serif',
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      crosshair: {
        vertLine: { color: '#6366f1', width: 1, style: 2 },
        horzLine: { color: '#6366f1', width: 1, style: 2 },
      },
      rightPriceScale: { borderColor: gridColor },
      timeScale: { borderColor: gridColor, timeVisible: true, secondsVisible: false },
    })

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
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
          color: d.close >= d.open ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)',
        })) as any,
      )
    }

    // OR High line
    if (overlay.orHigh !== null && chartData.length > 0) {
      const orHighLine = chart.addLineSeries({
        color: '#f97316',
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
        color: '#f97316',
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
        color: '#3b82f6',
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
          color: '#22c55e',
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
        <div className="px-6 pt-4">
          <h3 className="text-sm font-semibold">{symbol} — ORB Scanner</h3>
        </div>
      )}
      <CardContent>
        <div ref={containerRef} />
      </CardContent>
    </Card>
  )
}
