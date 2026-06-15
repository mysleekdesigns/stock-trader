import { useEffect, useRef } from 'react'
import { createChart, type IChartApi, type Time, ColorType } from 'lightweight-charts'
import { Card, CardContent } from '@/components/ui/card'

interface EquityCurveProps {
  data: { time: Time; value: number }[]
  height?: number
  title?: string
}

export default function EquityCurve({ data, height = 300, title = 'Equity Curve' }: EquityCurveProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const isDark = document.documentElement.classList.contains('dark')
    const bg = isDark ? '#221e18' : '#fdfcfa'
    const textColor = isDark ? '#8a8275' : '#8c8678'
    const gridColor = isDark ? '#2c2720' : '#ece8e0'

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: { background: { type: ColorType.Solid, color: bg }, textColor, fontFamily: 'IBM Plex Mono, monospace' },
      grid: { vertLines: { color: gridColor }, horzLines: { color: gridColor } },
      rightPriceScale: { borderColor: gridColor },
      timeScale: { borderColor: gridColor, timeVisible: true },
      crosshair: { vertLine: { color: '#e8ae49', width: 1, style: 2 }, horzLine: { color: '#e8ae49', width: 1, style: 2 } },
    })

    const series = chart.addAreaSeries({
      lineColor: '#e8ae49', topColor: 'rgba(232, 174, 73, 0.28)', bottomColor: 'rgba(232, 174, 73, 0.02)', lineWidth: 2,
    })

    if (data.length > 0) { series.setData(data); chart.timeScale().fitContent() }
    chartRef.current = chart

    const handleResize = () => { if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth }) }
    window.addEventListener('resize', handleResize)
    return () => { window.removeEventListener('resize', handleResize); chart.remove() }
  }, [data, height])

  return (
    <Card>
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <div className="eyebrow">Performance</div>
          <h3 className="font-display text-sm font-semibold tracking-tight">{title}</h3>
        </div>
      </div>
      <CardContent><div ref={containerRef} /></CardContent>
    </Card>
  )
}
