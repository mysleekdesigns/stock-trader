import { useEffect, useRef } from 'react'
import { createChart, type IChartApi, type Time, ColorType } from 'lightweight-charts'
import { Card, CardContent } from '@/components/ui/card'

interface DrawdownChartProps {
  data: { time: Time; value: number }[]
  height?: number
}

export default function DrawdownChart({ data, height = 200 }: DrawdownChartProps) {
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
    })

    const series = chart.addAreaSeries({
      lineColor: '#f4615a', topColor: 'rgba(244, 97, 90, 0.05)', bottomColor: 'rgba(244, 97, 90, 0.3)', lineWidth: 2, invertFilledArea: true,
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
          <div className="eyebrow">Risk</div>
          <h3 className="font-display text-sm font-semibold tracking-tight">Drawdown</h3>
        </div>
      </div>
      <CardContent><div ref={containerRef} /></CardContent>
    </Card>
  )
}
