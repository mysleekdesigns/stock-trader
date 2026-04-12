import { useEffect, useRef } from 'react'
import { createChart, type IChartApi, ColorType } from 'lightweight-charts'
import { Card, CardContent } from '@/components/ui/card'

interface EquityCurveProps {
  data: { time: string; value: number }[]
  height?: number
  title?: string
}

export default function EquityCurve({ data, height = 300, title = 'Equity Curve' }: EquityCurveProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const isDark = document.documentElement.classList.contains('dark')
    const bg = isDark ? '#1e293b' : '#ffffff'
    const textColor = isDark ? '#94a3b8' : '#64748b'
    const gridColor = isDark ? '#334155' : '#e2e8f0'

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: { background: { type: ColorType.Solid, color: bg }, textColor, fontFamily: 'Inter, system-ui, sans-serif' },
      grid: { vertLines: { color: gridColor }, horzLines: { color: gridColor } },
      rightPriceScale: { borderColor: gridColor },
      timeScale: { borderColor: gridColor, timeVisible: true },
      crosshair: { vertLine: { color: '#6366f1', width: 1, style: 2 }, horzLine: { color: '#6366f1', width: 1, style: 2 } },
    })

    const series = chart.addAreaSeries({
      lineColor: '#6366f1', topColor: 'rgba(99, 102, 241, 0.3)', bottomColor: 'rgba(99, 102, 241, 0.02)', lineWidth: 2,
    })

    if (data.length > 0) { series.setData(data); chart.timeScale().fitContent() }
    chartRef.current = chart

    const handleResize = () => { if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth }) }
    window.addEventListener('resize', handleResize)
    return () => { window.removeEventListener('resize', handleResize); chart.remove() }
  }, [data, height])

  return (
    <Card>
      <h3 className="text-sm font-semibold">{title}</h3>
      <CardContent><div ref={containerRef} /></CardContent>
    </Card>
  )
}
