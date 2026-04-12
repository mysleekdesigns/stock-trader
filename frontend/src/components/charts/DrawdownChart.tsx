import { useEffect, useRef } from 'react'
import { createChart, type IChartApi, ColorType } from 'lightweight-charts'
import { Card, CardContent } from '@/components/ui/card'

interface DrawdownChartProps {
  data: { time: string; value: number }[]
  height?: number
}

export default function DrawdownChart({ data, height = 200 }: DrawdownChartProps) {
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
    })

    const series = chart.addAreaSeries({
      lineColor: '#ef4444', topColor: 'rgba(239, 68, 68, 0.05)', bottomColor: 'rgba(239, 68, 68, 0.3)', lineWidth: 2, invertFilledArea: true,
    })

    if (data.length > 0) { series.setData(data); chart.timeScale().fitContent() }
    chartRef.current = chart

    const handleResize = () => { if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth }) }
    window.addEventListener('resize', handleResize)
    return () => { window.removeEventListener('resize', handleResize); chart.remove() }
  }, [data, height])

  return (
    <Card>
      <h3 className="text-sm font-semibold">Drawdown</h3>
      <CardContent><div ref={containerRef} /></CardContent>
    </Card>
  )
}
