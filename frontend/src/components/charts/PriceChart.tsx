import { useEffect, useRef } from 'react'
import { createChart, type IChartApi, type ISeriesApi, ColorType } from 'lightweight-charts'
import { Card, CardContent } from '@/components/ui/card'

interface CandlestickData { time: string; open: number; high: number; low: number; close: number; volume?: number }
interface PriceChartProps { data: CandlestickData[]; symbol?: string; height?: number }

export default function PriceChart({ data, symbol, height = 400 }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const isDark = document.documentElement.classList.contains('dark')
    const bg = isDark ? '#1e293b' : '#ffffff'
    const textColor = isDark ? '#94a3b8' : '#64748b'
    const gridColor = isDark ? '#334155' : '#e2e8f0'

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth, height,
      layout: { background: { type: ColorType.Solid, color: bg }, textColor, fontFamily: 'Inter, system-ui, sans-serif' },
      grid: { vertLines: { color: gridColor }, horzLines: { color: gridColor } },
      crosshair: { vertLine: { color: '#6366f1', width: 1, style: 2 }, horzLine: { color: '#6366f1', width: 1, style: 2 } },
      rightPriceScale: { borderColor: gridColor }, timeScale: { borderColor: gridColor, timeVisible: true },
    })

    const series = chart.addCandlestickSeries({
      upColor: '#22c55e', downColor: '#ef4444', borderUpColor: '#22c55e', borderDownColor: '#ef4444', wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    })

    chartRef.current = chart; seriesRef.current = series
    const handleResize = () => { if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth }) }
    window.addEventListener('resize', handleResize)
    return () => { window.removeEventListener('resize', handleResize); chart.remove() }
  }, [height])

  useEffect(() => {
    if (seriesRef.current && data.length > 0) { seriesRef.current.setData(data); chartRef.current?.timeScale().fitContent() }
  }, [data])

  return (
    <Card>
      {symbol && <h3 className="text-sm font-semibold">{symbol}</h3>}
      <CardContent><div ref={containerRef} /></CardContent>
    </Card>
  )
}
