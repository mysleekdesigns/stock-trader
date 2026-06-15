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
    const bg = isDark ? '#221e18' : '#fdfcfa'
    const textColor = isDark ? '#8a8275' : '#8c8678'
    const gridColor = isDark ? '#2c2720' : '#ece8e0'

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth, height,
      layout: { background: { type: ColorType.Solid, color: bg }, textColor, fontFamily: 'IBM Plex Mono, monospace' },
      grid: { vertLines: { color: gridColor }, horzLines: { color: gridColor } },
      crosshair: { vertLine: { color: '#e8ae49', width: 1, style: 2 }, horzLine: { color: '#e8ae49', width: 1, style: 2 } },
      rightPriceScale: { borderColor: gridColor }, timeScale: { borderColor: gridColor, timeVisible: true },
    })

    const series = chart.addCandlestickSeries({
      upColor: '#2fcf8e', downColor: '#f4615a', borderUpColor: '#2fcf8e', borderDownColor: '#f4615a', wickUpColor: '#2fcf8e', wickDownColor: '#f4615a',
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
      {symbol && (
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="eyebrow">Price</div>
            <h3 className="font-mono text-sm font-semibold tracking-tight tabular-nums">{symbol}</h3>
          </div>
        </div>
      )}
      <CardContent><div ref={containerRef} /></CardContent>
    </Card>
  )
}
