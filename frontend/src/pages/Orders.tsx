import { useEffect, useState } from 'react'
import { getOrders, cancelOrder } from '../api/client'
import { useTradeStore } from '../stores/useTradeStore'
import TradeLog from '../components/orders/TradeLog'
import { Loader2 } from 'lucide-react'

export default function Orders() {
  const { orders, setOrders, updateOrder } = useTradeStore()
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      try {
        const data = await getOrders()
        setOrders(data)
      } catch {
        // Use placeholder data
        setOrders([
          {
            id: '1',
            symbol: 'AAPL',
            side: 'buy',
            type: 'limit',
            quantity: 100,
            price: 178.5,
            status: 'filled',
            filledQuantity: 100,
            filledPrice: 178.45,
            createdAt: '2026-04-12T09:30:00Z',
            updatedAt: '2026-04-12T09:30:05Z',
            strategyId: 'mean_reversion',
          },
          {
            id: '2',
            symbol: 'MSFT',
            side: 'sell',
            type: 'market',
            quantity: 50,
            status: 'filled',
            filledQuantity: 50,
            filledPrice: 412.3,
            createdAt: '2026-04-12T10:15:00Z',
            updatedAt: '2026-04-12T10:15:01Z',
            strategyId: 'momentum',
          },
          {
            id: '3',
            symbol: 'GOOGL',
            side: 'buy',
            type: 'limit',
            quantity: 30,
            price: 165.0,
            status: 'pending',
            filledQuantity: 0,
            createdAt: '2026-04-12T11:00:00Z',
            updatedAt: '2026-04-12T11:00:00Z',
          },
          {
            id: '4',
            symbol: 'NVDA',
            side: 'buy',
            type: 'stop_limit',
            quantity: 25,
            price: 880.0,
            status: 'pending',
            filledQuantity: 0,
            createdAt: '2026-04-12T11:30:00Z',
            updatedAt: '2026-04-12T11:30:00Z',
            strategyId: 'lstm',
          },
          {
            id: '5',
            symbol: 'TSLA',
            side: 'sell',
            type: 'limit',
            quantity: 40,
            price: 175.0,
            status: 'cancelled',
            filledQuantity: 0,
            createdAt: '2026-04-11T14:00:00Z',
            updatedAt: '2026-04-11T14:30:00Z',
          },
        ])
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [setOrders])

  async function handleCancel(id: string) {
    try {
      await cancelOrder(id)
      updateOrder(id, { status: 'cancelled' })
    } catch {
      // Handle error
    }
  }

  if (loading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  const pendingCount = orders.filter((o) => o.status === 'pending').length

  return (
    <div className="space-y-7">
      {/* ---- Page header ---- */}
      <div className="animate-rise space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="space-y-1.5">
            <div className="eyebrow">06 — Order Flow</div>
            <h1 className="font-display text-3xl font-semibold tracking-tight">Orders</h1>
            <p className="text-sm text-muted-foreground">
              Execution log across strategies — fills, partials, and live working orders.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="panel flex items-center gap-2 px-3 py-2">
              <span className="eyebrow !text-[0.6rem]">Pending</span>
              <span className="font-mono text-sm font-semibold tabular-nums text-primary">
                {String(pendingCount).padStart(2, '0')}
              </span>
            </div>
          </div>
        </div>
        <div className="rule" />
      </div>

      {/* ---- Trade log ---- */}
      <div className="animate-rise" style={{ animationDelay: '80ms' }}>
        <TradeLog orders={orders} onCancel={handleCancel} />
      </div>
    </div>
  )
}
