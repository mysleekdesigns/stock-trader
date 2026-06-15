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
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Orders</h1>
        <span className="text-xs text-muted-foreground">
          {orders.filter((o) => o.status === 'pending').length} pending orders
        </span>
      </div>

      <TradeLog orders={orders} onCancel={handleCancel} />
    </div>
  )
}
