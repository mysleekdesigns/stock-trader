import { create } from 'zustand'
import type { Position, Order, Strategy } from '../api/client'

export interface Signal {
  id: string
  strategyId: string
  symbol: string
  action: 'buy' | 'sell' | 'hold'
  strength: number
  timestamp: string
}

interface PortfolioState {
  totalValue: number
  cash: number
  unrealizedPnl: number
  drawdown: number
  dailyPnl: number
  weeklyPnl: number
  monthlyPnl: number
}

interface TradeStoreState {
  portfolio: PortfolioState
  positions: Position[]
  strategies: Strategy[]
  orders: Order[]
  prices: Map<string, number>
  signals: Signal[]
  equityCurve: { time: string; value: number }[]
  connected: boolean

  updatePortfolio: (portfolio: Partial<PortfolioState>) => void
  setPositions: (positions: Position[]) => void
  updatePositions: (positions: Position[]) => void
  setStrategies: (strategies: Strategy[]) => void
  updateStrategy: (id: string, update: Partial<Strategy>) => void
  addOrder: (order: Order) => void
  setOrders: (orders: Order[]) => void
  updateOrder: (id: string, update: Partial<Order>) => void
  updatePrice: (symbol: string, price: number) => void
  addSignal: (signal: Signal) => void
  setEquityCurve: (curve: { time: string; value: number }[]) => void
  setConnected: (connected: boolean) => void
}

export const useTradeStore = create<TradeStoreState>((set) => ({
  portfolio: {
    totalValue: 0,
    cash: 0,
    unrealizedPnl: 0,
    drawdown: 0,
    dailyPnl: 0,
    weeklyPnl: 0,
    monthlyPnl: 0,
  },
  positions: [],
  strategies: [],
  orders: [],
  prices: new Map(),
  signals: [],
  equityCurve: [],
  connected: false,

  updatePortfolio: (portfolio) =>
    set((state) => ({
      portfolio: { ...state.portfolio, ...portfolio },
    })),

  setPositions: (positions) => set({ positions }),

  updatePositions: (newPositions) =>
    set((state) => {
      const posMap = new Map(state.positions.map((p) => [p.symbol, p]))
      newPositions.forEach((p) => posMap.set(p.symbol, p))
      return { positions: Array.from(posMap.values()) }
    }),

  setStrategies: (strategies) => set({ strategies }),

  updateStrategy: (id, update) =>
    set((state) => ({
      strategies: state.strategies.map((s) =>
        s.id === id ? { ...s, ...update } : s,
      ),
    })),

  addOrder: (order) =>
    set((state) => ({
      orders: [order, ...state.orders].slice(0, 500),
    })),

  setOrders: (orders) => set({ orders }),

  updateOrder: (id, update) =>
    set((state) => ({
      orders: state.orders.map((o) => (o.id === id ? { ...o, ...update } : o)),
    })),

  updatePrice: (symbol, price) =>
    set((state) => {
      const prices = new Map(state.prices)
      prices.set(symbol, price)
      return { prices }
    }),

  addSignal: (signal) =>
    set((state) => ({
      signals: [signal, ...state.signals].slice(0, 200),
    })),

  setEquityCurve: (equityCurve) => set({ equityCurve }),

  setConnected: (connected) => set({ connected }),
}))
