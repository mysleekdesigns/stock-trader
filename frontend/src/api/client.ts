import axios from 'axios'
import type { AxiosInstance, AxiosResponse, InternalAxiosRequestConfig } from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || '/api'

const client: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

client.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('auth_token')
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error),
)

client.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  },
)

// --- Dashboard ---

export interface DashboardData {
  portfolio: {
    totalValue: number
    cash: number
    unrealizedPnl: number
    drawdown: number
    dailyPnl: number
  }
  positions: Position[]
  recentOrders: Order[]
  equityCurve: { time: string; value: number }[]
}

export interface Position {
  symbol: string
  quantity: number
  avgPrice: number
  currentPrice: number
  marketValue: number
  unrealizedPnl: number
  unrealizedPnlPct: number
  side: 'long' | 'short'
}

export interface Order {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop' | 'stop_limit'
  quantity: number
  price?: number
  status: 'pending' | 'filled' | 'partial' | 'cancelled' | 'rejected'
  filledQuantity: number
  filledPrice?: number
  createdAt: string
  updatedAt: string
  strategyId?: string
}

export interface Strategy {
  id: string
  name: string
  description: string
  type: string
  enabled: boolean
  pnl: number
  winRate: number
  tradesCount: number
  sharpeRatio: number
  maxDrawdown: number
  status: 'running' | 'stopped' | 'error'
  lastSignal?: string
  lastSignalTime?: string
}

export interface BacktestRequest {
  strategy: string
  start_date: string
  end_date: string
  symbols: string[]
  initial_capital: number
  timeframe?: string
}

export interface BacktestResult {
  id: string
  status: 'pending' | 'completed' | 'failed'
  metrics: Record<string, number>
  equity_curve: number[]
  trades: Record<string, unknown>[]
  report_url?: string | null
}

export interface ModelInfo {
  id: string
  name: string
  type: string
  version: string
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  lastTrained: string
  status: 'active' | 'training' | 'inactive'
  features: string[]
  predictions: { time: string; predicted: number; actual: number }[]
}

export interface RiskMetrics {
  portfolioBeta: number
  portfolioVaR: number
  sharpeRatio: number
  sortinoRatio: number
  maxDrawdown: number
  currentDrawdown: number
  exposure: {
    long: number
    short: number
    net: number
    gross: number
  }
  sectorExposure: { sector: string; weight: number }[]
  concentrationRisk: number
}

export async function getDashboard(): Promise<DashboardData> {
  const { data } = await client.get<DashboardData>('/dashboard')
  return data
}

export async function getStrategies(): Promise<Strategy[]> {
  const { data } = await client.get<Strategy[]>('/strategies')
  return data
}

export async function toggleStrategy(id: string, enabled: boolean): Promise<Strategy> {
  const { data } = await client.patch<Strategy>(`/strategies/${id}`, { enabled })
  return data
}

export async function getOrders(params?: {
  status?: string
  symbol?: string
  limit?: number
  offset?: number
}): Promise<Order[]> {
  const { data } = await client.get<Order[]>('/orders', { params })
  return data
}

export async function submitOrder(order: {
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop' | 'stop_limit'
  quantity: number
  price?: number
}): Promise<Order> {
  const { data } = await client.post<Order>('/orders', order)
  return data
}

export async function cancelOrder(id: string): Promise<void> {
  await client.delete(`/orders/${id}`)
}

export async function launchBacktest(request: BacktestRequest): Promise<BacktestResult> {
  const { data } = await client.post<BacktestResult>('/backtest', request)
  return data
}

export async function getBacktestStatus(id: string): Promise<BacktestResult> {
  const { data } = await client.get<BacktestResult>(`/backtest/${id}`)
  return data
}

export async function runBacktest(
  request: BacktestRequest,
  onProgress?: (status: string) => void,
): Promise<BacktestResult> {
  const launched = await launchBacktest(request)
  if (launched.status !== 'pending') return launched

  // Poll until complete or failed
  const POLL_INTERVAL = 1000
  const MAX_POLLS = 120
  for (let i = 0; i < MAX_POLLS; i++) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL))
    onProgress?.(`Running... (${i + 1}s)`)
    const result = await getBacktestStatus(launched.id)
    if (result.status === 'completed' || result.status === 'failed') return result
  }
  throw new Error('Backtest timed out after 120s')
}

export async function getModels(): Promise<ModelInfo[]> {
  const { data } = await client.get<ModelInfo[]>('/models')
  return data
}

export async function getRiskMetrics(): Promise<RiskMetrics> {
  const { data } = await client.get<RiskMetrics>('/risk')
  return data
}

export default client
