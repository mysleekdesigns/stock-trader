import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  GitBranch,
  Crosshair,
  FlaskConical,
  Brain,
  Receipt,
  Settings,
  TrendingUp,
  Wifi,
  WifiOff,
  Sun,
  Moon,
} from 'lucide-react'
import { useTradeStore } from '../stores/useTradeStore'
import { useTheme } from '../hooks/useTheme'
import { Button } from '@/components/ui/button'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/strategies', icon: GitBranch, label: 'Strategies' },
  { to: '/orb', icon: Crosshair, label: 'ORB Scanner' },
  { to: '/backtest', icon: FlaskConical, label: 'Backtest' },
  { to: '/models', icon: Brain, label: 'Models' },
  { to: '/orders', icon: Receipt, label: 'Orders' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value)
}

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const { portfolio, connected } = useTradeStore()
  const { theme, toggleTheme } = useTheme()

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-sidebar border-r border-sidebar-border flex flex-col">
        {/* Logo */}
        <div className="h-16 flex items-center gap-3 px-6 border-b border-sidebar-border">
          <TrendingUp className="w-6 h-6 text-primary" />
          <span className="text-lg font-semibold tracking-tight">StockTrader</span>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 px-3 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-sidebar-accent text-sidebar-primary'
                    : 'text-muted-foreground hover:text-foreground hover:bg-sidebar-accent'
                }`
              }
            >
              <Icon className="w-5 h-5" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Connection status */}
        <div className="p-4 border-t border-sidebar-border">
          <div className="flex items-center gap-2 text-xs">
            {connected ? (
              <>
                <Wifi className="w-4 h-4 text-up" />
                <span className="text-up">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-down" />
                <span className="text-down">Disconnected</span>
              </>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 flex-shrink-0 bg-card border-b border-border flex items-center justify-between px-6">
          <div className="flex items-center gap-8">
            <div>
              <div className="text-xs text-muted-foreground">Portfolio Value</div>
              <div className="text-xl font-semibold font-mono">
                {formatCurrency(portfolio.totalValue)}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Daily P&L</div>
              <div
                className={`text-lg font-semibold font-mono ${
                  portfolio.dailyPnl >= 0 ? 'pnl-positive' : 'pnl-negative'
                }`}
              >
                {portfolio.dailyPnl >= 0 ? '+' : ''}
                {formatCurrency(portfolio.dailyPnl)}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Drawdown</div>
              <div className="text-lg font-semibold font-mono text-down">
                {(portfolio.drawdown * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={toggleTheme} title="Toggle theme">
              {theme === 'dark' ? (
                <Sun className="w-4 h-4" />
              ) : (
                <Moon className="w-4 h-4" />
              )}
            </Button>
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  connected ? 'bg-up animate-pulse' : 'bg-down'
                }`}
              />
              <span className="text-xs text-muted-foreground">
                {connected ? 'Live' : 'Offline'}
              </span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  )
}
