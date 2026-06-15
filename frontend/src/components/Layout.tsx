import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  GitBranch,
  Crosshair,
  FlaskConical,
  Brain,
  Receipt,
  Settings,
  Sun,
  Moon,
} from 'lucide-react'
import { useTradeStore } from '../stores/useTradeStore'
import { useTheme } from '../hooks/useTheme'

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

/** Brand mark: a tiny amber candlestick cluster. */
function CandleMark({ className = '' }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" aria-hidden="true">
      <rect x="4" y="8" width="3" height="9" rx="0.6" fill="currentColor" opacity="0.55" />
      <line x1="5.5" y1="5" x2="5.5" y2="20" stroke="currentColor" strokeWidth="1" opacity="0.55" />
      <rect x="10.5" y="4" width="3" height="11" rx="0.6" fill="currentColor" />
      <line x1="12" y1="2" x2="12" y2="18" stroke="currentColor" strokeWidth="1" />
      <rect x="17" y="10" width="3" height="7" rx="0.6" fill="currentColor" opacity="0.55" />
      <line x1="18.5" y1="7" x2="18.5" y2="20" stroke="currentColor" strokeWidth="1" opacity="0.55" />
    </svg>
  )
}

/** US-equities session clock in Eastern Time, updating every second. */
function useMarketClock() {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    weekday: 'short',
  }).formatToParts(now)
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? ''
  const time = `${get('hour')}:${get('minute')}:${get('second')}`
  const weekday = get('weekday')
  const minutes = Number(get('hour')) * 60 + Number(get('minute'))
  const isWeekday = !['Sat', 'Sun'].includes(weekday)
  const open = isWeekday && minutes >= 570 && minutes < 960 // 09:30–16:00 ET
  return { time, open }
}

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const { portfolio, connected } = useTradeStore()
  const { theme, toggleTheme } = useTheme()
  const { time, open } = useMarketClock()

  const pnlUp = portfolio.dailyPnl >= 0

  return (
    <div className="relative flex h-screen overflow-hidden bg-background text-foreground">
      {/* Atmosphere — fixed behind everything */}
      <div className="pointer-events-none fixed inset-0 -z-10 bg-grid" aria-hidden="true" />
      <div className="grain pointer-events-none fixed inset-0 -z-10" aria-hidden="true" />
      <div
        className="pointer-events-none fixed -top-40 left-1/3 -z-10 h-[34rem] w-[34rem] rounded-full opacity-[0.07] blur-3xl"
        style={{ background: 'radial-gradient(circle, var(--primary), transparent 70%)' }}
        aria-hidden="true"
      />

      {/* ---- Sidebar ---- */}
      <aside className="relative z-10 flex w-62 flex-shrink-0 flex-col border-r border-sidebar-border bg-sidebar/80 backdrop-blur-xl">
        {/* Brand */}
        <div className="flex h-16 items-center gap-3 border-b border-sidebar-border px-5">
          <div className="grid h-9 w-9 place-items-center rounded-md border border-primary/30 bg-primary/10 text-primary">
            <CandleMark className="h-5 w-5" />
          </div>
          <div className="leading-none">
            <div className="font-mono text-[0.95rem] font-semibold tracking-tight">
              STOCK<span className="text-primary">TRADER</span>
            </div>
            <div className="eyebrow mt-1.5 !text-[0.6rem]">Algo Terminal · v0.1</div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto px-3 py-5">
          <div className="eyebrow px-3 pb-3">Navigation</div>
          <div className="space-y-0.5">
            {navItems.map(({ to, icon: Icon, label }, i) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `group relative flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-all duration-200 ${
                    isActive
                      ? 'bg-sidebar-accent text-foreground'
                      : 'text-muted-foreground hover:translate-x-0.5 hover:bg-sidebar-accent/60 hover:text-foreground'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className={`absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-primary transition-all duration-200 ${
                        isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-40'
                      }`}
                    />
                    <span
                      className={`font-mono text-[0.65rem] tabular-nums ${
                        isActive ? 'text-primary' : 'text-muted-foreground/60'
                      }`}
                    >
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    <Icon className={`h-[18px] w-[18px] ${isActive ? 'text-primary' : ''}`} strokeWidth={1.75} />
                    <span className="font-medium">{label}</span>
                  </>
                )}
              </NavLink>
            ))}
          </div>
        </nav>

        {/* System block */}
        <div className="space-y-3 border-t border-sidebar-border px-5 py-4">
          <div className="eyebrow">System</div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Market</span>
            <span className="flex items-center gap-1.5">
              <span
                className={`relative inline-flex h-1.5 w-1.5 rounded-full ${open ? 'bg-up' : 'bg-muted-foreground'}`}
              >
                {open && (
                  <span className="absolute inset-0 inline-flex rounded-full bg-up animate-pulse-ring" />
                )}
              </span>
              <span className={`font-mono text-[0.7rem] ${open ? 'text-up' : 'text-muted-foreground'}`}>
                {open ? 'OPEN' : 'CLOSED'}
              </span>
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Feed</span>
            <span className={`font-mono text-[0.7rem] ${connected ? 'text-up' : 'text-down'}`}>
              {connected ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">ET</span>
            <span className="font-mono text-[0.7rem] tabular-nums text-foreground/80">{time}</span>
          </div>
        </div>
      </aside>

      {/* ---- Main ---- */}
      <div className="relative z-10 flex flex-1 flex-col overflow-hidden">
        {/* Topbar instrument readout */}
        <header className="relative flex h-16 flex-shrink-0 items-center justify-between border-b border-border bg-card/70 px-6 backdrop-blur-xl">
          <div className="flex items-stretch">
            <Readout label="Portfolio Value">
              <span className="font-mono text-lg font-semibold tabular-nums text-foreground">
                {formatCurrency(portfolio.totalValue)}
              </span>
            </Readout>
            <Divider />
            <Readout label="Daily P&L">
              <span
                className={`font-mono text-lg font-semibold tabular-nums ${pnlUp ? 'text-up' : 'text-down'}`}
              >
                {pnlUp ? '+' : ''}
                {formatCurrency(portfolio.dailyPnl)}
              </span>
            </Readout>
            <Divider />
            <Readout label="Drawdown">
              <span className="font-mono text-lg font-semibold tabular-nums text-down">
                {(portfolio.drawdown * 100).toFixed(2)}%
              </span>
            </Readout>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden items-center gap-2 rounded-full border border-border bg-background/60 px-3 py-1.5 sm:flex">
              <span className={`h-2 w-2 rounded-full ${connected ? 'bg-up animate-pulse-ring' : 'bg-down'}`} />
              <span className="font-mono text-[0.7rem] tracking-wider text-muted-foreground">
                {connected ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>
            <button
              onClick={toggleTheme}
              title="Toggle theme"
              className="grid h-9 w-9 place-items-center rounded-md border border-border bg-background/60 text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
            >
              {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>

          {/* Hairline amber accent under the header */}
          <span className="pointer-events-none absolute bottom-0 left-0 h-px w-full bg-gradient-to-r from-primary/40 via-transparent to-transparent" />
        </header>

        <main className="flex-1 overflow-y-auto px-6 py-7 lg:px-8">{children}</main>
      </div>
    </div>
  )
}

function Readout({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col justify-center gap-1 pr-6">
      <span className="eyebrow !text-[0.6rem]">{label}</span>
      {children}
    </div>
  )
}

function Divider() {
  return <span className="my-2 w-px self-stretch bg-border" />
}
