import { ArrowUpRight, ArrowDownRight } from 'lucide-react'
import type { Position } from '../../api/client'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(value)
}

function formatPct(value: number): string {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`
}

export default function PositionsTable({ positions }: { positions: Position[] }) {
  return (
    <Card>
      <div className="flex items-center justify-between">
        <h3 className="font-display text-base font-semibold tracking-tight">Open Positions</h3>
        <span className="eyebrow tabular-nums">{positions.length} Open</span>
      </div>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="eyebrow">Symbol</TableHead>
              <TableHead className="eyebrow">Side</TableHead>
              <TableHead className="eyebrow text-right">Qty</TableHead>
              <TableHead className="eyebrow text-right">Avg Price</TableHead>
              <TableHead className="eyebrow text-right">Current</TableHead>
              <TableHead className="eyebrow text-right">Mkt Value</TableHead>
              <TableHead className="eyebrow text-right">P&L</TableHead>
              <TableHead className="eyebrow text-right">P&L %</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {positions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="py-10 text-center text-sm text-muted-foreground">No open positions</TableCell>
              </TableRow>
            ) : (
              positions.map((pos) => (
                <TableRow key={pos.symbol} className="transition-colors hover:bg-accent/40">
                  <TableCell className="font-mono font-semibold tracking-tight">{pos.symbol}</TableCell>
                  <TableCell><Badge variant={pos.side === 'long' ? 'up' : 'down'}>{pos.side.toUpperCase()}</Badge></TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{pos.quantity}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{formatCurrency(pos.avgPrice)}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{formatCurrency(pos.currentPrice)}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{formatCurrency(pos.marketValue)}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    <span className="flex items-center justify-end gap-1">
                      {pos.unrealizedPnl >= 0 ? <ArrowUpRight className="h-3 w-3 text-up" /> : <ArrowDownRight className="h-3 w-3 text-down" />}
                      <span className={pos.unrealizedPnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>{formatCurrency(Math.abs(pos.unrealizedPnl))}</span>
                    </span>
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    <span className={pos.unrealizedPnlPct >= 0 ? 'pnl-positive' : 'pnl-negative'}>{formatPct(pos.unrealizedPnlPct)}</span>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
