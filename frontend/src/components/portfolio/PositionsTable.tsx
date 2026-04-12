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
      <h3 className="text-sm font-semibold">Open Positions</h3>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="text-muted-foreground">Symbol</TableHead>
              <TableHead className="text-muted-foreground">Side</TableHead>
              <TableHead className="text-right text-muted-foreground">Qty</TableHead>
              <TableHead className="text-right text-muted-foreground">Avg Price</TableHead>
              <TableHead className="text-right text-muted-foreground">Current</TableHead>
              <TableHead className="text-right text-muted-foreground">Mkt Value</TableHead>
              <TableHead className="text-right text-muted-foreground">P&L</TableHead>
              <TableHead className="text-right text-muted-foreground">P&L %</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {positions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="text-center text-muted-foreground py-8">No open positions</TableCell>
              </TableRow>
            ) : (
              positions.map((pos) => (
                <TableRow key={pos.symbol} className="hover:bg-accent/50">
                  <TableCell className="font-mono font-semibold">{pos.symbol}</TableCell>
                  <TableCell><Badge variant={pos.side === 'long' ? 'up' : 'down'}>{pos.side.toUpperCase()}</Badge></TableCell>
                  <TableCell className="text-right font-mono">{pos.quantity}</TableCell>
                  <TableCell className="text-right font-mono">{formatCurrency(pos.avgPrice)}</TableCell>
                  <TableCell className="text-right font-mono">{formatCurrency(pos.currentPrice)}</TableCell>
                  <TableCell className="text-right font-mono">{formatCurrency(pos.marketValue)}</TableCell>
                  <TableCell className="text-right font-mono">
                    <span className="flex items-center justify-end gap-1">
                      {pos.unrealizedPnl >= 0 ? <ArrowUpRight className="w-3 h-3 text-up" /> : <ArrowDownRight className="w-3 h-3 text-down" />}
                      <span className={pos.unrealizedPnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>{formatCurrency(Math.abs(pos.unrealizedPnl))}</span>
                    </span>
                  </TableCell>
                  <TableCell className="text-right font-mono">
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
