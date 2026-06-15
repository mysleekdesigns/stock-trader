import { useState } from 'react'
import { X } from 'lucide-react'
import type { Order } from '../../api/client'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(value)
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const statusVariant: Record<string, 'up' | 'down' | 'neutral' | 'outline'> = {
  pending: 'neutral', filled: 'up', partial: 'outline', cancelled: 'neutral', rejected: 'down',
}

export default function TradeLog({ orders, onCancel }: { orders: Order[]; onCancel?: (id: string) => void }) {
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [filterSymbol, setFilterSymbol] = useState('')

  const filtered = orders.filter((o) => {
    if (filterStatus !== 'all' && o.status !== filterStatus) return false
    if (filterSymbol && !o.symbol.toLowerCase().includes(filterSymbol.toLowerCase())) return false
    return true
  })

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-1">
          <div className="eyebrow">Execution Ledger</div>
          <h3 className="font-display text-base font-semibold tracking-tight">Trade Log</h3>
        </div>
        <div className="flex items-center gap-3">
          <Input type="text" placeholder="Filter symbol..." value={filterSymbol}
            onChange={(e) => setFilterSymbol(e.target.value)} className="h-8 w-32 font-mono text-xs" />
          <Select value={filterStatus} onValueChange={setFilterStatus}>
            <SelectTrigger className="h-8 w-32 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="filled">Filled</SelectItem>
              <SelectItem value="partial">Partial</SelectItem>
              <SelectItem value="cancelled">Cancelled</SelectItem>
              <SelectItem value="rejected">Rejected</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="eyebrow">Time</TableHead>
              <TableHead className="eyebrow">Symbol</TableHead>
              <TableHead className="eyebrow">Side</TableHead>
              <TableHead className="eyebrow">Type</TableHead>
              <TableHead className="eyebrow text-right">Qty</TableHead>
              <TableHead className="eyebrow text-right">Price</TableHead>
              <TableHead className="eyebrow text-right">Filled</TableHead>
              <TableHead className="eyebrow text-center">Status</TableHead>
              <TableHead className="eyebrow text-center">Action</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.length === 0 ? (
              <TableRow className="hover:bg-transparent">
                <TableCell colSpan={9} className="py-8 text-center font-mono text-xs text-muted-foreground">No orders found</TableCell>
              </TableRow>
            ) : (
              filtered.map((order) => (
                <TableRow key={order.id} className="transition-colors hover:bg-accent/40">
                  <TableCell className="font-mono text-xs tabular-nums text-muted-foreground">{formatTime(order.createdAt)}</TableCell>
                  <TableCell className="font-mono font-semibold tracking-tight text-foreground">{order.symbol}</TableCell>
                  <TableCell><Badge variant={order.side === 'buy' ? 'up' : 'down'}>{order.side.toUpperCase()}</Badge></TableCell>
                  <TableCell className="font-mono text-xs uppercase tracking-wider text-muted-foreground">{order.type}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{order.quantity}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{order.price ? formatCurrency(order.price) : '--'}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {order.filledQuantity}/{order.quantity}
                    {order.filledPrice && <span className="ml-1 text-muted-foreground">@{formatCurrency(order.filledPrice)}</span>}
                  </TableCell>
                  <TableCell className="text-center">
                    <Badge variant={statusVariant[order.status] || 'neutral'}
                      className={order.status === 'partial' ? 'border border-primary/25 bg-primary/15 text-primary' : ''}>
                      {order.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-center">
                    {order.status === 'pending' && onCancel && (
                      <Button variant="ghost" size="icon-xs" onClick={() => onCancel(order.id)} title="Cancel order">
                        <X className="h-3.5 w-3.5 text-down" />
                      </Button>
                    )}
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
