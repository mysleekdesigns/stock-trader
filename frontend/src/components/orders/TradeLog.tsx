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
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Trade Log</h3>
        <div className="flex items-center gap-3">
          <Input type="text" placeholder="Filter symbol..." value={filterSymbol}
            onChange={(e) => setFilterSymbol(e.target.value)} className="w-32 h-8 text-xs" />
          <Select value={filterStatus} onValueChange={setFilterStatus}>
            <SelectTrigger className="w-32 h-8 text-xs"><SelectValue /></SelectTrigger>
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
              <TableHead className="text-muted-foreground">Time</TableHead>
              <TableHead className="text-muted-foreground">Symbol</TableHead>
              <TableHead className="text-muted-foreground">Side</TableHead>
              <TableHead className="text-muted-foreground">Type</TableHead>
              <TableHead className="text-right text-muted-foreground">Qty</TableHead>
              <TableHead className="text-right text-muted-foreground">Price</TableHead>
              <TableHead className="text-right text-muted-foreground">Filled</TableHead>
              <TableHead className="text-center text-muted-foreground">Status</TableHead>
              <TableHead className="text-center text-muted-foreground">Action</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.length === 0 ? (
              <TableRow>
                <TableCell colSpan={9} className="text-center text-muted-foreground py-8">No orders found</TableCell>
              </TableRow>
            ) : (
              filtered.map((order) => (
                <TableRow key={order.id} className="hover:bg-accent/50">
                  <TableCell className="text-xs font-mono text-muted-foreground">{formatTime(order.createdAt)}</TableCell>
                  <TableCell className="font-mono font-semibold">{order.symbol}</TableCell>
                  <TableCell><Badge variant={order.side === 'buy' ? 'up' : 'down'}>{order.side.toUpperCase()}</Badge></TableCell>
                  <TableCell className="text-xs uppercase text-muted-foreground">{order.type}</TableCell>
                  <TableCell className="text-right font-mono">{order.quantity}</TableCell>
                  <TableCell className="text-right font-mono">{order.price ? formatCurrency(order.price) : '--'}</TableCell>
                  <TableCell className="text-right font-mono">
                    {order.filledQuantity}/{order.quantity}
                    {order.filledPrice && <span className="text-muted-foreground ml-1">@{formatCurrency(order.filledPrice)}</span>}
                  </TableCell>
                  <TableCell className="text-center">
                    <Badge variant={statusVariant[order.status] || 'neutral'}
                      className={order.status === 'partial' ? 'bg-amber-500/15 text-amber-400 border-transparent' : ''}>
                      {order.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-center">
                    {order.status === 'pending' && onCancel && (
                      <Button variant="ghost" size="icon-xs" onClick={() => onCancel(order.id)} title="Cancel order">
                        <X className="w-3.5 h-3.5 text-down" />
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
