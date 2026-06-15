import { useState } from 'react'
import { X, RotateCcw, Save } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'

export type ParamValue = number | boolean | string

export interface StrategyParam {
  key: string
  /** Human-readable label. Falls back to a prettified `key`. */
  label?: string
  value: ParamValue
  /** Optional bounds/step for numeric inputs. */
  min?: number
  max?: number
  step?: number
  description?: string
}

interface StrategyConfigProps {
  strategyName: string
  params: StrategyParam[]
  onSave: (values: Record<string, ParamValue>) => void
  onClose?: () => void
  saving?: boolean
}

function prettify(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

/**
 * Editable panel for a strategy's tunable parameters.
 *
 * Maintains a local draft so edits are non-destructive until the user clicks
 * Save; Reset reverts to the values the panel was opened with. Numeric, boolean
 * and string parameters are each rendered with the appropriate control.
 */
export default function StrategyConfig({ strategyName, params, onSave, onClose, saving = false }: StrategyConfigProps) {
  const initial = Object.fromEntries(params.map((p) => [p.key, p.value])) as Record<string, ParamValue>
  // Numeric fields are held as raw strings while editing so partial input
  // ("-", "1.", "1e") is preserved instead of collapsing to NaN; they are
  // coerced back to numbers on save.
  const [draft, setDraft] = useState<Record<string, ParamValue>>(initial)

  const dirty = params.some((p) => String(draft[p.key]) !== String(initial[p.key]))

  function update(key: string, value: ParamValue) {
    setDraft((prev) => ({ ...prev, [key]: value }))
  }

  function handleSave() {
    const result: Record<string, ParamValue> = {}
    for (const p of params) {
      const v = draft[p.key]
      if (typeof p.value === 'number') {
        const n = Number(v)
        result[p.key] = Number.isFinite(n) ? n : p.value
      } else {
        result[p.key] = v
      }
    }
    onSave(result)
  }

  return (
    <Card className="w-full max-w-lg">
      <CardContent>
        <div className="flex items-center justify-between mb-1">
          <div>
            <h3 className="text-sm font-semibold">Configure Strategy</h3>
            <p className="text-xs text-muted-foreground mt-0.5">{strategyName}</p>
          </div>
          {onClose && (
            <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close">
              <X />
            </Button>
          )}
        </div>

        <Separator className="my-3" />

        <div className="space-y-4">
          {params.map((param) => {
            const label = param.label ?? prettify(param.key)
            const current = draft[param.key]

            if (typeof param.value === 'boolean') {
              return (
                <div key={param.key} className="flex items-center justify-between gap-4">
                  <div>
                    <Label htmlFor={param.key}>{label}</Label>
                    {param.description && (
                      <p className="text-xs text-muted-foreground mt-0.5">{param.description}</p>
                    )}
                  </div>
                  <Switch
                    id={param.key}
                    checked={current as boolean}
                    onCheckedChange={(checked) => update(param.key, checked)}
                  />
                </div>
              )
            }

            const isNumber = typeof param.value === 'number'
            return (
              <div key={param.key} className="space-y-1.5">
                <Label htmlFor={param.key}>{label}</Label>
                {param.description && (
                  <p className="text-xs text-muted-foreground">{param.description}</p>
                )}
                <Input
                  id={param.key}
                  type={isNumber ? 'number' : 'text'}
                  className="font-mono"
                  value={current === undefined || current === null ? '' : String(current)}
                  min={param.min}
                  max={param.max}
                  step={param.step ?? (isNumber ? 'any' : undefined)}
                  onChange={(e) => update(param.key, e.target.value)}
                />
              </div>
            )
          })}
          {params.length === 0 && (
            <p className="text-sm text-muted-foreground py-4 text-center">
              This strategy has no adjustable parameters.
            </p>
          )}
        </div>

        <Separator className="my-4" />

        <div className="flex items-center justify-end gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setDraft(initial)}
            disabled={!dirty || saving}
          >
            <RotateCcw />
            Reset
          </Button>
          <Button size="sm" onClick={handleSave} disabled={!dirty || saving}>
            <Save />
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
