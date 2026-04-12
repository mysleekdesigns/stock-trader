import { useState } from 'react'
import { Save, Eye, EyeOff } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

interface SettingsField {
  key: string; label: string; type: 'text' | 'password' | 'number' | 'select'
  placeholder?: string; options?: { label: string; value: string }[]; description?: string
}

const sections: { title: string; fields: SettingsField[] }[] = [
  {
    title: 'Broker API',
    fields: [
      { key: 'broker', label: 'Broker', type: 'select', options: [
        { label: 'Alpaca', value: 'alpaca' }, { label: 'Interactive Brokers', value: 'ibkr' }, { label: 'Paper Trading', value: 'paper' },
      ]},
      { key: 'apiKey', label: 'API Key', type: 'password', placeholder: 'Enter API key' },
      { key: 'apiSecret', label: 'API Secret', type: 'password', placeholder: 'Enter API secret' },
      { key: 'baseUrl', label: 'Base URL', type: 'text', placeholder: 'https://paper-api.alpaca.markets' },
    ],
  },
  {
    title: 'Data Providers',
    fields: [
      { key: 'polygonKey', label: 'Polygon.io API Key', type: 'password', placeholder: 'Enter Polygon API key' },
      { key: 'dataSource', label: 'Primary Data Source', type: 'select', options: [
        { label: 'Yahoo Finance', value: 'yfinance' }, { label: 'Polygon.io', value: 'polygon' }, { label: 'Alpaca', value: 'alpaca' },
      ]},
    ],
  },
  {
    title: 'Risk Limits',
    fields: [
      { key: 'maxPositionSize', label: 'Max Position Size (%)', type: 'number', placeholder: '10', description: 'Maximum allocation per single position' },
      { key: 'maxDrawdown', label: 'Max Drawdown (%)', type: 'number', placeholder: '5', description: 'Stop trading if drawdown exceeds this' },
      { key: 'maxDailyLoss', label: 'Max Daily Loss ($)', type: 'number', placeholder: '5000', description: 'Maximum allowed loss per day' },
      { key: 'maxGrossExposure', label: 'Max Gross Exposure (%)', type: 'number', placeholder: '200', description: 'Maximum long + short exposure' },
      { key: 'maxConcentration', label: 'Max Sector Concentration (%)', type: 'number', placeholder: '30', description: 'Maximum exposure to a single sector' },
    ],
  },
  {
    title: 'Notifications',
    fields: [
      { key: 'webhookUrl', label: 'Webhook URL', type: 'text', placeholder: 'https://hooks.slack.com/...' },
      { key: 'emailAlerts', label: 'Email', type: 'text', placeholder: 'alerts@example.com' },
    ],
  },
]

export default function Settings() {
  const [values, setValues] = useState<Record<string, string>>({})
  const [visiblePasswords, setVisiblePasswords] = useState<Set<string>>(new Set())
  const [saved, setSaved] = useState(false)

  function handleChange(key: string, value: string) { setValues((prev) => ({ ...prev, [key]: value })); setSaved(false) }
  function togglePasswordVisibility(key: string) {
    setVisiblePasswords((prev) => { const next = new Set(prev); next.has(key) ? next.delete(key) : next.add(key); return next })
  }
  function handleSave() { setSaved(true); setTimeout(() => setSaved(false), 3000) }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Settings</h1>
        <Button onClick={handleSave}><Save className="w-4 h-4" />{saved ? 'Saved!' : 'Save Changes'}</Button>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {sections.map((section) => (
          <Card key={section.title}>
            <h3 className="text-sm font-semibold">{section.title}</h3>
            <CardContent className="space-y-4">
              {section.fields.map((field) => (
                <div key={field.key} className="space-y-1.5">
                  <Label className="text-xs text-muted-foreground">{field.label}</Label>
                  {field.type === 'select' ? (
                    <Select value={values[field.key] || ''} onValueChange={(v) => handleChange(field.key, v)}>
                      <SelectTrigger><SelectValue placeholder="Select..." /></SelectTrigger>
                      <SelectContent>
                        {field.options?.map((opt) => <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  ) : field.type === 'password' ? (
                    <div className="relative">
                      <Input type={visiblePasswords.has(field.key) ? 'text' : 'password'} value={values[field.key] || ''}
                        onChange={(e) => handleChange(field.key, e.target.value)} placeholder={field.placeholder} className="pr-10" />
                      <Button type="button" variant="ghost" size="icon-xs" onClick={() => togglePasswordVisibility(field.key)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
                        {visiblePasswords.has(field.key) ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </Button>
                    </div>
                  ) : (
                    <Input type={field.type} value={values[field.key] || ''} onChange={(e) => handleChange(field.key, e.target.value)} placeholder={field.placeholder} />
                  )}
                  {field.description && <p className="text-[10px] text-muted-foreground">{field.description}</p>}
                </div>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
