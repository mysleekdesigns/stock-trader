import { useEffect, useRef, useState, useCallback } from 'react'

interface WebSocketMessage {
  channel: string
  data: unknown
  timestamp: string
}

interface UseWebSocketReturn {
  messages: WebSocketMessage[]
  lastMessage: WebSocketMessage | null
  send: (data: unknown) => void
  subscribe: (channel: string) => void
  unsubscribe: (channel: string) => void
  connected: boolean
}

export function useWebSocket(url: string): UseWebSocketReturn {
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const reconnectAttemptsRef = useRef(0)
  const subscribedChannelsRef = useRef<Set<string>>(new Set())
  const maxMessages = 100

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const wsUrl = url.startsWith('ws')
      ? url
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}${url}`

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      setConnected(true)
      reconnectAttemptsRef.current = 0

      subscribedChannelsRef.current.forEach((channel) => {
        ws.send(JSON.stringify({ action: 'subscribe', channel }))
      })
    }

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data)
        setLastMessage(message)
        setMessages((prev) => {
          const next = [...prev, message]
          return next.length > maxMessages ? next.slice(-maxMessages) : next
        })
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      setConnected(false)
      wsRef.current = null

      const delay = Math.min(1000 * 2 ** reconnectAttemptsRef.current, 30000)
      reconnectAttemptsRef.current += 1
      reconnectTimeoutRef.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      ws.close()
    }

    wsRef.current = ws
  }, [url])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connect])

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const subscribe = useCallback(
    (channel: string) => {
      subscribedChannelsRef.current.add(channel)
      send({ action: 'subscribe', channel })
    },
    [send],
  )

  const unsubscribe = useCallback(
    (channel: string) => {
      subscribedChannelsRef.current.delete(channel)
      send({ action: 'unsubscribe', channel })
    },
    [send],
  )

  return { messages, lastMessage, send, subscribe, unsubscribe, connected }
}
