import type {
  ChatRequestPayload,
  ChatResponsePayload,
  HealthResponse,
} from "@/types/chat"

const API_BASE = import.meta.env.VITE_API_BASE ?? ""

async function parseError(response: Response): Promise<string> {
  try {
    const payload = await response.json()
    if (typeof payload.detail === "string") return payload.detail
    return JSON.stringify(payload.detail ?? payload)
  } catch {
    return response.statusText || "Request failed"
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/api/health`)
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  return response.json()
}

export async function sendChatMessage(
  payload: ChatRequestPayload,
): Promise<ChatResponsePayload> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  return response.json()
}

type StreamEvent =
  | { type: "sources"; sources: ChatResponsePayload["sources"]; retrieval_ms?: number }
  | { type: "token"; delta: string }
  | { type: "done"; answer: string; sources: ChatResponsePayload["sources"]; latency_ms: number; model?: string; backend?: ChatResponsePayload["backend"]; metrics?: ChatResponsePayload["metrics"] }
  | { type: "error"; detail: string }

export interface StreamChatHandlers {
  onSources?: (sources: ChatResponsePayload["sources"]) => void
  onToken: (delta: string) => void
  onDone: (result: ChatResponsePayload) => void
}

export async function streamChatMessage(
  payload: ChatRequestPayload,
  handlers: StreamChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  })
  if (!response.ok) {
    throw new Error(await parseError(response))
  }
  if (!response.body) {
    throw new Error("Streaming response has no body.")
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  const handleEvent = (event: StreamEvent) => {
    if (event.type === "sources") {
      handlers.onSources?.(event.sources)
      return
    }
    if (event.type === "token") {
      handlers.onToken(event.delta)
      return
    }
    if (event.type === "done") {
      handlers.onDone({
        answer: event.answer,
        sources: event.sources,
        latency_ms: event.latency_ms,
        model: event.model,
        backend: event.backend,
        metrics: event.metrics,
      })
      return
    }
    if (event.type === "error") {
      throw new Error(event.detail)
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf("\n\n")
    while (boundary !== -1) {
      const raw = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of raw.split("\n")) {
        if (!line.startsWith("data:")) continue
        const data = line.slice(5).trim()
        if (!data) continue
        handleEvent(JSON.parse(data) as StreamEvent)
      }
      boundary = buffer.indexOf("\n\n")
    }
  }

  if (buffer.trim()) {
    for (const line of buffer.split("\n")) {
      if (!line.startsWith("data:")) continue
      const data = line.slice(5).trim()
      if (data) handleEvent(JSON.parse(data) as StreamEvent)
    }
  }
}
