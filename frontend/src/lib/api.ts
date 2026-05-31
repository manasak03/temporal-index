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
