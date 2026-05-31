export type MessageRole = "user" | "assistant"

export type RetrievalMethod = "dense" | "sparse" | "hybrid"

export interface SourceItem {
  id?: string | null
  rank?: number | null
  score?: number | null
  chunk_type?: string | null
  collection?: string | null
  model?: string | null
  references?: string | null
  text?: string | null
  retrieval_method?: RetrievalMethod | null
  source_path?: string | null
}

export interface PipelineMetrics {
  retrieval_ms: number
  generation_ms: number
  total_ms: number
  chunks_retrieved: number
  context_chars: number
  model: string
  backend: string
  fusion_alpha: number
  fusion_method: string
  prompt_tokens?: number | null
  completion_tokens?: number | null
  total_tokens?: number | null
  tokens_per_second?: number | null
}

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  sources?: SourceItem[]
  latencyMs?: number
  metrics?: PipelineMetrics
  createdAt: number
}

export interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

export interface GenerationSettings {
  model: string
  temperature: number
  topP: number
  alpha: number
}

export interface ChatRequestPayload {
  query: string
  history: Array<{ role: MessageRole; content: string }>
  model?: string
  temperature?: number
  top_p?: number
  alpha?: number
}

export interface ChatResponsePayload {
  answer: string
  sources: SourceItem[]
  latency_ms: number
  model?: string
  backend?: ModelBackend
  metrics?: PipelineMetrics
}

export interface HealthResponse {
  status: "ok" | "degraded"
  milvus: {
    db_path: string
    db_exists: boolean
    collection: string
    collection_ready: boolean
  }
  ollama: {
    reachable?: boolean
    available?: boolean
    configured_model?: string
    available_models?: string[]
    model_ready?: boolean
    error?: string
  }
  mlx?: {
    available?: boolean
    models?: string[]
    loaded_paths?: string[]
    error?: string
    catalog?: Array<{ id: string; path?: string | null; loaded?: boolean }>
  }
  models?: ModelOption[]
  any_backend_available?: boolean
}

export const DEFAULT_MODELS = [
  "qwen2.5:3b",
  "gemma2",
  "llama3",
  "qwen3.6:27b",
  "gemma4:31b-mlx",
  "gemma4:26b",
  "llama3.1:latest",
] as const

export type ModelBackend = "ollama" | "mlx"

export interface ModelOption {
  id: string
  backend: ModelBackend
  ready?: boolean
  detail?: string | null
  mlx_path?: string | null
}

export const DEFAULT_GENERATION_SETTINGS: GenerationSettings = {
  model: "llama3.1:latest",
  temperature: 0.2,
  topP: 0.9,
  alpha: 0.7,
}
