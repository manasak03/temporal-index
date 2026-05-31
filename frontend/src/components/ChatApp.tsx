import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"
import { MessageBubble } from "@/components/MessageBubble"
import { PipelineStatusSheet } from "@/components/PipelineStatusSheet"
import { SettingsDialog } from "@/components/SettingsDialog"
import { fetchHealth, sendChatMessage } from "@/lib/api"
import { cn } from "@/lib/utils"
import type {
  ChatMessage,
  ChatSession,
  GenerationSettings,
  HealthResponse,
  ModelOption,
} from "@/types/chat"
import { DEFAULT_GENERATION_SETTINGS, DEFAULT_MODELS } from "@/types/chat"
import {
  AlertCircle,
  Clock,
  MessageSquarePlus,
  SendHorizontal,
  Settings2,
} from "lucide-react"

function createId() {
  return crypto.randomUUID()
}

const WELCOME_MESSAGE: ChatMessage = {
  id: createId(),
  role: "assistant",
  content:
    "Welcome to Temporal Index — answers are grounded in hybrid-retrieved catalog and waitlist data. Ask about references, RRP, complications, or wait times.",
  createdAt: Date.now(),
}

const STARTER_PROMPTS = [
  "What is the wait time for a steel Submariner?",
  "Compare Daytona Panda retail vs grey market.",
  "Which GMT-Master II references are in the catalog?",
]

function createSession(title = "New conversation"): ChatSession {
  const now = Date.now()
  return {
    id: createId(),
    title,
    messages: [{ ...WELCOME_MESSAGE, id: createId(), createdAt: now }],
    createdAt: now,
    updatedAt: now,
  }
}

function deriveTitle(messages: ChatMessage[]) {
  const firstUser = messages.find((message) => message.role === "user")
  if (!firstUser) return "New conversation"
  return firstUser.content.length > 42
    ? `${firstUser.content.slice(0, 42)}…`
    : firstUser.content
}

function HeaderNav({ onNewChat }: { onNewChat: () => void }) {
  const { open, openMobile, isMobile } = useSidebar()
  const sidebarVisible = isMobile ? openMobile : open

  return (
    <div className="flex items-center gap-2">
      <SidebarTrigger className="-ml-1 shrink-0 text-muted-foreground hover:text-foreground" />
      <Button
        size="sm"
        onClick={onNewChat}
        aria-hidden={sidebarVisible}
        tabIndex={sidebarVisible ? -1 : 0}
        className={cn(
          "shrink-0 gap-2 overflow-hidden bg-primary text-primary-foreground transition-all duration-200 ease-out hover:bg-primary/90",
          sidebarVisible
            ? "pointer-events-none max-w-0 scale-95 px-0 opacity-0"
            : "max-w-[9.5rem] scale-100 px-3 opacity-100",
        )}
      >
        <MessageSquarePlus />
        <span className="whitespace-nowrap">New Chat</span>
      </Button>
    </div>
  )
}

interface ChatMainPanelProps {
  sessions: ChatSession[]
  activeSessionId: string
  messages: ChatMessage[]
  loading: boolean
  loadingMeta: {
    model: string
    backend: string
    alpha: number
    startedAt: number
  } | null
  error: string | null
  health: HealthResponse | null
  settings: GenerationSettings
  modelOptions: ModelOption[]
  degraded: boolean
  input: string
  bottomRef: RefObject<HTMLDivElement | null>
  onNewChat: () => void
  onSelectSession: (sessionId: string) => void
  onOpenSettings: () => void
  onInputChange: (value: string) => void
  onSubmit: (text: string) => void
  onModelChange: (model: string) => void
  onSettingsChange: (settings: GenerationSettings) => void
  onRefreshHealth: () => void
}

function ChatMainPanel({
  sessions,
  activeSessionId,
  messages,
  loading,
  loadingMeta,
  error,
  health,
  settings,
  modelOptions,
  degraded,
  input,
  bottomRef,
  onNewChat,
  onSelectSession,
  onOpenSettings,
  onInputChange,
  onSubmit,
  onModelChange,
  onSettingsChange,
  onRefreshHealth,
}: ChatMainPanelProps) {
  const ollamaModels = modelOptions.filter((model) => model.backend === "ollama")
  const mlxModels = modelOptions.filter((model) => model.backend === "mlx")
  const selected = modelOptions.find((model) => model.id === settings.model)

  return (
    <>
      <Sidebar
        collapsible="offcanvas"
        className="border-sidebar-border bg-zinc-950 text-sidebar-foreground"
      >
        <SidebarHeader className="border-b border-sidebar-border/60 p-3">
          <Button
            className="w-full justify-start gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
            onClick={onNewChat}
          >
            <MessageSquarePlus data-icon="inline-start" />
            New Chat
          </Button>
        </SidebarHeader>

        <SidebarContent className="px-2 py-3">
          <SidebarMenu>
            {sessions.map((session) => (
              <SidebarMenuItem key={session.id}>
                <SidebarMenuButton
                  isActive={session.id === activeSessionId}
                  onClick={() => onSelectSession(session.id)}
                  className="h-9"
                >
                  <span className="truncate text-sm">{session.title}</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarContent>

        <SidebarFooter className="border-t border-sidebar-border/60 p-3">
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton onClick={onOpenSettings} className="h-9">
                <Settings2 />
                <span>Settings</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarFooter>
      </Sidebar>

      <SidebarInset className="flex h-svh min-h-0 flex-col overflow-hidden bg-background">
        <header className="sticky top-0 z-20 flex shrink-0 flex-wrap items-center gap-2 border-b border-border/60 bg-background/80 px-3 py-2.5 backdrop-blur-md sm:gap-3 sm:px-4 sm:py-3 md:px-6">
          <HeaderNav onNewChat={onNewChat} />

          <div className="flex min-w-0 flex-1 basis-full items-center gap-2 sm:basis-auto sm:gap-3">
            <div className="hidden items-center gap-2 md:flex">
              <Clock className="size-4 text-primary" />
              <span className="text-sm font-medium tracking-tight">Temporal Index</span>
            </div>

            <Select
              value={settings.model}
              onValueChange={(value) => {
                if (value) onModelChange(value)
              }}
            >
              <SelectTrigger
                size="sm"
                className="ml-auto w-full min-w-0 max-w-full border-border/60 sm:ml-0 sm:max-w-[12.5rem] md:max-w-[14rem]"
              >
                <SelectValue placeholder="Model">
                  {selected ? (
                    <span className="flex items-center gap-2 truncate">
                      <span className="truncate">{selected.id}</span>
                      <Badge variant="outline" className="shrink-0 px-1 py-0 text-[9px] uppercase">
                        {selected.backend}
                      </Badge>
                    </span>
                  ) : (
                    "Model"
                  )}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {ollamaModels.length > 0 && (
                  <SelectGroup>
                    <span className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
                      Ollama
                    </span>
                    {ollamaModels.map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        <span className="flex w-full items-center justify-between gap-2">
                          <span className="truncate">{model.id}</span>
                          {!model.ready && (
                            <span className="text-[10px] text-muted-foreground">not pulled</span>
                          )}
                        </span>
                      </SelectItem>
                    ))}
                  </SelectGroup>
                )}
                {mlxModels.length > 0 && (
                  <SelectGroup>
                    <span className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
                      Apple MLX
                    </span>
                    {mlxModels.map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        <span className="flex w-full items-center justify-between gap-2">
                          <span className="truncate">{model.id}</span>
                          <Badge variant="secondary" className="shrink-0 px-1 py-0 text-[9px]">
                            mlx
                          </Badge>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectGroup>
                )}
              </SelectContent>
            </Select>
          </div>

          <div className="ml-auto flex shrink-0 items-center gap-2">
            <Badge
              variant={degraded ? "destructive" : "outline"}
              className="hidden font-mono text-[10px] md:inline-flex"
            >
              {health?.status ?? "…"}
            </Badge>
            <PipelineStatusSheet
              health={health}
              settings={settings}
              modelOptions={modelOptions}
              onModelSelect={onModelChange}
              onSettingsChange={onSettingsChange}
              onRefresh={onRefreshHealth}
            />
          </div>
        </header>

        {(degraded || error) && (
          <div className="shrink-0 px-4 pt-3 md:px-6">
            {degraded && (
              <Alert variant="destructive" className="mb-3">
                <AlertCircle />
                <AlertTitle>System degraded</AlertTitle>
                <AlertDescription>
                  Ensure Milvus index exists. For Ollama models, keep Ollama running; for MLX
                  models (e.g. <code>gemma4:31b-mlx</code>), mlx-lm loads weights on first use.
                </AlertDescription>
              </Alert>
            )}
            {error && (
              <Alert className="mb-3">
                <AlertCircle />
                <AlertTitle>Request error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </div>
        )}

        <ScrollArea className="min-h-0 flex-1">
          <div className="mx-auto flex w-full max-w-3xl flex-col px-3 py-4 sm:px-4 sm:py-6 md:px-8">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                isLoading={loading && message.content === "" && message.role === "assistant"}
                loadingMeta={
                  loading && message.content === "" && message.role === "assistant"
                    ? (loadingMeta ?? undefined)
                    : undefined
                }
                alpha={settings.alpha}
              />
            ))}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>

        <footer className="shrink-0 border-t border-border/60 bg-background/90 px-3 py-3 backdrop-blur-md sm:px-4 sm:py-4 md:px-6">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-2 sm:gap-3">
            <div className="hidden gap-2 overflow-x-auto pb-0.5 sm:flex sm:flex-wrap">
              {STARTER_PROMPTS.map((prompt) => (
                <Button
                  key={prompt}
                  variant="outline"
                  size="sm"
                  className="border-border/60 text-xs text-muted-foreground hover:text-foreground"
                  onClick={() => void onSubmit(prompt)}
                  disabled={loading}
                >
                  {prompt}
                </Button>
              ))}
            </div>
            <form
              className="flex gap-2 sm:gap-2"
              onSubmit={(event) => {
                event.preventDefault()
                void onSubmit(input)
              }}
            >
              <Input
                value={input}
                onChange={(event) => onInputChange(event.target.value)}
                placeholder="Ask about a reference, RRP, or wait time…"
                disabled={loading}
                className="h-11 border-border/60 bg-muted/20"
              />
              <Button type="submit" disabled={loading || !input.trim()} className="h-11 px-4">
                <SendHorizontal />
                <span className="sr-only sm:not-sr-only">Send</span>
              </Button>
            </form>
          </div>
        </footer>
      </SidebarInset>
    </>
  )
}

export function ChatApp() {
  const [sessions, setSessions] = useState<ChatSession[]>(() => [createSession()])
  const [activeSessionId, setActiveSessionId] = useState(() => sessions[0]?.id ?? createId())
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [loadingMeta, setLoadingMeta] = useState<{
    model: string
    backend: string
    alpha: number
    startedAt: number
  } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_GENERATION_SETTINGS)
  const bottomRef = useRef<HTMLDivElement>(null)

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) ?? sessions[0],
    [sessions, activeSessionId],
  )

  const messages = activeSession?.messages ?? []

  const modelOptions = useMemo((): ModelOption[] => {
    const fromHealth = health?.models ?? []
    const byId = new Map<string, ModelOption>()

    for (const id of DEFAULT_MODELS) {
      byId.set(id, { id, backend: id.endsWith("-mlx") ? "mlx" : "ollama" })
    }
    for (const entry of fromHealth) {
      byId.set(entry.id, entry)
    }
    if (settings.model && !byId.has(settings.model)) {
      byId.set(settings.model, {
        id: settings.model,
        backend: settings.model.endsWith("-mlx") ? "mlx" : "ollama",
      })
    }
    return Array.from(byId.values())
  }, [health, settings.model])

  const refreshHealth = useCallback(async () => {
    try {
      const payload = await fetchHealth()
      setHealth(payload)
      if (payload.ollama?.configured_model && !settings.model) {
        setSettings((prev) => ({ ...prev, model: payload.ollama!.configured_model! }))
      }
    } catch (err) {
      setHealth(null)
      setError(err instanceof Error ? err.message : "Unable to reach API server.")
    }
  }, [settings.model])

  useEffect(() => {
    void refreshHealth()
  }, [refreshHealth])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading, activeSessionId])

  const updateSessionMessages = useCallback(
    (sessionId: string, updater: (messages: ChatMessage[]) => ChatMessage[]) => {
      setSessions((prev) =>
        prev.map((session) => {
          if (session.id !== sessionId) return session
          const nextMessages = updater(session.messages)
          return {
            ...session,
            messages: nextMessages,
            title: deriveTitle(nextMessages),
            updatedAt: Date.now(),
          }
        }),
      )
    },
    [],
  )

  const handleNewChat = () => {
    const session = createSession()
    setSessions((prev) => [session, ...prev])
    setActiveSessionId(session.id)
    setInput("")
    setError(null)
  }

  const submit = async (text: string) => {
    const query = text.trim()
    if (!query || loading || !activeSession) return

    setError(null)
    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: query,
      createdAt: Date.now(),
    }
    const pendingId = createId()
    const sessionId = activeSession.id

    updateSessionMessages(sessionId, (prev) => [
      ...prev,
      userMessage,
      { id: pendingId, role: "assistant", content: "", createdAt: Date.now() },
    ])
    setInput("")
    setLoading(true)
    const selectedModel = modelOptions.find((option) => option.id === settings.model)
    setLoadingMeta({
      model: settings.model,
      backend: selectedModel?.backend ?? (settings.model.endsWith("-mlx") ? "mlx" : "ollama"),
      alpha: settings.alpha,
      startedAt: Date.now(),
    })

    try {
      const history = messages
        .filter((message) => message.content)
        .map((message) => ({ role: message.role, content: message.content }))

      const response = await sendChatMessage({
        query,
        history,
        model: settings.model,
        temperature: settings.temperature,
        top_p: settings.topP,
        alpha: settings.alpha,
      })

      updateSessionMessages(sessionId, (prev) =>
        prev.map((message) =>
          message.id === pendingId
            ? {
                ...message,
                content: response.answer,
                sources: response.sources,
                latencyMs: response.latency_ms,
                metrics: response.metrics,
              }
            : message,
        ),
      )
    } catch (err) {
      updateSessionMessages(sessionId, (prev) =>
        prev.filter((message) => message.id !== pendingId),
      )
      setError(err instanceof Error ? err.message : "Chat request failed.")
    } finally {
      setLoading(false)
      setLoadingMeta(null)
    }
  }

  const degraded = health?.status === "degraded"

  return (
    <SidebarProvider defaultOpen className="flex h-svh min-h-0 overflow-hidden bg-background">
      <ChatMainPanel
        sessions={sessions}
        activeSessionId={activeSessionId}
        messages={messages}
        loading={loading}
        loadingMeta={loadingMeta}
        error={error}
        health={health}
        settings={settings}
        modelOptions={modelOptions}
        degraded={degraded}
        input={input}
        bottomRef={bottomRef}
        onNewChat={handleNewChat}
        onSelectSession={(sessionId) => {
          setActiveSessionId(sessionId)
          setError(null)
        }}
        onOpenSettings={() => setSettingsOpen(true)}
        onInputChange={setInput}
        onSubmit={submit}
        onModelChange={(model) => setSettings((prev) => ({ ...prev, model }))}
        onSettingsChange={setSettings}
        onRefreshHealth={() => void refreshHealth()}
      />

      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        settings={settings}
        onSettingsChange={setSettings}
      />
    </SidebarProvider>
  )
}
