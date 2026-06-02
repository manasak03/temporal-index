import { cn } from "@/lib/utils"
import type { ChatMessage } from "@/types/chat"
import { MarkdownMessage } from "@/components/MarkdownMessage"
import { PipelineMetrics } from "@/components/PipelineMetrics"
import { RetrievalInsights } from "@/components/RetrievalInsights"

interface LoadingMeta {
  model: string
  backend: string
  alpha: number
  startedAt: number
}

interface MessageBubbleProps {
  message: ChatMessage
  isLoading?: boolean
  isStreaming?: boolean
  loadingMeta?: LoadingMeta
  alpha?: number
}

export function MessageBubble({
  message,
  isLoading,
  isStreaming,
  loadingMeta,
  alpha,
}: MessageBubbleProps) {
  const isUser = message.role === "user"
  const showAssistantContent = !isUser && message.content && (isStreaming || !isLoading)

  return (
    <div
      className={cn(
        "flex flex-col py-5",
        isUser ? "items-end" : "items-start",
      )}
    >
      <div
        className={cn(
          "max-w-3xl text-[15px] leading-7",
          isUser ? "text-right text-foreground/90" : "text-foreground",
        )}
      >
        {!isUser && (isLoading || message.metrics) && (
          <PipelineMetrics
            metrics={message.metrics}
            loading={isLoading}
            loadingMeta={loadingMeta}
          />
        )}

        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : showAssistantContent ? (
          isStreaming ? (
            <p
              className={cn(
                "whitespace-pre-wrap font-sans",
                message.metrics ? "mt-3" : undefined,
              )}
            >
              {message.content}
              <span
                className="ml-0.5 inline-block h-[1.1em] w-0.5 animate-pulse bg-primary align-[-0.15em]"
                aria-hidden
              />
            </p>
          ) : (
            <MarkdownMessage
              content={message.content}
              className={message.metrics ? "mt-3" : undefined}
            />
          )
        ) : null}
      </div>

      {!isUser && !isLoading && !isStreaming && message.sources && message.sources.length > 0 && (
        <RetrievalInsights
          sources={message.sources}
          alpha={alpha}
        />
      )}
    </div>
  )
}
