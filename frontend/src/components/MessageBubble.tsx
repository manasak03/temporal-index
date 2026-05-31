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
  loadingMeta?: LoadingMeta
  alpha?: number
}

export function MessageBubble({ message, isLoading, loadingMeta, alpha }: MessageBubbleProps) {
  const isUser = message.role === "user"

  return (
    <div
      className={cn(
        "flex flex-col py-5",
        isUser ? "items-end" : "items-start",
      )}
    >
      <div
        className={cn(
          "max-w-3xl text-[15px] leading-7 tracking-tight",
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

        {!isLoading &&
          (isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <MarkdownMessage
              content={message.content}
              className={message.metrics ? "mt-3" : undefined}
            />
          ))}
      </div>

      {!isUser && !isLoading && message.sources && message.sources.length > 0 && (
        <RetrievalInsights
          sources={message.sources}
          alpha={alpha}
        />
      )}
    </div>
  )
}
