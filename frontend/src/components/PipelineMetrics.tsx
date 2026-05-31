import { useEffect, useState } from "react"
import { Badge } from "@/components/ui/badge"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { PipelineMetrics as PipelineMetricsType } from "@/types/chat"
import { Activity, Cpu, Database, Loader2, Zap } from "lucide-react"

interface LoadingMeta {
  model: string
  backend: string
  alpha: number
  startedAt: number
}

interface PipelineMetricsProps {
  metrics?: PipelineMetricsType
  loading?: boolean
  loadingMeta?: LoadingMeta
}

function formatDuration(ms: number) {
  if (ms < 1000) return `${ms} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

function formatNumber(value?: number | null) {
  if (value == null) return "n/a"
  return value.toLocaleString()
}

function StageRow({
  label,
  detail,
  status,
}: {
  label: string
  detail: string
  status: "pending" | "active" | "done"
}) {
  return (
    <div className="flex items-start gap-2.5 text-xs">
      <div className="mt-0.5 flex size-4 shrink-0 items-center justify-center">
        {status === "active" ? (
          <Loader2 className="size-3.5 animate-spin text-primary" />
        ) : (
          <span
            className={cn(
              "size-2 rounded-full",
              status === "done" ? "bg-primary" : "bg-muted-foreground/30",
            )}
          />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <p
          className={cn(
            "font-medium",
            status === "pending" ? "text-muted-foreground/60" : "text-foreground",
          )}
        >
          {label}
        </p>
        <p className="font-mono text-[10px] text-muted-foreground">{detail}</p>
      </div>
    </div>
  )
}

function LoadingPanel({ meta }: { meta: LoadingMeta }) {
  const [elapsedMs, setElapsedMs] = useState(0)
  const [phase, setPhase] = useState<"retrieval" | "generation">("retrieval")

  useEffect(() => {
    const tick = () => setElapsedMs(Date.now() - meta.startedAt)
    tick()
    const interval = window.setInterval(tick, 100)
    return () => window.clearInterval(interval)
  }, [meta.startedAt])

  useEffect(() => {
    const timer = window.setTimeout(() => setPhase("generation"), 800)
    return () => window.clearTimeout(timer)
  }, [])

  return (
    <Card className="mt-1 w-full max-w-2xl border-border/50 bg-muted/10 py-0 shadow-none">
      <CardContent className="flex flex-col gap-3 px-3 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <Activity className="size-3.5 text-primary" />
          <span className="text-xs font-medium">Running RAG pipeline</span>
          <Badge variant="outline" className="font-mono text-[10px]">
            {formatDuration(elapsedMs)}
          </Badge>
          <Badge variant="secondary" className="font-mono text-[10px]">
            {meta.model} · {meta.backend}
          </Badge>
        </div>

        <div className="flex flex-col gap-2.5 border-l border-border/40 pl-3">
          <StageRow
            label="Hybrid retrieval"
            detail={`Milvus dense+sparse fusion · α=${meta.alpha}`}
            status={phase === "retrieval" ? "active" : "done"}
          />
          <StageRow
            label="Local LLM generation"
            detail={`${meta.model} via ${meta.backend}`}
            status={phase === "generation" ? "active" : "pending"}
          />
        </div>

        <p className="font-mono text-[10px] text-muted-foreground">
          Metrics (tokens, throughput, stage timings) will appear when the response completes.
        </p>
      </CardContent>
    </Card>
  )
}

function MetricsSummary({ metrics }: { metrics: PipelineMetricsType }) {
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      <Badge variant="outline" className="gap-1 font-mono text-[10px]">
        <Zap className="size-3" />
        {formatDuration(metrics.total_ms)}
      </Badge>
      {metrics.completion_tokens != null && (
        <Badge variant="outline" className="gap-1 font-mono text-[10px]">
          <Cpu className="size-3" />
          {formatNumber(metrics.completion_tokens)} out tok
        </Badge>
      )}
      {metrics.tokens_per_second != null && (
        <Badge variant="outline" className="gap-1 font-mono text-[10px]">
          {metrics.tokens_per_second.toFixed(1)} tok/s
        </Badge>
      )}
      <Badge variant="outline" className="gap-1 font-mono text-[10px]">
        <Database className="size-3" />
        {metrics.chunks_retrieved} chunks · {formatDuration(metrics.retrieval_ms)} retrieval
      </Badge>
    </div>
  )
}

export function PipelineMetrics({ metrics, loading, loadingMeta }: PipelineMetricsProps) {
  if (loading && loadingMeta) {
    return <LoadingPanel meta={loadingMeta} />
  }

  if (!metrics) return null

  return (
    <div className="w-full max-w-2xl">
      <MetricsSummary metrics={metrics} />
      <Accordion className="mt-1">
        <AccordionItem value="pipeline-metrics" className="border-none">
          <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:text-foreground hover:no-underline">
            Pipeline analytics
          </AccordionTrigger>
          <AccordionContent className="max-h-[min(40vh,16rem)] overflow-y-auto overscroll-contain">
            <Card className="border-border/50 bg-muted/10 py-0 shadow-none">
              <CardContent className="grid gap-2 px-3 py-3 font-mono text-[10px] leading-relaxed text-muted-foreground sm:grid-cols-2">
                <div>
                  <span className="text-foreground/70">Model</span>
                  <p>
                    {metrics.model} ({metrics.backend})
                  </p>
                </div>
                <div>
                  <span className="text-foreground/70">Fusion</span>
                  <p>
                    {metrics.fusion_method} · α={metrics.fusion_alpha}
                  </p>
                </div>
                <div>
                  <span className="text-foreground/70">Retrieval</span>
                  <p>
                    {formatDuration(metrics.retrieval_ms)} · {metrics.chunks_retrieved} chunks ·{" "}
                    {formatNumber(metrics.context_chars)} ctx chars
                  </p>
                </div>
                <div>
                  <span className="text-foreground/70">Generation</span>
                  <p>{formatDuration(metrics.generation_ms)}</p>
                </div>
                <div>
                  <span className="text-foreground/70">Prompt tokens</span>
                  <p>{formatNumber(metrics.prompt_tokens)}</p>
                </div>
                <div>
                  <span className="text-foreground/70">Completion tokens</span>
                  <p>{formatNumber(metrics.completion_tokens)}</p>
                </div>
                <div>
                  <span className="text-foreground/70">Total tokens</span>
                  <p>{formatNumber(metrics.total_tokens)}</p>
                </div>
                <div>
                  <span className="text-foreground/70">Throughput</span>
                  <p>
                    {metrics.tokens_per_second != null
                      ? `${metrics.tokens_per_second.toFixed(1)} tok/s`
                      : "n/a"}
                  </p>
                </div>
                <div className="sm:col-span-2">
                  <span className="text-foreground/70">End-to-end</span>
                  <p>{formatDuration(metrics.total_ms)}</p>
                </div>
              </CardContent>
            </Card>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  )
}
