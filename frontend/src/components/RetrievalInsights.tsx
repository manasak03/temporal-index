import { Badge } from "@/components/ui/badge"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { RetrievalMethod, SourceItem } from "@/types/chat"
import { FileText, Layers } from "lucide-react"

interface RetrievalInsightsProps {
  sources: SourceItem[]
  latencyMs?: number
  alpha?: number
}

function formatScore(score?: number | null) {
  if (score == null) return "n/a"
  return score.toFixed(6)
}

function resolveSourcePath(source: SourceItem) {
  if (source.source_path) return source.source_path

  const parts = [
    source.collection && `data/chunks/${source.collection}`,
    source.references && `refs/${source.references}`,
    source.chunk_type,
    source.model,
  ].filter(Boolean)

  return parts.length > 0 ? parts.join(" · ") : "Unknown source location"
}

function resolveMethod(source: SourceItem, alpha = 0.7): RetrievalMethod {
  if (source.retrieval_method) return source.retrieval_method
  if (alpha >= 0.65) return "dense"
  if (alpha <= 0.35) return "sparse"
  return "hybrid"
}

function methodLabel(method: RetrievalMethod) {
  switch (method) {
    case "dense":
      return "Dense"
    case "sparse":
      return "Sparse (BM25)"
    default:
      return "Hybrid Fusion"
  }
}

function methodVariant(method: RetrievalMethod): "default" | "secondary" | "outline" {
  switch (method) {
    case "dense":
      return "default"
    case "sparse":
      return "secondary"
    default:
      return "outline"
  }
}

export function RetrievalInsights({ sources, latencyMs, alpha = 0.7 }: RetrievalInsightsProps) {
  if (!sources.length) return null

  return (
    <Accordion className="mt-2 w-full max-w-2xl">
      <AccordionItem value="insights" className="border-none">
        <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:text-foreground hover:no-underline">
          <div className="flex items-center gap-2">
            <Layers className="size-3.5 text-primary" />
            <span>Grounding sources</span>
            <Badge variant="outline" className="font-mono text-[10px]">
              {sources.length} chunks
            </Badge>
            {latencyMs != null && (
              <Badge variant="outline" className="font-mono text-[10px]">
                {latencyMs} ms total
              </Badge>
            )}
          </div>
        </AccordionTrigger>
        <AccordionContent className="max-h-[min(50vh,22rem)] overflow-y-auto overscroll-contain">
          <div className="flex flex-col gap-2.5 pb-1 pr-1">
            {sources.map((source, index) => {
              const method = resolveMethod(source, alpha)
              const sourcePath = resolveSourcePath(source)

              return (
                <Card
                  key={source.id ?? index}
                  className="border-border/50 bg-muted/15 py-0 shadow-none"
                >
                  <CardHeader className="gap-2 px-3 pt-3 pb-2">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <CardTitle className="text-xs font-medium">
                        Rank #{source.rank ?? index + 1}
                      </CardTitle>
                      <Badge variant={methodVariant(method)} className="text-[10px]">
                        {methodLabel(method)}
                      </Badge>
                      <Badge variant="outline" className="font-mono text-[10px]">
                        IP {formatScore(source.score)}
                      </Badge>
                      {source.chunk_type && (
                        <Badge variant="outline" className="text-[10px]">
                          {source.chunk_type}
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-start gap-1.5 text-[11px] text-muted-foreground">
                      <FileText className="mt-0.5 size-3 shrink-0 text-primary/80" />
                      <span className="font-mono leading-relaxed break-all">{sourcePath}</span>
                    </div>
                  </CardHeader>
                  <CardContent className="px-3 pb-3">
                    <ScrollArea className="max-h-28 rounded-md border border-border/40 bg-background/40">
                      <pre className="whitespace-pre-wrap p-2 font-mono text-[10px] leading-relaxed text-muted-foreground">
                        {source.text ?? "No chunk text returned."}
                      </pre>
                    </ScrollArea>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}
