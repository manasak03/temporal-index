import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Slider } from "@/components/ui/slider"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { cn } from "@/lib/utils"
import type { GenerationSettings, HealthResponse, ModelOption } from "@/types/chat"
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Circle,
  Cpu,
  Database,
  RefreshCw,
  Server,
  XCircle,
} from "lucide-react"

interface PipelineStatusSheetProps {
  health: HealthResponse | null
  settings: GenerationSettings
  modelOptions: ModelOption[]
  onModelSelect?: (model: string) => void
  onSettingsChange?: (settings: GenerationSettings) => void
  onRefresh?: () => void
}

function sliderValue(values: number | readonly number[]) {
  return Array.isArray(values) ? (values[0] ?? 0) : values
}

function modelStatusLabel(model: ModelOption): string {
  if (model.ready) return "Ready"
  if (model.backend === "mlx") return "Unavailable"
  return "Not pulled"
}

function StatusDot({ ok }: { ok: boolean }) {
  return ok ? (
    <CheckCircle2 className="size-3.5 shrink-0 text-primary" />
  ) : (
    <XCircle className="size-3.5 shrink-0 text-destructive" />
  )
}

function PipelineStep({
  label,
  ok,
  detail,
}: {
  label: string
  ok: boolean
  detail: string
}) {
  return (
    <div className="flex min-w-0 items-start gap-2.5 rounded-lg border border-border/60 bg-muted/15 p-3">
      <StatusDot ok={ok} />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium">{label}</p>
        <p className="text-xs leading-relaxed text-muted-foreground">{detail}</p>
      </div>
    </div>
  )
}

function modelHint(model: ModelOption, inOllama: boolean): string | null {
  if (model.ready) {
    if (model.backend === "mlx" && inOllama) {
      return "Runs via mlx-lm on Apple Silicon (Ollama copy ignored)"
    }
    if (model.backend === "mlx" && model.mlx_path) {
      return model.mlx_path
    }
    return null
  }
  if (model.detail) return model.detail
  if (model.backend === "mlx") return "Install mlx-lm; weights download on first use"
  return `Run: ollama pull ${model.id}`
}

export function PipelineStatusSheet({
  health,
  settings,
  modelOptions,
  onModelSelect,
  onSettingsChange,
  onRefresh,
}: PipelineStatusSheetProps) {
  const degraded = health?.status === "degraded"
  const selected = modelOptions.find((model) => model.id === settings.model)
  const ollamaNames = new Set(health?.ollama?.available_models ?? [])
  const catalogIds = new Set(modelOptions.map((model) => model.id))
  const extraOllamaModels = (health?.ollama?.available_models ?? []).filter(
    (name) => !catalogIds.has(name),
  )
  const readyModels = modelOptions.filter((model) => model.ready).length

  const retrievalOk = Boolean(health?.milvus.collection_ready)
  const ollamaOk = Boolean(health?.ollama?.reachable)
  const mlxOk = Boolean(health?.mlx?.available)
  const generationOk = selected?.ready ?? false

  return (
    <Sheet>
      <SheetTrigger
        render={
          <Button variant="outline" size="sm" className="gap-1.5 border-border/60">
            <Activity data-icon="inline-start" />
            <span className="hidden sm:inline">Pipeline</span>
          </Button>
        }
      />
      <SheetContent
        side="right"
        className="flex h-full max-h-svh w-full flex-col gap-0 overflow-hidden p-0 sm:max-w-lg"
      >
        <SheetHeader className="shrink-0 space-y-1 border-b border-border/60 px-4 pt-4 pb-3">
          <div className="flex items-start justify-between gap-2 pr-8">
            <div className="min-w-0">
              <SheetTitle>Pipeline Status</SheetTitle>
              <SheetDescription className="text-xs leading-relaxed">
                Hybrid retrieval over Milvus Lite, then local generation via Ollama or Apple MLX.
              </SheetDescription>
            </div>
            {onRefresh && (
              <Button
                variant="ghost"
                size="icon-sm"
                className="shrink-0"
                onClick={onRefresh}
                aria-label="Refresh pipeline status"
              >
                <RefreshCw className="size-4" />
              </Button>
            )}
          </div>

          <div className="flex flex-wrap gap-1.5 pt-1">
            <Badge variant={degraded ? "destructive" : "secondary"}>
              API {health?.status ?? "unknown"}
            </Badge>
            <Badge variant={retrievalOk ? "outline" : "destructive"}>
              {readyModels}/{modelOptions.length} models ready
            </Badge>
            {selected && (
              <Badge variant="outline" className="max-w-full truncate font-mono text-[10px]">
                {selected.backend}: {selected.id}
              </Badge>
            )}
          </div>
        </SheetHeader>

        <ScrollArea className="min-h-0 flex-1">
          <div className="flex flex-col gap-5 px-4 py-4 pb-8">
            <section className="flex flex-col gap-2">
              <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                RAG pipeline
              </h3>
              <div className="grid gap-2 sm:grid-cols-[1fr_auto_1fr] sm:items-stretch">
                <PipelineStep
                  label="1 · Hybrid retrieval"
                  ok={retrievalOk}
                  detail={
                    retrievalOk
                      ? `Milvus · ${health?.milvus.collection ?? "temporal_index"} · α=${settings.alpha.toFixed(2)}`
                      : "Index missing — run python -m embedding.hybrid"
                  }
                />
                <div className="hidden items-center justify-center sm:flex">
                  <ArrowRight className="size-4 text-muted-foreground" />
                </div>
                <PipelineStep
                  label="2 · Local generation"
                  ok={generationOk}
                  detail={
                    generationOk
                      ? `${settings.model} via ${selected?.backend ?? "—"}`
                      : selected
                        ? modelHint(selected, ollamaNames.has(selected.id)) ?? "Model not ready"
                        : "Select a model from the header"
                  }
                />
              </div>
            </section>

            <section className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                  Model catalog
                </h3>
                <span className="text-[10px] text-muted-foreground">
                  Tap a ready model to select it
                </span>
              </div>
              <div className="overflow-hidden rounded-lg border border-border/60">
                <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-x-2 border-b border-border/60 bg-muted/25 px-3 py-2 text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
                  <span>Model</span>
                  <span className="text-right">Backend</span>
                  <span className="text-right">Status</span>
                </div>
                <ul className="divide-y divide-border/60">
                  {modelOptions.map((model) => {
                    const isSelected = model.id === settings.model
                    const hint = modelHint(model, ollamaNames.has(model.id))
                    const canSelect = model.ready && onModelSelect

                    return (
                      <li key={model.id}>
                        <button
                          type="button"
                          disabled={!canSelect}
                          onClick={() => canSelect && onModelSelect(model.id)}
                          className={cn(
                            "grid w-full grid-cols-[minmax(0,1fr)_auto_auto] items-start gap-x-2 px-3 py-2.5 text-left text-sm transition-colors",
                            canSelect && "hover:bg-muted/30",
                            isSelected && "bg-primary/10",
                            !canSelect && "cursor-default opacity-80",
                          )}
                        >
                          <div className="min-w-0">
                            <p className="truncate font-mono text-xs">{model.id}</p>
                            {hint && (
                              <p className="mt-0.5 line-clamp-2 text-[10px] leading-relaxed text-muted-foreground">
                                {hint}
                              </p>
                            )}
                          </div>
                          <Badge
                            variant="outline"
                            className="mt-0.5 shrink-0 px-1.5 py-0 text-[9px] uppercase"
                          >
                            {model.backend}
                          </Badge>
                          <div className="flex shrink-0 items-center justify-end gap-1 pt-0.5">
                            {model.ready ? (
                              <>
                                <CheckCircle2 className="size-3.5 text-primary" />
                                <span className="text-[10px] text-primary">Ready</span>
                              </>
                            ) : (
                              <>
                                <Circle className="size-3.5 text-muted-foreground/50" />
                                <span className="text-[10px] text-muted-foreground">
                                  {modelStatusLabel(model)}
                                </span>
                              </>
                            )}
                          </div>
                        </button>
                      </li>
                    )
                  })}
                </ul>
              </div>
              {extraOllamaModels.length > 0 && (
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  Also in Ollama (not in catalog):{" "}
                  <span className="font-mono">{extraOllamaModels.join(", ")}</span>
                </p>
              )}
            </section>

            <section className="flex flex-col gap-3 rounded-lg border border-border/60 bg-muted/15 p-3">
              <div>
                <p className="text-sm font-medium">Generation settings</p>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  Pick a model in the catalog above or the header dropdown. Sliders apply to
                  your next message.
                </p>
              </div>
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-xs">
                <dt className="text-muted-foreground">Model</dt>
                <dd className="truncate font-mono text-foreground">{settings.model}</dd>
                <dt className="text-muted-foreground">Backend</dt>
                <dd className="font-mono text-foreground">{selected?.backend ?? "—"}</dd>
              </dl>
              {onSettingsChange && (
                <div className="flex flex-col gap-4 border-t border-border/40 pt-3">
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="pipeline-temperature" className="text-xs">
                        Temperature
                      </Label>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {settings.temperature.toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      id="pipeline-temperature"
                      min={0}
                      max={1.5}
                      step={0.05}
                      value={[settings.temperature]}
                      onValueChange={(value) =>
                        onSettingsChange({
                          ...settings,
                          temperature: sliderValue(value) || settings.temperature,
                        })
                      }
                    />
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="pipeline-top-p" className="text-xs">
                        Top P
                      </Label>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {settings.topP.toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      id="pipeline-top-p"
                      min={0.1}
                      max={1}
                      step={0.05}
                      value={[settings.topP]}
                      onValueChange={(value) =>
                        onSettingsChange({
                          ...settings,
                          topP: sliderValue(value) || settings.topP,
                        })
                      }
                    />
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="pipeline-alpha" className="text-xs">
                        Hybrid α (dense weight)
                      </Label>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {settings.alpha.toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      id="pipeline-alpha"
                      min={0}
                      max={1}
                      step={0.05}
                      value={[settings.alpha]}
                      onValueChange={(value) =>
                        onSettingsChange({
                          ...settings,
                          alpha: sliderValue(value) || settings.alpha,
                        })
                      }
                    />
                  </div>
                </div>
              )}
            </section>

            <Accordion>
              <AccordionItem value="infrastructure">
                <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:text-foreground">
                  Infrastructure details
                </AccordionTrigger>
                <AccordionContent className="max-h-[min(45vh,18rem)] overflow-y-auto overscroll-contain">
                  <div className="flex flex-col gap-3 pr-1">
                    <div className="rounded-lg border border-border/60 bg-muted/10 p-3">
                      <div className="mb-1.5 flex items-center gap-2 text-sm font-medium">
                        <Database className="size-4 text-primary" />
                        Milvus Lite
                        <StatusDot ok={retrievalOk} />
                      </div>
                      <p className="font-mono text-[10px] leading-relaxed break-all text-muted-foreground">
                        {health?.milvus.db_path ?? "—"}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Collection{" "}
                        <code className="text-foreground">
                          {health?.milvus.collection ?? "temporal_index"}
                        </code>
                        {health?.milvus.db_exists ? "" : " · database file missing"}
                      </p>
                    </div>

                    <div className="rounded-lg border border-border/60 bg-muted/10 p-3">
                      <div className="mb-1.5 flex items-center gap-2 text-sm font-medium">
                        <Server className="size-4 text-primary" />
                        Ollama
                        <StatusDot ok={ollamaOk} />
                      </div>
                      <p className="text-xs leading-relaxed text-muted-foreground">
                        {ollamaOk
                          ? `Ollama is running on localhost port 11434 with ${health?.ollama?.available_models?.length ?? 0} downloaded model(s). Chat uses these when backend is "ollama".`
                          : health?.ollama?.error ?? "Not reachable — start Ollama or check port 11434"}
                      </p>
                    </div>

                    <div className="rounded-lg border border-border/60 bg-muted/10 p-3">
                      <div className="mb-1.5 flex items-center gap-2 text-sm font-medium">
                        <Cpu className="size-4 text-primary" />
                        Apple MLX
                        <StatusDot ok={mlxOk} />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {mlxOk
                          ? "mlx-lm runtime available — MLX models bypass Ollama"
                          : health?.mlx?.error ?? "mlx-lm not installed"}
                      </p>
                      {health?.mlx?.catalog?.map((entry) => (
                        <p
                          key={entry.id}
                          className="mt-1 font-mono text-[10px] leading-relaxed break-all text-muted-foreground"
                        >
                          {entry.id} → {entry.path}
                          {entry.loaded ? " (weights loaded)" : ""}
                        </p>
                      ))}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}
