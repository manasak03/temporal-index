import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import type { GenerationSettings } from "@/types/chat"

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  settings: GenerationSettings
  onSettingsChange: (settings: GenerationSettings) => void
}

function formatSliderValue(value: number, decimals = 2) {
  return value.toFixed(decimals)
}

function sliderValue(values: number | readonly number[]) {
  return Array.isArray(values) ? (values[0] ?? 0) : values
}

export function SettingsDialog({
  open,
  onOpenChange,
  settings,
  onSettingsChange,
}: SettingsDialogProps) {
  const update = (partial: Partial<GenerationSettings>) => {
    onSettingsChange({ ...settings, ...partial })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[min(92vh,36rem)] overflow-y-auto border-border/60 bg-popover sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Generation Settings</DialogTitle>
          <DialogDescription>
            Tune retrieval fusion and LLM sampling parameters. Changes apply to the next message.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-6 py-2">
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="temperature">Temperature</Label>
              <span className="font-mono text-xs text-muted-foreground">
                {formatSliderValue(settings.temperature)}
              </span>
            </div>
            <Slider
              id="temperature"
              min={0}
              max={1.5}
              step={0.05}
              value={[settings.temperature]}
              onValueChange={(value) =>
                update({ temperature: sliderValue(value) || settings.temperature })
              }
            />
          </div>

          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="top-p">Top P</Label>
              <span className="font-mono text-xs text-muted-foreground">
                {formatSliderValue(settings.topP)}
              </span>
            </div>
            <Slider
              id="top-p"
              min={0.1}
              max={1}
              step={0.05}
              value={[settings.topP]}
              onValueChange={(value) => update({ topP: sliderValue(value) || settings.topP })}
            />
          </div>

          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="alpha">Hybrid Alpha (dense weight)</Label>
              <span className="font-mono text-xs text-muted-foreground">
                {formatSliderValue(settings.alpha)}
              </span>
            </div>
            <Slider
              id="alpha"
              min={0}
              max={1}
              step={0.05}
              value={[settings.alpha]}
              onValueChange={(value) => update({ alpha: sliderValue(value) || settings.alpha })}
            />
            <p className="text-xs text-muted-foreground">
              Higher values favor dense vector similarity; lower values favor sparse BM25 retrieval.
            </p>
          </div>

          <div className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/30 px-3 py-3">
            <div className="flex flex-col gap-0.5">
              <Label htmlFor="grounding-strict">Strict grounding</Label>
              <span className="text-xs text-muted-foreground">
                Enforce context-only answers (system prompt default)
              </span>
            </div>
            <Switch id="grounding-strict" defaultChecked disabled />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
