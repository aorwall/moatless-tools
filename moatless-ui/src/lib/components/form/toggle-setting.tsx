import { Switch } from "@/lib/components/ui/switch"
import { Label } from "@/lib/components/ui/label"

interface ToggleSettingProps {
  label: string
  description?: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
}

export function ToggleSetting({ label, description, checked, onCheckedChange }: ToggleSettingProps) {
  return (
    <div className="flex items-center justify-between">
      <div className="space-y-0.5">
        <Label>{label}</Label>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>
      <Switch checked={checked} onCheckedChange={onCheckedChange} />
    </div>
  )
}

