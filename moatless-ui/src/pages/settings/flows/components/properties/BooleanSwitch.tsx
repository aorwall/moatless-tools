import { Control } from "react-hook-form";
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormDescription,
} from "@/lib/components/ui/form";
import { Switch } from "@/lib/components/ui/switch";
import { ComponentProperty } from "@/lib/types/flow";

interface BooleanSwitchProps {
  name: string;
  control: Control<any>;
  property: ComponentProperty;
}

export function BooleanSwitch({ name, control, property }: BooleanSwitchProps) {
  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3">
          <div className="space-y-0.5">
            <FormLabel>{property.title || name}</FormLabel>
            {property.description && (
              <FormDescription>{property.description}</FormDescription>
            )}
          </div>
          <FormControl>
            <Switch
              checked={field.value ?? false}
              onCheckedChange={field.onChange}
            />
          </FormControl>
        </FormItem>
      )}
    />
  );
} 