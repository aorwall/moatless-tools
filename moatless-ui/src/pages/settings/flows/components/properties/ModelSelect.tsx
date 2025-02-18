import { useModels } from "@/lib/hooks/useModels";
import { Control } from "react-hook-form";
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormDescription,
} from "@/lib/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { Loader2 } from "lucide-react";
import { ComponentProperty } from "@/lib/types/flow";

interface ModelSelectProps {
  name: string;
  control: Control<any>;
  property: ComponentProperty;
}

export function ModelSelect({ name, control, property }: ModelSelectProps) {
  const { data: models, isLoading } = useModels();

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{property.title || name}</FormLabel>
          <Select onValueChange={field.onChange} value={field.value ?? ""}>
            <FormControl>
              <SelectTrigger>
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <SelectValue placeholder="Select model" />
                )}
              </SelectTrigger>
            </FormControl>
            <SelectContent>
              <SelectItem value="">None</SelectItem>
              {models?.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  {model.id}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {property.description && (
            <FormDescription>{property.description}</FormDescription>
          )}
        </FormItem>
      )}
    />
  );
} 