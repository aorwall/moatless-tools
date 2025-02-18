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
import { ComponentProperty } from "@/lib/types/flow";

interface EnumSelectProps {
  name: string;
  control: Control<any>;
  property: ComponentProperty;
}

export function EnumSelect({ name, control, property }: EnumSelectProps) {
  if (!property.enum?.length) return null;

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
                <SelectValue placeholder={`Select ${property.title || name}`} />
              </SelectTrigger>
            </FormControl>
            <SelectContent>
              {property.enum.map((value) => (
                <SelectItem key={value} value={value}>
                  {value}
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