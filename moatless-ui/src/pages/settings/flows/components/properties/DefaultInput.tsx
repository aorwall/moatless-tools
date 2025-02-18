import { Control } from "react-hook-form";
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormDescription,
} from "@/lib/components/ui/form";
import { Input } from "@/lib/components/ui/input";
import { ComponentProperty } from "@/lib/types/flow";

interface DefaultInputProps {
  name: string;
  control: Control<any>;
  property: ComponentProperty;
}

export function DefaultInput({ name, control, property }: DefaultInputProps) {
  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{property.title || name}</FormLabel>
          <FormControl>
            <Input
              {...field}
              type={property.type === "integer" ? "number" : "text"}
              value={field.value ?? ""}
              onChange={e => {
                const value = property.type === "integer" 
                  ? e.target.value ? parseInt(e.target.value) : undefined
                  : e.target.value;
                field.onChange(value);
              }}
            />
          </FormControl>
          {property.description && (
            <FormDescription>{property.description}</FormDescription>
          )}
        </FormItem>
      )}
    />
  );
} 