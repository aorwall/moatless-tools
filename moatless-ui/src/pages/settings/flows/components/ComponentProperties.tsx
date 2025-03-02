import { ComponentSchema, ComponentProperty } from "@/lib/types/flow";
import { Control } from "react-hook-form";
import { ModelSelect } from "./properties/ModelSelect";
import { BooleanSwitch } from "./properties/BooleanSwitch";
import { EnumSelect } from "./properties/EnumSelect";
import { DefaultInput } from "./properties/DefaultInput";

interface ComponentPropertiesProps {
  schema: ComponentSchema;
  control: Control<any>;
  basePath: string;
}

export function ComponentProperties({
  schema,
  control,
  basePath,
}: ComponentPropertiesProps) {
  return (
    <div className="space-y-4 mt-4 ml-6 border-l-2 pl-4 border-muted">
      {Object.entries(schema.properties).map(([key, propRaw]) => {
        // Ensure the property matches our expected type
        const prop = propRaw as ComponentProperty;
        if (!prop || typeof prop.type !== "string") return null;

        const fieldName = `${basePath}.${key}`;

        // Special handling for model_id
        if (key === "model_id") {
          return (
            <ModelSelect
              key={key}
              name={fieldName}
              control={control}
              property={prop}
            />
          );
        }

        // Handle different property types
        if (prop.type === "boolean") {
          return (
            <BooleanSwitch
              key={key}
              name={fieldName}
              control={control}
              property={prop}
            />
          );
        }

        if (prop.enum?.length) {
          return (
            <EnumSelect
              key={key}
              name={fieldName}
              control={control}
              property={prop}
            />
          );
        }

        // Default input for other types
        return (
          <DefaultInput
            key={key}
            name={fieldName}
            control={control}
            property={prop}
          />
        );
      })}
    </div>
  );
}
