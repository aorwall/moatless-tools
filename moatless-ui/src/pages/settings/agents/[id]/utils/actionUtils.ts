import type { ActionSchemaWithClass, ActionConfig } from "@/lib/types/agent";

export function createActionConfigFromSchema(
  actionSchema: ActionSchemaWithClass,
): ActionConfig {
  const defaultProperties: Record<string, any> = {};

  if (actionSchema.properties) {
    Object.entries(actionSchema.properties).forEach(([key, prop]) => {
      defaultProperties[key] = prop.default;
    });
  }

  return {
    title: actionSchema.title || '',
    action_class: actionSchema.action_class,
    properties: defaultProperties,
  };
}
