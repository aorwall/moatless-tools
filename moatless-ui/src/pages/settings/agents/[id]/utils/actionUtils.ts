import type { ActionSchema, ActionConfig } from "@/lib/types/agent";

export function createActionConfigFromSchema(
  actionSchema: ActionSchema,
): ActionConfig {
  const defaultProperties: Record<string, any> = {};

  Object.entries(actionSchema.properties).forEach(([key, prop]) => {
    defaultProperties[key] = prop.default;
  });

  defaultProperties["action_class"] = actionSchema.action_class;

  return {
    title: actionSchema.title,
    properties: defaultProperties,
  };
}
