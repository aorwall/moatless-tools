import { z } from "zod";

export interface ActionProperty {
  type: string;
  title: string;
  description: string;
  default?: any;
}

export interface ActionSchema {
  title: string;
  description: string;
  type: string;
  properties: Record<string, ActionProperty>;
}

export interface ActionSchemaWithClass extends ActionSchema {
  action_class: string;
}

export const ActionConfigSchema = z.object({
  action_class: z.string(),
}).passthrough();

export type ActionConfig = z.infer<typeof ActionConfigSchema>;

export const AgentConfigSchema = z.object({
  agent_id: z.string(),
  model_id: z.string().optional(),
  response_format: z.enum(["TOOL_CALL", "REACT"]).optional(),
  actions: z.array(ActionConfigSchema).default([]),
  system_prompt: z.string().optional(),
  memory: z.record(z.any()).optional(),
});

export type AgentConfig = z.infer<typeof AgentConfigSchema>;

export interface AgentData {
  configs: Record<string, AgentConfig>;
}

export interface ActionInfo {
  name: string;
  description: string;
  category?: string;
}

export interface ActionsResponse {
  actions: Record<string, ActionSchema>;
}
