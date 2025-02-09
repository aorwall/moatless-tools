import { z } from 'zod';

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

export const ActionConfigSchema = z.object({
  title: z.string(),
  properties: z.record(z.string(), z.any()).default({}),
});

export type ActionConfig = z.infer<typeof ActionConfigSchema>;

export const AgentConfigSchema = z.object({
  id: z.string(),
  model_id: z.string(),
  response_format: z.enum(['TOOL_CALL', 'REACT']),
  actions: z.array(ActionConfigSchema).default([]),
  system_prompt: z.string().optional(),
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