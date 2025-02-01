import { z } from 'zod';

export const AgentConfigSchema = z.object({
  id: z.string(),
  model_id: z.string(),
  response_format: z.enum(['TOOL_CALL', 'REACT']),
  // Add other agent config fields as needed
});

export type AgentConfig = z.infer<typeof AgentConfigSchema>;

export interface AgentData {
  agents: Record<string, AgentConfig>;
  supportedModels: string[];
  modelConfigs: Record<string, any>;
} 