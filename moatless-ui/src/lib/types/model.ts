import { z } from "zod";

export const ModelConfigSchema = z.object({
  id: z.string(),
  model: z.string(),
  model_base_url: z.string().nullable().optional(),
  model_api_key: z.string().nullable().optional(),
  temperature: z.number().optional(),
  max_tokens: z.number().optional(),
  timeout: z.number(),
  thoughts_in_action: z.boolean(),
  disable_thoughts: z.boolean(),
  merge_same_role_messages: z.boolean(),
  message_cache: z.boolean(),
  few_shot_examples: z.boolean(),
  response_format: z.enum(["tool_call", "react"]),
  message_history_type: z.enum(["messages", "react"]),
});

export type ModelConfig = z.infer<typeof ModelConfigSchema>;

export type ModelsResponse = {
  models: ModelConfig[];
};
