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

export type BaseModelsResponse = {
  models: ModelConfig[];
};

export type AddModelFromBaseRequest = {
  base_model_id: string;
  new_model_id: string;
  updates?: Partial<ModelConfig>;
};

export type CreateModelRequest = Omit<ModelConfig, 'id'> & { id: string };

export const ModelTestResultSchema = z.object({
  success: z.boolean(),
  message: z.string(),
  model_id: z.string(),
  model_response: z.string().optional(),
  error_type: z.string().optional(),
  error_details: z.string().optional(),
  response_time_ms: z.number().optional(),
  test_timestamp: z.string(),
});

export type ModelTestResult = z.infer<typeof ModelTestResultSchema>;
