import { z } from "zod";

// Component schemas
export const ComponentPropertySchema = z.object({
  type: z.string(),
  title: z.string(),
  description: z.string().optional(),
  default: z.any().optional(),
  enum: z.array(z.string()).optional(),
  anyOf: z
    .array(
      z.object({
        type: z.string().optional(),
        $ref: z.string().optional(),
      }),
    )
    .optional(),
  $ref: z.string().optional(),
});

export const ComponentSchemaType = z.object({
  title: z.string().optional(),
  description: z.string().optional(),
  type: z.string(),
  properties: z.record(ComponentPropertySchema),
  $defs: z.record(z.any()).optional(),
  $id: z.string().optional(),
});

// Define a schema for artifact handler components
export const ArtifactHandlerSchema = z
  .object({
    artifact_handler_class: z.string(),
  })
  .catchall(z.any());

// Flow configuration
export const FlowConfigSchema = z.object({
  id: z.string(),
  description: z.string().optional(),
  flow_type: z.enum(["tree", "loop"]),

  // Common fields
  max_iterations: z.number().default(100),
  max_cost: z.number().default(4.0),
  agent_id: z.string().optional(),
  artifact_handlers: z.array(ArtifactHandlerSchema).optional(),

  // Tree-specific fields
  max_expansions: z.number().optional(),
  max_depth: z.number().optional(),
  min_finished_nodes: z.number().optional(),
  max_finished_nodes: z.number().optional(),
  reward_threshold: z.number().optional(),

  // Component references
  selector: z.string().optional(),
  expander: z.string().optional(),
  value_function: z.string().optional(),
  feedback_generator: z.string().optional(),
  discriminator: z.any().optional(),

  // Component configurations
  selector_config: z.record(z.any()).optional(),
  expander_config: z.record(z.any()).optional(),
  value_function_config: z.record(z.any()).optional(),
  feedback_generator_config: z.record(z.any()).optional(),
});

export const FlowConfigListSchema = z.array(FlowConfigSchema);

export type FlowConfig = z.infer<typeof FlowConfigSchema>;
export type ComponentProperty = z.infer<typeof ComponentPropertySchema>;
export type ComponentSchema = z.infer<typeof ComponentSchemaType>;
