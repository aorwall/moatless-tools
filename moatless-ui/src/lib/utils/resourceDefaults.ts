import type { FlowConfig } from "@/lib/types/flow";

export const createDefaultFlow = (): Omit<FlowConfig, "id"> => ({
  description: "",
  flow_type: "tree",
  max_expansions: 3,
  max_iterations: 100,
  max_cost: 4.0,
  max_depth: 20,
  min_finished_nodes: 2,
  max_finished_nodes: 3,
  reward_threshold: 90,
  agent_id: undefined,
  selector: undefined,
  expander: undefined,
  value_function: undefined,
  feedback_generator: undefined,
  discriminator: undefined,
});

export const generateResourceId = (prefix: string) =>
  `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
