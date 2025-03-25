import { apiRequest } from "@/lib/api/config";
import { FlowConfig } from "@/lib/types/flow";
import { ModelConfig } from "@/lib/types/model";

export interface Dataset {
  name: string;
  description: string;
  instance_count: number;
}

export interface EvaluationRequest {
  flow_id: string;
  model_id: string;
  name: string;
  dataset: string;
  num_concurrent_instances: number;
  instance_ids?: string[];
}

export interface EvaluationInstance {
  instance_id: string;
  status: string;
  job_status: string;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  evaluated_at?: string;
  iterations?: number;
  error_at?: string;
  resolved?: boolean;
  resolved_by?: number;
  reward?: number;
  usage?: {
    completion_cost?: number;
    prompt_tokens?: number;
    completion_tokens?: number;
    cache_read_tokens?: number;
  };
}

export interface Evaluation {
  evaluation_name: string;
  dataset_name: string;
  status: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  flow: FlowConfig;
  model: ModelConfig;
  instances: EvaluationInstance[];
  num_workers: number;
  description?: string;
}

export interface EvaluationStatusSummary {
  pending: number;
  running: number;
  evaluating: number;
  completed: number;
  error: number;
  resolved: number;
  failed: number;
}

export interface EvaluationListItem {
  evaluation_name: string;
  dataset_name: string;
  status: string;
  started_at: string;
  completed_at?: string;
  instance_count: number;
  flow_id: string;
  model_id: string;
  status_summary?: EvaluationStatusSummary;
  total_cost: number;
  prompt_tokens: number;
  completion_tokens: number;
  cached_tokens: number;
  resolved_count: number;
  failed_count: number;
}

export interface EvaluationListResponse {
  evaluations: EvaluationListItem[];
}

export interface StartEvaluationParams {
  evaluationId: string;
  numConcurrentInstances: number;
}

export const evaluationApi = {
  getDatasets: () => apiRequest<{ datasets: Dataset[] }>("/swebench/datasets"),

  startEvaluation: ({
    evaluationId,
    numConcurrentInstances,
  }: StartEvaluationParams) =>
    apiRequest<Evaluation>(`/swebench/evaluations/${evaluationId}/start`, {
      method: "POST",
      body: JSON.stringify({
        num_concurrent_instances: numConcurrentInstances,
      }),
    }),

  cloneEvaluation: (evaluationId: string) =>
    apiRequest<Evaluation>(`/swebench/evaluations/${evaluationId}/clone`),

  processEvaluationResults: (evaluationId: string) =>
    apiRequest<Evaluation>(`/swebench/evaluations/${evaluationId}/process`, {
      method: "POST",
    }),

  getEvaluation: (evaluationId: string) =>
    apiRequest<Evaluation>(`/swebench/evaluations/${evaluationId}`),

  listEvaluations: () =>
    apiRequest<EvaluationListItem[]>("/swebench/evaluations"),

  createEvaluation: (data: EvaluationRequest) =>
    apiRequest<Evaluation>(`/swebench/evaluations`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
};
