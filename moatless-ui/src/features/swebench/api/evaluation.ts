import { apiRequest } from '@/lib/api/config';

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
}

export interface EvaluationInstance {
  instance_id: string;
  status: string;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  evaluated_at?: string;
  resolved?: boolean;
}

export interface Evaluation {
  evaluation_name: string;
  dataset_name: string;
  status: string;
  started_at: string;
  completed_at?: string;
  instances: EvaluationInstance[];
  num_workers: number;
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
  getDatasets: () => 
    apiRequest<{ datasets: Dataset[] }>('/swebench/datasets'),

  startEvaluation: ({ evaluationId, numConcurrentInstances }: StartEvaluationParams) =>
    apiRequest<Evaluation>(`/swebench/evaluations/${evaluationId}/start`, {
      method: 'POST',
      body: JSON.stringify({ num_concurrent_instances: numConcurrentInstances }),
    }),

  getEvaluation: (evaluationId: string) =>
    apiRequest<Evaluation>(`/swebench/evaluations/${evaluationId}`),

  listEvaluations: () =>
    apiRequest<EvaluationListResponse>('/swebench/evaluations'),

  createEvaluation: (data: EvaluationRequest) =>
    apiRequest<Evaluation>(`/swebench/evaluations`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
}; 