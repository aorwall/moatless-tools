import { apiRequest } from '@/lib/api/config';

export interface Dataset {
  name: string;
  description: string;
  instance_count: number;
}

export interface EvaluationRequest {
  agent_id: string;
  model_id: string;
  dataset: string;
  num_workers: number;
  max_iterations: number;
  max_expansions: number;
}

export interface EvaluationInstance {
  instance_id: string;
  status: string;
  error?: string;
  start_time?: string;
  finish_time?: string;
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
  started: number;
  completed: number;
  error: number;
}

export interface EvaluationListItem {
  evaluation_name: string;
  dataset_name: string;
  status: string;
  started_at: string;
  completed_at?: string;
  instance_count: number;
  status_summary?: EvaluationStatusSummary;
}

export interface EvaluationListResponse {
  evaluations: EvaluationListItem[];
}

export const evaluationApi = {
  getDatasets: () => 
    apiRequest<{ datasets: Dataset[] }>('/swebench/datasets'),

  startEvaluation: (evaluationId: string) =>
    apiRequest<Evaluation>(`/swebench/evaluations/${evaluationId}/start`),

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