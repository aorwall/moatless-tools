import { AttachmentData } from "@/types/attachments";
import { Trajectory } from "../types/trajectory";
import { apiRequest } from "./config";

export interface SWEBenchInstance {
  instance_id: string;
  problem_statement: string;
  resolved_count: number;
}

export interface SWEBenchInstancesResponse {
  instances: SWEBenchInstance[];
}

export interface ValidationRequest {
  instance_id: string;
  model_id: string;
  agent_id: string;
}

export interface ValidationResponse {
  run_id: string;
}

export interface RunStatus {
  status: string;
  iterations: number;
  cost: number;
  current_action?: string;
  result?: any;
  error?: string;
}

export interface LoopResponse {
  project_id: string;
  trajectory_id: string;
}

export interface LoopRequest {
  agent_id: string;
  model_id: string;
  message: string;
  attachments?: AttachmentData[];
  repository_path?: string;
}

export const swebenchApi = {
  getInstances: (
    page: number,
    limit: number,
    sortBy: string = "instance_id",
    order: string = "asc",
  ) =>
    apiRequest<SWEBenchInstancesResponse>(
      `/swebench/instances?page=${page}&limit=${limit}&sort_by=${sortBy}&order=${order}`,
    ),

  startValidation: (data: ValidationRequest) =>
    apiRequest<ValidationResponse>("/swebench/validate", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  startLoop: (data: LoopRequest) =>
    apiRequest<LoopResponse>("/loop", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  startInstance: (evaluationName: string, instanceId: string) =>
    apiRequest<any>(
      `/swebench/evaluations/${evaluationName}/instances/${instanceId}/start`,
      {
        method: "POST",
      },
    ),

  retryInstance: (evaluationName: string, instanceId: string) =>
    apiRequest<any>(
      `/swebench/evaluations/${evaluationName}/instances/${instanceId}/retry`,
      {
        method: "POST",
      },
    ),

  getEvaluationInstance: (evaluationName: string, instanceId: string) =>
    apiRequest<Trajectory>(
      `/swebench/evaluations/${evaluationName}/instances/${instanceId}`,
    ),
};
