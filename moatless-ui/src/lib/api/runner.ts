import { apiRequest } from "./config";

export interface RunnerInfo {
  runner_type: string;
  status: string;
  data: {
    active_workers: number;
    total_workers: number;
  };
}

export interface RunnerStatusResponse {
  info: RunnerInfo;
  jobs: any[];
}

export const runnerApi = {
  getStatus: () => apiRequest<RunnerStatusResponse>("/runner"),
};
