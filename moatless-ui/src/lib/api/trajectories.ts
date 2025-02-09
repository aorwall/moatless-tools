import { apiRequest } from "./config";
import { TrajectoryListItem } from "@/lib/types";
import { ResumeTrajectoryRequest, Trajectory } from "@/lib/types/trajectory";

export const trajectoriesApi = {
  getTrajectories: async (): Promise<TrajectoryListItem[]> => {
    return apiRequest("/trajectories");
  },

  getTrajectory: async (trajectoryId: string): Promise<Trajectory> => {
    return apiRequest(`/trajectories/${trajectoryId}`, {
      method: "GET",
    });
  },

  resume: (trajectoryId: string, data: ResumeTrajectoryRequest) =>
    apiRequest(`/trajectories/${trajectoryId}/resume`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  expandNode: async (trajectoryId: string, nodeId: number, params: {
    agent_id: string;
    model_id: string;
    message: string;
    attachments?: { name: string; data: string; }[];
  }): Promise<{ run_id: string }> => {
    return apiRequest(`/trajectories/${trajectoryId}/expand`, {
      method: 'POST',
      body: JSON.stringify({ ...params, node_id: nodeId }),
    });
  },

  retryNode: async (trajectoryId: string, nodeId: number, params?: {
    agent_id?: string;
    model_id?: string;
  }): Promise<{ run_id: string }> => {
    return apiRequest(`/trajectories/${trajectoryId}/retry`, {
      method: 'POST',
      body: JSON.stringify({ ...params, node_id: nodeId }),
    });
  },
};
