import { apiRequest } from "./config";
import { TrajectoryListItem } from "@/lib/types";
import { ResumeTrajectoryRequest, Trajectory } from "@/lib/types/trajectory";

export const trajectoriesApi = {
  getTrajectories: async (): Promise<TrajectoryListItem[]> => {
    return apiRequest("/trajectories");
  },

  getTrajectory: async (projectId: string, trajectoryId: string): Promise<Trajectory> => {
    return apiRequest(`/trajectories/${projectId}/${trajectoryId}`, {
      method: "GET",
    });
  },

  start: (projectId: string, trajectoryId: string) =>
    apiRequest(`/trajectories/${projectId}/${trajectoryId}/start`, {
      method: 'POST',
    }),

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

  retryNode: async (trajectoryId: string, projectId: string, nodeId: number, params?: {
    agent_id?: string;
    model_id?: string;
  }): Promise<{ run_id: string }> => {
    return apiRequest(`/trajectories/${projectId}/${trajectoryId}/retry`, {
      method: 'POST',
      body: JSON.stringify({ ...params, node_id: nodeId }),
    });
  },

  executeNode: async (
    trajectoryId: string, 
    projectId: string, 
    nodeId: number,
  ): Promise<any> => {
    if (!nodeId) {
      throw new Error("Node ID is required to execute a node");
    }
    return apiRequest(`/trajectories/${projectId}/${trajectoryId}/execute`, {
      method: 'POST',
      body: JSON.stringify({ node_id: nodeId }),
    });
  },
};
