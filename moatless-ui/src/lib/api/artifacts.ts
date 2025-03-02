import { apiRequest } from "./config";
import type { ArtifactResponse } from "@/lib/hooks/useArtifact";
import type { ArtifactListItem } from "@/lib/hooks/useArtifact";

export const artifactsApi = {
  getArtifact: async (
    trajectoryId: string,
    artifactType: string,
    artifactId: string,
  ) => {
    return apiRequest<ArtifactResponse>(
      `/artifacts/${trajectoryId}/${artifactType}/${artifactId}`,
    );
  },

  persistArtifact: async (
    trajectoryId: string,
    artifactType: string,
    artifactId: string,
  ) => {
    return apiRequest<ArtifactResponse>(
      `/artifacts/${trajectoryId}/${artifactType}/${artifactId}/persist`,
      {
        method: "POST",
      },
    );
  },

  listArtifacts: async (trajectoryId: string) => {
    return apiRequest<ArtifactListItem[]>(`/artifacts/${trajectoryId}`);
  },
};
