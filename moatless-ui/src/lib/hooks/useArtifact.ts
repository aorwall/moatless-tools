import { artifactsApi } from "@/lib/api/artifacts";
import type { ContentStructure } from "@/lib/types/content";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export interface ArtifactResponse {
  id: string;
  type: string;
  name: string | null;
  created_at: number;
  references: Array<{
    id: string;
    type: string;
  }>;
  data: Record<string, any>;
  status: string;
  can_persist: boolean;
  content: ContentStructure;
}

export interface ArtifactListItem {
  id: string;
  type: string;
  name: string | null;
  created_at: number;
}

export interface ArtifactFilters {
  type?: string;
  sortBy?: "name" | "created_at";
  sortOrder?: "asc" | "desc";
}

export const artifactKeys = {
  all: ["artifacts"] as const,
  details: () => [...artifactKeys.all, "detail"] as const,
  detail: (trajectoryId: string, artifactType: string, artifactId: string) =>
    [
      ...artifactKeys.details(),
      trajectoryId,
      artifactType,
      artifactId,
    ] as const,
  list: (trajectoryId: string, filters?: ArtifactFilters) =>
    [...artifactKeys.all, "list", trajectoryId, filters] as const,
};

export const useArtifact = (
  trajectoryId: string,
  artifactType: string,
  artifactId: string,
) => {
  return useQuery({
    queryKey: artifactKeys.detail(trajectoryId, artifactType, artifactId),
    queryFn: () =>
      artifactsApi.getArtifact(trajectoryId, artifactType, artifactId),
    enabled: !!trajectoryId && !!artifactType && !!artifactId,
  });
};

export const usePersistArtifact = (
  trajectoryId: string,
  artifactType: string,
  artifactId: string,
) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () =>
      artifactsApi.persistArtifact(trajectoryId, artifactType, artifactId),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: artifactKeys.detail(trajectoryId, artifactType, artifactId),
      });

      queryClient.invalidateQueries({
        queryKey: artifactKeys.list(trajectoryId),
      });
    },
  });
};

interface ArtifactsQueryResult {
  artifacts: ArtifactListItem[];
  types: string[];
}

export const useListArtifacts = (
  trajectoryId: string,
  filters?: ArtifactFilters,
) => {
  console.log("fetching artifacts for trajectoryId", trajectoryId);
  return useQuery({
    queryKey: artifactKeys.list(trajectoryId, filters),
    queryFn: () =>
      artifactsApi.listArtifacts(trajectoryId).then((artifacts) => {
        // Ensure artifacts is an array
        const artifactsArray = Array.isArray(artifacts) ? artifacts : [];

        // Get all unique types before filtering
        const types = Array.from(new Set(artifactsArray.map((a) => a.type)));

        let filteredArtifacts = [...artifactsArray];

        // Apply type filter
        if (filters?.type) {
          filteredArtifacts = filteredArtifacts.filter(
            (artifact) => artifact.type === filters.type,
          );
        }

        // Apply sorting
        if (filters?.sortBy) {
          filteredArtifacts.sort((a, b) => {
            const aValue =
              filters.sortBy === "name"
                ? a.name || `${a.type} #${a.id}`
                : a.created_at;
            const bValue =
              filters.sortBy === "name"
                ? b.name || `${b.type} #${b.id}`
                : b.created_at;

            if (aValue < bValue) return filters.sortOrder === "desc" ? 1 : -1;
            if (aValue > bValue) return filters.sortOrder === "desc" ? -1 : 1;
            return 0;
          });
        }

        return {
          artifacts: filteredArtifacts,
          types: types,
        };
      }),
    enabled: !!trajectoryId,
  });
};
