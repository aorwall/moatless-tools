import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { modelsApi } from "@/lib/api/models";
import type { ModelConfig } from '@/lib/types/model';

export const modelKeys = {
  all: ["models"] as const,
  lists: () => [...modelKeys.all, "list"] as const,
  list: (filters: Record<string, any>) =>
    [...modelKeys.lists(), { filters }] as const,
  details: () => [...modelKeys.all, "detail"] as const,
  detail: (id: string) => [...modelKeys.details(), id] as const,
};

// Hooks
export function useModels() {
  return useQuery({
    queryKey: modelKeys.lists(),
    queryFn: () => modelsApi.getModels().then((res) => res.models),
  });
}

export function useModel(id: string) {
  return useQuery({
    queryKey: ['model', id],
    queryFn: () => modelsApi.getModel(id),
    enabled: !!id,
  });
}

export function useUpdateModel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: modelsApi.updateModel,
    onSuccess: (updatedModel) => {
      queryClient.setQueryData(['model', updatedModel.id], updatedModel);
      queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}
