import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { modelsApi } from "@/lib/api/models";
import type {
  ModelConfig,
  AddModelFromBaseRequest,
  ModelTestResult,
  CreateModelRequest,
} from "@/lib/types/model";

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
    queryKey: ["models"],
    queryFn: modelsApi.getModels,
  });
}

export function useBaseModels() {
  return useQuery({
    queryKey: ["baseModels"],
    queryFn: modelsApi.getBaseModels,
  });
}

export function useModel(id: string) {
  return useQuery({
    queryKey: ["models", id],
    queryFn: () => modelsApi.getModel(id),
    enabled: !!id,
  });
}

export function useBaseModel(id: string) {
  return useQuery({
    queryKey: ["baseModels", id],
    queryFn: () => modelsApi.getBaseModel(id),
    enabled: !!id,
  });
}

export function useAddFromBase() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: AddModelFromBaseRequest) =>
      modelsApi.addFromBase(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useCreateModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: CreateModelRequest) => modelsApi.createModel(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useUpdateModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (model: ModelConfig) => modelsApi.updateModel(model),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["models", data.id] });
    },
  });
}

export function useDeleteModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => modelsApi.deleteModel(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useTestModel() {
  return useMutation({
    mutationFn: (id: string) => modelsApi.testModel(id),
  });
}
