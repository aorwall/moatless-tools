import type { ModelConfig, ModelsResponse } from "@/lib/types/model";
import { apiRequest } from "./config";

export const modelsApi = {
  getModels: () => apiRequest<ModelsResponse>("/models"),
  getModel: (id: string) => apiRequest<ModelConfig>(`/models/${id}`),
  updateModel: (model: ModelConfig) =>
    apiRequest<ModelConfig>(`/models/${model.id}`, {
      method: "PUT",
      body: JSON.stringify(model),
    }),
};
