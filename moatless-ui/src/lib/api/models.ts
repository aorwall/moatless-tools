import type {
  ModelConfig,
  ModelsResponse,
  BaseModelsResponse,
  AddModelFromBaseRequest,
  ModelTestResult,
  CreateModelRequest,
} from "@/lib/types/model";
import { apiRequest } from "./config";

export const modelsApi = {
  getModels: () => apiRequest<ModelsResponse>("/models"),
  getBaseModels: () => apiRequest<BaseModelsResponse>("/models/base"),
  getModel: (id: string) => apiRequest<ModelConfig>(`/models/${id}`),
  getBaseModel: (id: string) => apiRequest<ModelConfig>(`/models/base/${id}`),
  addFromBase: (request: AddModelFromBaseRequest) =>
    apiRequest<ModelConfig>("/models/base", {
      method: "POST",
      body: JSON.stringify(request),
    }),
  createModel: (request: CreateModelRequest) =>
    apiRequest<ModelConfig>("/models", {
      method: "POST",
      body: JSON.stringify(request),
    }),
  updateModel: (model: ModelConfig) =>
    apiRequest<ModelConfig>(`/models/${model.id}`, {
      method: "PUT",
      body: JSON.stringify(model),
    }),
  deleteModel: (id: string) =>
    apiRequest(`/models/${id}`, {
      method: "DELETE",
    }),
  testModel: (id: string) =>
    apiRequest<ModelTestResult>(`/models/${id}/test`, {
      method: "POST",
    }),
};
