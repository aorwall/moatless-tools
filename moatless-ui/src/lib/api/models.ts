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
  getModel: (model_id: string) => apiRequest<ModelConfig>(`/models/${model_id}`),
  getBaseModel: (model_id: string) =>
    apiRequest<ModelConfig>(`/models/base/${model_id}`),
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
    apiRequest<ModelConfig>(`/models/${model.model_id}`, {
      method: "PUT",
      body: JSON.stringify(model),
    }),
  deleteModel: (model_id: string) =>
    apiRequest(`/models/${model_id}`, {
      method: "DELETE",
    }),
  testModel: (model_id: string) =>
    apiRequest<ModelTestResult>(`/models/${model_id}/test`, {
      method: "POST",
    }),
};
