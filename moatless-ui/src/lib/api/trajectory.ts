import { apiRequest, API_CONFIG } from "@/lib/api/config";
import { Trajectory } from "@/lib/types/trajectory";

export const trajectoryApi = {
  getTrajectory: async (path: string): Promise<Trajectory> => {
    return apiRequest("/trajectory", {
      params: { file_path: path },
    });
  },

  uploadTrajectory: async (file: File): Promise<{ path: string }> => {
    const formData = new FormData();
    formData.append("file", file);

    // For file uploads, we need to use fetch directly since apiRequest assumes JSON
    const response = await fetch(`${API_CONFIG.baseUrl}/trajectory/upload`, {
      method: "POST",
      body: formData,
      credentials: "include",
    });

    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ detail: "Failed to upload trajectory" }));
      throw new Error(error.detail || "Failed to upload trajectory");
    }

    return response.json();
  },
};
