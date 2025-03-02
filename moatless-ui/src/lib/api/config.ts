export const API_CONFIG = {
  baseUrl: "http://localhost:8000/api",
  defaultHeaders: {
    "Content-Type": "application/json",
  },
} as const;

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public data?: any,
  ) {
    super(message);
    this.name = "ApiError";
  }

  toString() {
    if (this.data && typeof this.data === "object") {
      const details = Object.entries(this.data)
        .map(([key, value]) => `${key}: ${value}`)
        .join(", ");
      return details || this.message;
    }
    return this.message;
  }
}

type FetchOptions = RequestInit & {
  params?: Record<string, string>;
};

export async function apiRequest<T>(
  endpoint: string,
  options: FetchOptions = {},
): Promise<T> {
  const { params, ...fetchOptions } = options;

  let url = `${API_CONFIG.baseUrl}${endpoint}`;
  if (params) {
    const searchParams = new URLSearchParams(params);
    url += `?${searchParams.toString()}`;
  }

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      headers: {
        ...API_CONFIG.defaultHeaders,
        ...fetchOptions.headers,
      },
      credentials: "include",
    });

    if (!response.ok) {
      let errorMessage: string;
      let errorData: any;

      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        try {
          errorData = await response.json();
          // Handle FastAPI validation errors
          if (errorData.detail && Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail
              .map((err: any) => `${err.loc.join(".")}: ${err.msg}`)
              .join(", ");
          } else {
            errorMessage =
              errorData.detail || errorData.message || response.statusText;
          }
        } catch {
          errorMessage = "Failed to parse error response";
          errorData = null;
        }
      } else {
        errorMessage =
          (await response.text()) || response.statusText || "Unknown error";
        errorData = null;
      }

      throw new ApiError(response.status, errorMessage, errorData);
    }

    return response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    // Handle network errors or other exceptions
    throw new ApiError(
      0,
      error instanceof Error ? error.message : "Network error",
      error,
    );
  }
}
