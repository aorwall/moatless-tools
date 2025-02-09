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
    public data?: any
  ) {
    super(message);
    this.name = 'ApiError';
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
    
    try {
      errorData = await response.json();
      errorMessage = errorData.detail || response.statusText;
    } catch {
      errorMessage = response.statusText || 'UnknownError';
      errorData = null;
    }

    throw new ApiError(
      response.status,
      errorMessage,
      errorData
    );
  }

  return response.json();
}
