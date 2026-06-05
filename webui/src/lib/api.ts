import type {
  ApiErrorBody,
  CatalogListResponse,
  CatalogModel,
  CatalogQuery,
  DownloadResult,
  DownloadSource,
  DownloadSourceName,
  HealthResponse,
  LoadedModelsResponse,
  LoadResult,
  ServerInfo,
} from "./types";

export class ApiError extends Error {
  status: number;
  code: string;
  type: string;
  param: string | null;

  constructor(status: number, body: ApiErrorBody) {
    super(body.error.message);
    this.name = "ApiError";
    this.status = status;
    this.code = body.error.code;
    this.type = body.error.type;
    this.param = body.error.param;
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NetworkError";
  }
}

type RequestOptions = RequestInit & {
  token?: string;
};

async function parseApiError(response: Response): Promise<ApiError> {
  let body: ApiErrorBody;
  try {
    body = (await response.json()) as ApiErrorBody;
  } catch {
    body = {
      error: {
        message: `HTTP ${response.status}`,
        type: "server_error",
        param: null,
        code: "http_error",
      },
    };
  }
  return new ApiError(response.status, body);
}

export async function apiFetch(path: string, options: RequestOptions = {}): Promise<Response> {
  const headers = new Headers(options.headers);
  if (options.token) {
    headers.set("Authorization", `Bearer ${options.token}`);
  }
  try {
    const response = await fetch(path, { ...options, headers });
    if (!response.ok) {
      throw await parseApiError(response);
    }
    return response;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new NetworkError(error instanceof Error ? error.message : "Network request failed");
  }
}

async function apiJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await apiFetch(path, options);
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return apiJson<HealthResponse>("/health");
}

export function getServerInfo(token: string): Promise<ServerInfo> {
  return apiJson<ServerInfo>("/v1/server/info", { token });
}

export function getDownloadSource(token: string): Promise<DownloadSource> {
  return apiJson<DownloadSource>("/v1/download-source", { token });
}

export function updateDownloadSource(token: string, source: DownloadSourceName, endpoint: string | null): Promise<DownloadSource> {
  return apiJson<DownloadSource>("/v1/download-source", {
    method: "POST",
    token,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, endpoint }),
  });
}

export function listCatalogModels(token: string, query: CatalogQuery): Promise<CatalogListResponse> {
  const params = new URLSearchParams();
  params.set("supported", query.supported);
  params.set("local", query.local);
  if (query.category.trim()) {
    params.set("category", query.category.trim());
  }
  if (query.q.trim()) {
    params.set("q", query.q.trim());
  }
  return apiJson<CatalogListResponse>(`/v1/catalog/models?${params.toString()}`, { token });
}

export function getCatalogModel(token: string, model: string): Promise<CatalogModel> {
  return apiJson<CatalogModel>(`/v1/catalog/models/${encodeURIComponent(model)}`, { token });
}

export function listLoadedModels(token: string): Promise<LoadedModelsResponse> {
  return apiJson<LoadedModelsResponse>("/v1/models", { token });
}

export function downloadModel(token: string, model: string, source?: DownloadSourceName, endpoint?: string | null): Promise<DownloadResult> {
  const payload: Record<string, unknown> = { model };
  if (source) {
    payload.source = source;
  }
  if (endpoint !== undefined) {
    payload.endpoint = endpoint;
  }
  return apiJson<DownloadResult>("/v1/models/download", {
    method: "POST",
    token,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function loadModel(
  token: string,
  model: string,
  inferenceParams: Record<string, unknown> | null,
  source?: DownloadSourceName,
  endpoint?: string | null,
): Promise<LoadResult> {
  const payload: Record<string, unknown> = { model };
  if (inferenceParams && Object.keys(inferenceParams).length) {
    payload.inference_params = inferenceParams;
  }
  if (source) {
    payload.source = source;
  }
  if (endpoint !== undefined) {
    payload.endpoint = endpoint;
  }
  return apiJson<LoadResult>("/v1/models/load", {
    method: "POST",
    token,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
