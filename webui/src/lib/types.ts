export type DownloadSourceName = "modelscope" | "huggingface" | "hf-mirror";
export type View = "dashboard" | "catalog" | "loaded" | "separate" | "settings";
export type Theme = "light" | "dark";
export type SeparationResponseFormat = "zip" | "json";
export type OutputAudioFormat = "wav" | "flac" | "pcm_f32le";

export interface ApiErrorBody {
  error: {
    message: string;
    type: string;
    param: string | null;
    code: string;
  };
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_loading: boolean;
  model: string | null;
  device: string | null;
}

export interface ServerInfo {
  object: "server.info";
  webui: {
    enabled: boolean;
    path: string | null;
  };
  auth: {
    api_key_required: boolean;
  };
  limits: {
    max_audio_seconds: number;
    max_request_bytes: number;
    max_queue_size: number;
    request_timeout_seconds: number;
  };
  download_source: {
    source: DownloadSourceName;
    endpoint: string | null;
  };
  model_dir: string;
}

export interface DownloadSource {
  object: "download.source";
  source: DownloadSourceName;
  endpoint: string | null;
  model_dir: string;
}

export interface CatalogModel {
  id: string;
  object: "pymss.model_catalog_entry";
  owned_by: "pymss";
  pymss: {
    name: string;
    aliases: string[];
    model_type: string | null;
    architecture: string;
    category: string;
    primary_category: string;
    secondary_category: string;
    target_stem: string;
    supported: boolean;
    unsupported_reason: string;
    size_bytes: number;
    local: {
      complete: boolean;
      missing_count: number;
      model_dir: string;
    };
    remote: {
      available: boolean;
      source: DownloadSourceName;
      endpoint: string | null;
    };
    files?: CatalogModelFile[];
  };
}

export interface CatalogModelFile {
  role: "model" | "config" | "auxiliary";
  relpath: string;
  exists: boolean;
  size_bytes: number;
  remote_url: string;
}

export interface CatalogListResponse {
  object: "list";
  data: CatalogModel[];
  pymss: {
    model_dir: string;
    source: DownloadSourceName;
    endpoint: string | null;
    total: number;
  };
}

export interface LoadedModel {
  id: string;
  object: "model";
  created: number;
  owned_by: "pymss";
  pymss: {
    catalog_name: string;
    model_type: string | null;
    architecture: string;
    category: string;
    catalog_target_stem: string;
    supported: boolean;
    sample_rate: number;
    instruments: string[];
    instruments_source: string;
    supported_parameters: Record<string, string[]>;
  };
}

export interface LoadedModelsResponse {
  object: "list";
  data: LoadedModel[];
}

export interface DownloadResult {
  object: "model.download";
  model: CatalogModel;
  source: DownloadSourceName;
  endpoint: string | null;
  downloaded: string[];
  skipped: string[];
}

export interface LoadResult {
  object: "model.load";
  previous_model_loaded: boolean;
  model_loaded: boolean;
  model: LoadedModel;
}

export interface SeparationJsonResult {
  id: string;
  object: "audio.separation";
  created: number;
  model: string;
  outputs: Array<{
    stem: string;
    audio: {
      format: Extract<OutputAudioFormat, "pcm_f32le">;
      sample_rate: number;
      channels: number;
      data: string;
    };
  }>;
  metadata: {
    input_seconds: number;
    output_stems: string[];
    device: string;
  };
  usage: {
    type: "duration";
    seconds: number;
  };
}

export interface CatalogQuery {
  supported: "true" | "false" | "all";
  local: "all" | "complete" | "missing";
  category: string;
  q: string;
}
