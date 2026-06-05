import {
  apiFetch,
  downloadModel,
  getCatalogModel,
  getDownloadSource,
  getHealth,
  getServerInfo,
  listCatalogModels,
  listLoadedModels,
  loadModel,
  updateDownloadSource,
} from "./api";
import { base64ToBlob, prepareAudioFile } from "./audio";
import { parseParams } from "./params";
import type {
  CatalogListResponse,
  CatalogModel,
  CatalogQuery,
  DownloadSource,
  DownloadSourceName,
  HealthResponse,
  LoadedModel,
  OutputAudioFormat,
  SeparationJsonResult,
  SeparationResponseFormat,
  ServerInfo,
} from "./types";

export interface AuthenticatedData {
  serverInfo: ServerInfo;
  downloadSource: DownloadSource;
  catalog: CatalogListResponse;
  loadedModels: LoadedModel[];
}

export interface SourceUpdate {
  downloadSource: DownloadSource;
  serverInfo: ServerInfo;
}

export interface CatalogDownloadUpdate {
  catalog: CatalogListResponse;
  serverInfo: ServerInfo;
  noticeMessage: string;
}

export interface ModelLoadUpdate {
  model: LoadedModel;
  health: HealthResponse;
  catalog: CatalogListResponse;
  noticeMessage: string;
}

export interface SeparationUpdate {
  resultUrl: string | null;
  resultName: string;
  jsonResult: SeparationJsonResult | null;
  noticeMessage: string;
}

export async function fetchAuthenticatedData(token: string, catalogQuery: CatalogQuery): Promise<AuthenticatedData> {
  const serverInfo = await getServerInfo(token);
  const downloadSource = await getDownloadSource(token);
  const loadedModels = (await listLoadedModels(token)).data;
  const catalog = await listCatalogModels(token, catalogQuery);
  return { serverInfo, downloadSource, loadedModels, catalog };
}

export function fetchHealth(): Promise<HealthResponse> {
  return getHealth();
}

export function fetchCatalog(token: string, catalogQuery: CatalogQuery): Promise<CatalogListResponse> {
  return listCatalogModels(token, catalogQuery);
}

export function fetchCatalogDetail(token: string, model: string): Promise<CatalogModel> {
  return getCatalogModel(token, model);
}

export async function saveDownloadSource(
  token: string,
  source: DownloadSourceName,
  endpointText: string,
): Promise<SourceUpdate> {
  const endpoint = endpointText.trim() || null;
  const downloadSource = await updateDownloadSource(token, source, endpoint);
  const serverInfo = await getServerInfo(token);
  return { downloadSource, serverInfo };
}

export async function downloadCatalogModel(
  token: string,
  model: string,
  catalogQuery: CatalogQuery,
): Promise<CatalogDownloadUpdate> {
  const result = await downloadModel(token, model);
  const catalog = await listCatalogModels(token, catalogQuery);
  const serverInfo = await getServerInfo(token);
  return {
    catalog,
    serverInfo,
    noticeMessage: `Downloaded ${result.downloaded.length}, skipped ${result.skipped.length}.`,
  };
}

export async function loadNamedModel(
  token: string,
  modelName: string,
  inferenceParamsText: string,
  sourceOverride: DownloadSourceName | "",
  endpointOverride: string,
  catalogQuery: CatalogQuery,
): Promise<ModelLoadUpdate> {
  const target = modelName.trim();
  if (!target) {
    throw new Error("Model name is required.");
  }

  const params = parseParams(inferenceParamsText);
  const source = sourceOverride || undefined;
  const endpoint = endpointOverride.trim() ? endpointOverride.trim() : undefined;
  const result = await loadModel(token, target, params, source, endpoint);
  const health = await getHealth();
  const catalog = await listCatalogModels(token, catalogQuery);
  return {
    model: result.model,
    health,
    catalog,
    noticeMessage: `Loaded ${result.model.id}.`,
  };
}

export async function separateAudio(
  token: string,
  model: LoadedModel,
  audioFile: File,
  selectedStems: string[],
  responseFormat: SeparationResponseFormat,
  outputAudioFormat: OutputAudioFormat,
): Promise<SeparationUpdate> {
  const prepared = await prepareAudioFile(audioFile, model.pymss.sample_rate);
  const params = new URLSearchParams({
    model: model.id,
    format: "pcm_f32le",
    sample_rate: String(prepared.sampleRate),
    channels: String(prepared.channels),
    response_format: responseFormat,
    output_audio_format: outputAudioFormat,
  });
  if (selectedStems.length) {
    params.set("stems", selectedStems.join(","));
  }

  const response = await apiFetch(`/v1/audio/separations?${params.toString()}`, {
    method: "POST",
    token,
    headers: { "Content-Type": "application/octet-stream" },
    body: prepared.bytes,
  });

  if (responseFormat === "zip") {
    const blob = await response.blob();
    return {
      resultUrl: URL.createObjectURL(blob),
      resultName: `pymss-${model.id.replace(/\.[^.]+$/, "")}-${timestampName()}.zip`,
      jsonResult: null,
      noticeMessage: `Processed ${prepared.seconds.toFixed(1)} seconds.`,
    };
  }

  return {
    resultUrl: null,
    resultName: "",
    jsonResult: (await response.json()) as SeparationJsonResult,
    noticeMessage: `Processed ${prepared.seconds.toFixed(1)} seconds.`,
  };
}

export function revokeResultUrl(resultUrl: string | null): void {
  if (resultUrl) {
    URL.revokeObjectURL(resultUrl);
  }
}

export function saveJsonOutput(output: SeparationJsonResult["outputs"][number]): void {
  const extension = output.audio.format === "pcm_f32le" ? "f32le" : output.audio.format;
  const blob = base64ToBlob(output.audio.data, "application/octet-stream");
  downloadBlob(blob, `${output.stem}.${extension}`);
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function timestampName(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}
