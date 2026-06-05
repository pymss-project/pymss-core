import {
  downloadCatalogModel,
  fetchAuthenticatedData,
  fetchCatalog,
  fetchCatalogDetail,
  fetchHealth,
  loadNamedModel,
  revokeResultUrl,
  saveDownloadSource as saveDownloadSourceAction,
  saveJsonOutput,
  separateAudio,
  type AuthenticatedData,
} from "./app-actions";
import {
  appErrorText,
  clearStoredToken,
  createCatalogQuery,
  loadStoredTheme,
  loadStoredToken,
  requiresTokenPrompt,
  saveStoredTheme,
  saveStoredToken,
  selectedStemsForModel,
  toggleStemSelection,
} from "./app-state";
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
  Theme,
  View,
} from "./types";

export class AppController {
  activeView = $state<View>("dashboard");
  token = $state("");
  theme = $state<Theme>("light");
  showTokenModal = $state(false);
  tokenInput = $state("");
  busyAction = $state<string | null>(null);
  errorMessage = $state("");
  noticeMessage = $state("");

  health = $state<HealthResponse | null>(null);
  serverInfo = $state<ServerInfo | null>(null);
  downloadSource = $state<DownloadSource | null>(null);
  catalog = $state<CatalogListResponse | null>(null);
  selectedCatalog = $state<CatalogModel | null>(null);
  loadedModels = $state<LoadedModel[]>([]);
  loadedModel = $derived(this.loadedModels[0] ?? null);

  catalogQuery = $state<CatalogQuery>(createCatalogQuery());
  downloadSourceForm = $state<DownloadSourceName>("modelscope");
  downloadEndpointForm = $state("");
  loadModelName = $state("");
  loadSourceOverride = $state<DownloadSourceName | "">("");
  loadEndpointOverride = $state("");
  inferenceParamsText = $state("");

  audioFile = $state<File | null>(null);
  selectedStems = $state<string[]>([]);
  responseFormat = $state<SeparationResponseFormat>("zip");
  outputAudioFormat = $state<OutputAudioFormat>("wav");
  resultUrl = $state<string | null>(null);
  resultName = $state("");
  jsonResult = $state<SeparationJsonResult | null>(null);

  panelState = $derived({
    activeView: this.activeView,
    errorMessage: this.errorMessage,
    noticeMessage: this.noticeMessage,
    health: this.health,
    serverInfo: this.serverInfo,
    downloadSource: this.downloadSource,
    loadedModel: this.loadedModel,
    catalog: this.catalog,
    selectedCatalog: this.selectedCatalog,
    busyAction: this.busyAction,
    selectedStems: this.selectedStems,
    resultUrl: this.resultUrl,
    resultName: this.resultName,
    jsonResult: this.jsonResult,
    token: this.token,
  });

  panelActions = {
    onSelectView: (view: View) => (this.activeView = view),
    onDismissError: () => (this.errorMessage = ""),
    onDismissNotice: () => (this.noticeMessage = ""),
    onApplyCatalogFilters: () => void this.applyCatalogFilters(),
    onShowCatalogDetail: (model: string) => void this.showCatalogDetail(model),
    onDownload: (model: string) => void this.performDownload(model),
    onSetLoadTarget: (model: string) => this.setLoadTarget(model),
    onCloseDetail: () => (this.selectedCatalog = null),
    onLoad: () => void this.performLoad(),
    onToggleStem: (stem: string, checked: boolean) => this.toggleStem(stem, checked),
    onSeparate: () => void this.performSeparation(),
    onDownloadJsonOutput: saveJsonOutput,
    onSaveDownloadSource: () => void this.saveDownloadSource(),
    onOpenToken: () => (this.showTokenModal = true),
    onClearToken: () => this.clearToken(),
    onThemeChange: (theme: Theme) => this.setTheme(theme),
  };

  constructor() {
    $effect(() => {
      if (this.responseFormat === "json") {
        this.outputAudioFormat = "pcm_f32le";
      }
    });

    $effect(() => {
      document.documentElement.dataset.theme = this.theme;
    });
  }

  mount(): () => void {
    this.token = loadStoredToken();
    this.tokenInput = this.token;
    this.theme = loadStoredTheme();
    void this.refreshAll();
    return () => this.clearResultUrl();
  }

  setTheme(nextTheme: Theme): void {
    this.theme = nextTheme;
    saveStoredTheme(nextTheme);
  }

  async refreshAll(): Promise<void> {
    await this.runBusy("refresh", async () => {
      this.health = await fetchHealth();
      this.applyAuthenticatedData(await fetchAuthenticatedData(this.token, this.catalogQuery));
    });
  }

  saveToken(): void {
    this.token = this.tokenInput.trim();
    saveStoredToken(this.token);
    this.showTokenModal = false;
    void this.refreshAll();
  }

  clearToken(): void {
    this.token = "";
    this.tokenInput = "";
    clearStoredToken();
    this.serverInfo = null;
    this.downloadSource = null;
    this.catalog = null;
    this.loadedModels = [];
  }

  async saveDownloadSource(): Promise<void> {
    await this.runBusy("source", async () => {
      const update = await saveDownloadSourceAction(this.token, this.downloadSourceForm, this.downloadEndpointForm);
      this.downloadSource = update.downloadSource;
      this.downloadEndpointForm = update.downloadSource.endpoint ?? "";
      this.serverInfo = update.serverInfo;
      this.noticeMessage = "Download source updated.";
    });
  }

  async applyCatalogFilters(): Promise<void> {
    await this.runBusy("catalog", async () => {
      this.catalog = await fetchCatalog(this.token, this.catalogQuery);
    });
  }

  async showCatalogDetail(model: string): Promise<void> {
    await this.runBusy("detail", async () => {
      this.selectedCatalog = await fetchCatalogDetail(this.token, model);
    });
  }

  async performDownload(model: string): Promise<void> {
    await this.runBusy("download", async () => {
      const update = await downloadCatalogModel(this.token, model, this.catalogQuery);
      this.catalog = update.catalog;
      this.serverInfo = update.serverInfo;
      this.noticeMessage = update.noticeMessage;
    });
  }

  setLoadTarget(model: string): void {
    this.loadModelName = model;
    this.activeView = "loaded";
  }

  async performLoad(model = this.loadModelName): Promise<void> {
    const target = model.trim();
    if (!target) {
      this.errorMessage = "Model name is required.";
      return;
    }

    await this.runBusy("load", async () => {
      const update = await loadNamedModel(
        this.token,
        target,
        this.inferenceParamsText,
        this.loadSourceOverride,
        this.loadEndpointOverride,
        this.catalogQuery,
      );
      this.loadedModels = [update.model];
      this.syncSelectedStems(update.model);
      this.health = update.health;
      this.catalog = update.catalog;
      this.noticeMessage = update.noticeMessage;
    });
  }

  toggleStem(stem: string, checked: boolean): void {
    this.selectedStems = toggleStemSelection(this.selectedStems, stem, checked);
  }

  clearResultUrl(): void {
    revokeResultUrl(this.resultUrl);
    this.resultUrl = null;
  }

  async performSeparation(): Promise<void> {
    const model = this.loadedModel;
    const audioFile = this.audioFile;
    if (!model) {
      this.errorMessage = "Load a model before separation.";
      return;
    }
    if (!audioFile) {
      this.errorMessage = "Select an audio file.";
      return;
    }

    await this.runBusy("separate", async () => {
      this.clearResultUrl();
      this.jsonResult = null;
      const update = await separateAudio(
        this.token,
        model,
        audioFile,
        this.selectedStems,
        this.responseFormat,
        this.outputAudioFormat,
      );
      this.resultUrl = update.resultUrl;
      this.resultName = update.resultName;
      this.jsonResult = update.jsonResult;
      this.noticeMessage = update.noticeMessage;
    });
  }

  private applyAuthenticatedData(data: AuthenticatedData): void {
    this.serverInfo = data.serverInfo;
    this.downloadSource = data.downloadSource;
    this.downloadSourceForm = data.downloadSource.source;
    this.downloadEndpointForm = data.downloadSource.endpoint ?? "";
    this.loadedModels = data.loadedModels;
    this.catalog = data.catalog;
    this.syncSelectedStems(data.loadedModels[0] ?? null);
  }

  private syncSelectedStems(model = this.loadedModels[0] ?? null): void {
    this.selectedStems = selectedStemsForModel(this.selectedStems, model);
  }

  private handleError(error: unknown): void {
    this.errorMessage = appErrorText(error);
    if (requiresTokenPrompt(error)) {
      this.showTokenModal = true;
    }
  }

  private async runBusy(label: string, task: () => Promise<void>): Promise<void> {
    this.busyAction = label;
    this.errorMessage = "";
    this.noticeMessage = "";
    try {
      await task();
    } catch (error) {
      this.handleError(error);
    } finally {
      this.busyAction = null;
    }
  }
}
