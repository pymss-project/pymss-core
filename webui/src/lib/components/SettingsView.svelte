<script lang="ts">
  import { Loader2 } from "@lucide/svelte";
  import { formatBytes } from "../params";
  import type { DownloadSourceName, ServerInfo, Theme } from "../types";

  export let serverInfo: ServerInfo | null;
  export let busyAction: string | null;
  export let token: string;
  export let theme: Theme;
  export let downloadSourceForm: DownloadSourceName;
  export let downloadEndpointForm: string;
  export let onSaveDownloadSource: () => void;
  export let onOpenToken: () => void;
  export let onClearToken: () => void;
  export let onThemeChange: (theme: Theme) => void;
</script>

<section class="space-y-8">
  <div>
    <h1 class="text-[30px] font-medium leading-tight tracking-normal">Settings</h1>
    <p class="mt-2 text-sm text-base-content/60">Download source, token and runtime limits.</p>
  </div>

  <div class="grid gap-6 xl:grid-cols-2">
    <section class="rounded-box border border-base-300 bg-base-100 p-5">
      <h2 class="mb-4 text-xl font-medium">Download source</h2>
      <form class="space-y-4" on:submit|preventDefault={onSaveDownloadSource}>
        <label class="field">
          <span class="field-label">Source</span>
          <select class="select-field" bind:value={downloadSourceForm}>
            <option value="modelscope">modelscope</option>
            <option value="huggingface">huggingface</option>
            <option value="hf-mirror">hf-mirror</option>
          </select>
        </label>
        <label class="field">
          <span class="field-label">Endpoint</span>
          <input class="text-field" bind:value={downloadEndpointForm} placeholder="optional" />
        </label>
        <button class="btn btn-primary rounded-full" type="submit" disabled={busyAction !== null}>
          {#if busyAction === "source"}
            <Loader2 class="size-4 animate-spin" />
          {/if}
          Save
        </button>
      </form>
    </section>

    <section class="rounded-box border border-base-300 bg-base-100 p-5">
      <h2 class="mb-4 text-xl font-medium">Access</h2>
      <div class="space-y-4">
        <div class="flex items-center justify-between gap-3">
          <span class="text-sm text-base-content/60">API token</span>
          <span class="badge badge-outline rounded-full">{token ? "stored" : "empty"}</span>
        </div>
        <div class="flex flex-wrap gap-2">
          <button class="btn btn-outline rounded-full" type="button" on:click={onOpenToken}>Edit token</button>
          <button class="btn btn-ghost rounded-full" type="button" on:click={onClearToken}>Clear</button>
        </div>
        <label class="field">
          <span class="field-label">Theme</span>
          <select class="select-field" bind:value={theme} on:change={() => onThemeChange(theme)}>
            <option value="light">light</option>
            <option value="dark">dark</option>
          </select>
        </label>
      </div>
    </section>
  </div>

  <section class="rounded-box border border-base-300 bg-base-100 p-5">
    <h2 class="mb-4 text-xl font-medium">Runtime</h2>
    <div class="divide-y divide-base-300 text-sm">
      <div class="grid gap-2 py-3 md:grid-cols-[10rem_minmax(0,1fr)]">
        <span class="text-base-content/60">Model dir</span>
        <span class="break-all font-mono text-xs">{serverInfo?.model_dir ?? "-"}</span>
      </div>
      <div class="grid gap-2 py-3 md:grid-cols-[10rem_minmax(0,1fr)]">
        <span class="text-base-content/60">WebUI</span>
        <span>{serverInfo?.webui.path ?? "-"}</span>
      </div>
      <div class="grid gap-2 py-3 md:grid-cols-[10rem_minmax(0,1fr)]">
        <span class="text-base-content/60">Request limit</span>
        <span>{formatBytes(serverInfo?.limits.max_request_bytes ?? 0)}</span>
      </div>
      <div class="grid gap-2 py-3 md:grid-cols-[10rem_minmax(0,1fr)]">
        <span class="text-base-content/60">Timeout</span>
        <span>{serverInfo?.limits.request_timeout_seconds ?? "-"} seconds</span>
      </div>
    </div>
  </section>
</section>
