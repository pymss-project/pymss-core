<script lang="ts">
  import { ArrowRight, FileAudio, Library, SlidersHorizontal } from "@lucide/svelte";
  import { formatBytes } from "../params";
  import type { DownloadSource, HealthResponse, LoadedModel, ServerInfo, View } from "../types";

  export let health: HealthResponse | null;
  export let serverInfo: ServerInfo | null;
  export let downloadSource: DownloadSource | null;
  export let loadedModel: LoadedModel | null;
  export let onSelectView: (view: View) => void;

  const actions = [
    { view: "catalog" as View, label: "Browse catalog", icon: Library },
    { view: "loaded" as View, label: "Load model", icon: SlidersHorizontal },
    { view: "separate" as View, label: "Separate audio", icon: FileAudio },
  ];
</script>

<section class="space-y-8">
  <div class="mx-auto max-w-3xl text-center">
    <div class="mb-4 flex justify-center">
      <span class="badge badge-outline rounded-full px-4 py-3">{health?.status ?? "offline"}</span>
    </div>
    <h1 class="text-[32px] font-medium leading-tight tracking-normal sm:text-4xl">Model separation server</h1>
    <p class="mx-auto mt-3 max-w-xl text-base leading-7 text-base-content/65">
      {loadedModel ? loadedModel.id : "Load a model, then send audio from the browser for separation."}
    </p>
  </div>

  <div class="mx-auto max-w-3xl rounded-full border border-base-300 bg-base-200 px-5 py-3 font-mono text-sm text-base-content">
    <div class="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
      <span>source={downloadSource?.source ?? "-"}</span>
      <span>device={health?.device ?? "-"}</span>
      <span>limit={formatBytes(serverInfo?.limits.max_request_bytes ?? 0)}</span>
    </div>
  </div>

  <div class="mx-auto grid max-w-3xl gap-3 md:grid-cols-3">
    {#each actions as action, index}
      <button
        class="btn {index === 0 ? 'btn-primary' : 'btn-outline'} h-auto min-h-12 rounded-full px-5"
        type="button"
        on:click={() => onSelectView(action.view)}
      >
        <svelte:component this={action.icon} class="size-4" />
        {action.label}
        {#if index === 0}
          <ArrowRight class="size-4" />
        {/if}
      </button>
    {/each}
  </div>
</section>
