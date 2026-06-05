<script lang="ts">
  import { Info, Loader2 } from "@lucide/svelte";
  import type { DownloadSourceName, LoadedModel } from "../types";

  export let loadedModel: LoadedModel | null;
  export let busyAction: string | null;
  export let loadModelName: string;
  export let loadSourceOverride: DownloadSourceName | "";
  export let loadEndpointOverride: string;
  export let inferenceParamsText: string;
  export let onLoad: () => void;
</script>

<section class="grid gap-8 xl:grid-cols-[minmax(0,1fr)_22rem]">
  <div class="space-y-5">
    <div>
      <h1 class="text-[30px] font-medium leading-tight tracking-normal">Loaded model</h1>
      <p class="mt-2 text-sm text-base-content/60">
        {loadedModel ? `${loadedModel.pymss.sample_rate} Hz · ${loadedModel.pymss.category}` : "No model is loaded."}
      </p>
    </div>

    <section class="rounded-box border border-base-300 bg-base-100 p-5">
      {#if loadedModel}
        <h2 class="break-words text-xl font-medium">{loadedModel.id}</h2>
        <div class="mt-4 flex flex-wrap gap-2">
          {#each loadedModel.pymss.instruments as stem}
            <span class="badge badge-outline rounded-full">{stem}</span>
          {/each}
        </div>
        <div class="mt-6 divide-y divide-base-300 text-sm">
          <div class="flex justify-between gap-4 py-3">
            <span class="text-base-content/60">Type</span>
            <span>{loadedModel.pymss.model_type ?? loadedModel.pymss.architecture}</span>
          </div>
          <div class="flex justify-between gap-4 py-3">
            <span class="text-base-content/60">Parameters</span>
            <span class="text-right">{Object.keys(loadedModel.pymss.supported_parameters).join(", ") || "-"}</span>
          </div>
        </div>
      {:else}
        <div class="alert border border-base-300 bg-base-200">
          <Info class="size-5" />
          <span>Choose a catalog model or enter an alias.</span>
        </div>
      {/if}
    </section>
  </div>

  <section class="rounded-box border border-base-300 bg-base-100 p-5">
    <h2 class="mb-4 text-xl font-medium">Load</h2>
    <form class="space-y-4" on:submit|preventDefault={onLoad}>
      <label class="field">
        <span class="field-label">Model</span>
        <input class="text-field" bind:value={loadModelName} placeholder="bs_roformer_voc_hyperacev2" />
      </label>
      <label class="field">
        <span class="field-label">Source</span>
        <select class="select-field" bind:value={loadSourceOverride}>
          <option value="">current default</option>
          <option value="modelscope">modelscope</option>
          <option value="huggingface">huggingface</option>
          <option value="hf-mirror">hf-mirror</option>
        </select>
      </label>
      <label class="field">
        <span class="field-label">Endpoint</span>
        <input class="text-field" bind:value={loadEndpointOverride} placeholder="optional" />
      </label>
      <label class="field">
        <span class="field-label">Inference params</span>
        <textarea class="text-field textarea-field font-mono text-sm" bind:value={inferenceParamsText} placeholder={"batch_size=2\nnormalize=true"}></textarea>
      </label>
      <button class="btn btn-primary w-full rounded-full" type="submit" disabled={busyAction !== null}>
        {#if busyAction === "load"}
          <Loader2 class="size-4 animate-spin" />
        {/if}
        Load
      </button>
    </form>
  </section>
</section>
