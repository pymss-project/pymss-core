<script lang="ts">
  import { Download, Loader2, Play, UploadCloud } from "@lucide/svelte";
  import type { LoadedModel, OutputAudioFormat, SeparationJsonResult, SeparationResponseFormat } from "../types";

  export let loadedModel: LoadedModel | null;
  export let busyAction: string | null;
  export let audioFile: File | null;
  export let selectedStems: string[];
  export let responseFormat: SeparationResponseFormat;
  export let outputAudioFormat: OutputAudioFormat;
  export let resultUrl: string | null;
  export let resultName: string;
  export let jsonResult: SeparationJsonResult | null;
  export let onToggleStem: (stem: string, checked: boolean) => void;
  export let onSeparate: () => void;
  export let onDownloadJsonOutput: (output: SeparationJsonResult["outputs"][number]) => void;
</script>

<section class="grid gap-8 xl:grid-cols-[minmax(0,1fr)_22rem]">
  <div class="space-y-5">
    <div>
      <h1 class="text-[30px] font-medium leading-tight tracking-normal">Separate audio</h1>
      <p class="mt-2 text-sm text-base-content/60">
        {loadedModel ? `Using ${loadedModel.id}` : "Load a model before running separation."}
      </p>
    </div>

    <section class="rounded-box border border-base-300 bg-base-100 p-5">
      <form class="space-y-5" on:submit|preventDefault={onSeparate}>
        <label class="field">
          <span class="field-label">Audio file</span>
          <input
            class="file-field"
            type="file"
            accept="audio/*"
            on:change={(event) => (audioFile = (event.currentTarget as HTMLInputElement).files?.[0] ?? null)}
          />
        </label>

        <div>
          <div class="mb-2 text-sm font-medium">Stems</div>
          <div class="flex flex-wrap gap-2">
            {#each loadedModel?.pymss.instruments ?? [] as stem}
              <label class="btn btn-outline btn-sm rounded-full">
                <input
                  class="check-field"
                  type="checkbox"
                  checked={selectedStems.includes(stem)}
                  on:change={(event) => onToggleStem(stem, (event.currentTarget as HTMLInputElement).checked)}
                />
                {stem}
              </label>
            {/each}
          </div>
        </div>

        <div class="grid gap-3 md:grid-cols-2">
          <label class="field">
            <span class="field-label">Response</span>
            <select class="select-field" bind:value={responseFormat}>
              <option value="zip">zip</option>
              <option value="json">json</option>
            </select>
          </label>
          <label class="field">
            <span class="field-label">Output</span>
            <select class="select-field" bind:value={outputAudioFormat} disabled={responseFormat === "json"}>
              <option value="wav">wav</option>
              <option value="flac">flac</option>
              <option value="pcm_f32le">pcm_f32le</option>
            </select>
          </label>
        </div>

        <button class="btn btn-primary rounded-full" type="submit" disabled={busyAction !== null || !loadedModel}>
          {#if busyAction === "separate"}
            <Loader2 class="size-4 animate-spin" />
          {:else}
            <Play class="size-4" />
          {/if}
          Separate
        </button>
      </form>
    </section>
  </div>

  <section class="rounded-box border border-base-300 bg-base-100 p-5">
    <h2 class="mb-4 text-xl font-medium">Result</h2>
    {#if resultUrl}
      <a class="btn btn-primary w-full rounded-full" href={resultUrl} download={resultName}>
        <Download class="size-4" /> Download ZIP
      </a>
    {:else if jsonResult}
      <div class="space-y-3">
        <div class="text-sm text-base-content/60">{jsonResult.metadata.input_seconds.toFixed(1)} seconds · {jsonResult.outputs.length} output(s)</div>
        {#each jsonResult.outputs as output}
          <button class="btn btn-outline btn-sm w-full justify-between rounded-full" type="button" on:click={() => onDownloadJsonOutput(output)}>
            <span>{output.stem}</span>
            <Download class="size-4" />
          </button>
        {/each}
      </div>
    {:else}
      <div class="alert border border-base-300 bg-base-200">
        <UploadCloud class="size-5" />
        <span>No result yet.</span>
      </div>
    {/if}
  </section>
</section>
