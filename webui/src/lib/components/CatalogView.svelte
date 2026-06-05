<script lang="ts">
  import { Download, Info, Loader2 } from "@lucide/svelte";
  import { formatBytes } from "../params";
  import type { CatalogListResponse, CatalogModel, CatalogQuery } from "../types";

  export let catalog: CatalogListResponse | null;
  export let selectedCatalog: CatalogModel | null;
  export let catalogQuery: CatalogQuery;
  export let busyAction: string | null;
  export let onApplyFilters: () => void;
  export let onShowDetail: (model: string) => void;
  export let onDownload: (model: string) => void;
  export let onSetLoadTarget: (model: string) => void;
  export let onCloseDetail: () => void;
</script>

<section class="space-y-6">
  <div class="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
    <div>
      <h1 class="text-[30px] font-medium leading-tight tracking-normal">Catalog</h1>
      <p class="mt-2 text-sm text-base-content/60">{catalog?.pymss.total ?? 0} model(s), {catalog?.pymss.source ?? "source unset"}</p>
    </div>
    <button class="btn btn-primary rounded-full" type="button" on:click={onApplyFilters} disabled={busyAction !== null}>
      {#if busyAction === "catalog"}
        <Loader2 class="size-4 animate-spin" />
      {/if}
      Apply
    </button>
  </div>

  <section class="rounded-box border border-base-300 bg-base-100 p-4">
    <div class="grid gap-3 lg:grid-cols-[10rem_10rem_minmax(0,1fr)_minmax(0,1.2fr)]">
      <label class="field">
        <span class="field-label">Supported</span>
        <select class="select-field" bind:value={catalogQuery.supported}>
          <option value="true">supported</option>
          <option value="all">all</option>
          <option value="false">unsupported</option>
        </select>
      </label>
      <label class="field">
        <span class="field-label">Local</span>
        <select class="select-field" bind:value={catalogQuery.local}>
          <option value="all">all</option>
          <option value="complete">complete</option>
          <option value="missing">missing</option>
        </select>
      </label>
      <label class="field">
        <span class="field-label">Category</span>
        <input class="text-field" bind:value={catalogQuery.category} placeholder="vocal" />
      </label>
      <label class="field">
        <span class="field-label">Search</span>
        <input class="text-field" bind:value={catalogQuery.q} placeholder="model, alias, stem" />
      </label>
    </div>
  </section>

  <section class="rounded-box border border-base-300 bg-base-100">
    <div class="overflow-x-auto">
      <table class="table">
        <tbody>
          {#each catalog?.data ?? [] as model}
            <tr class="border-base-300">
              <td>
                <div class="max-w-xl">
                  <div class="font-medium">{model.id}</div>
                  <div class="mt-1 flex flex-wrap gap-2 text-xs text-base-content/60">
                    <span>{model.pymss.model_type ?? model.pymss.architecture}</span>
                    <span>{model.pymss.category}</span>
                    <span>{model.pymss.target_stem}</span>
                    <span>{formatBytes(model.pymss.size_bytes)}</span>
                  </div>
                  {#if !model.pymss.supported}
                    <div class="mt-1 text-xs text-error">{model.pymss.unsupported_reason}</div>
                  {/if}
                </div>
              </td>
              <td class="hidden text-right sm:table-cell">
                <span class="badge badge-outline rounded-full">
                  {model.pymss.local.complete ? "local" : `${model.pymss.local.missing_count} missing`}
                </span>
              </td>
              <td>
                <div class="flex justify-end gap-1">
                  <button class="btn btn-ghost btn-circle btn-sm" type="button" on:click={() => onShowDetail(model.id)} aria-label="Detail">
                    <Info class="size-4" />
                  </button>
                  <button
                    class="btn btn-ghost btn-circle btn-sm"
                    type="button"
                    on:click={() => onDownload(model.id)}
                    disabled={busyAction !== null}
                    aria-label="Download"
                  >
                    <Download class="size-4" />
                  </button>
                  <button
                    class="btn btn-primary btn-sm rounded-full"
                    type="button"
                    on:click={() => onSetLoadTarget(model.id)}
                    disabled={!model.pymss.supported}
                  >
                    Load
                  </button>
                </div>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  {#if selectedCatalog}
    <section class="rounded-box border border-base-300 bg-base-100 p-5">
      <div class="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 class="text-xl font-medium">{selectedCatalog.id}</h2>
          <p class="mt-1 text-sm text-base-content/60">{selectedCatalog.pymss.aliases.join(", ") || "No aliases"}</p>
        </div>
        <button class="btn btn-ghost btn-sm rounded-full" type="button" on:click={onCloseDetail}>Close</button>
      </div>
      <div class="divide-y divide-base-300">
        {#each selectedCatalog.pymss.files ?? [] as file}
          <div class="grid gap-2 py-3 text-sm md:grid-cols-[7rem_minmax(0,1fr)_7rem]">
            <span class="text-base-content/60">{file.role}</span>
            <span class="truncate font-mono text-xs">{file.relpath}</span>
            <span class="text-right">{file.exists ? "exists" : "missing"} · {formatBytes(file.size_bytes)}</span>
          </div>
        {/each}
      </div>
    </section>
  {/if}
</section>
