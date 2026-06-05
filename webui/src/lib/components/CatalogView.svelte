<script lang="ts">
  import { Download, Info, Loader2 } from "@lucide/svelte";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/card/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Label } from "$lib/components/ui/label/index.js";
  import * as Table from "$lib/components/ui/table/index.js";
  import SelectField from "./SelectField.svelte";
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

  const supportedOptions = [
    { value: "true", label: "supported" },
    { value: "all", label: "all" },
    { value: "false", label: "unsupported" },
  ];

  const localOptions = [
    { value: "all", label: "all" },
    { value: "complete", label: "complete" },
    { value: "missing", label: "missing" },
  ];
</script>

<section class="space-y-6">
  <div class="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
    <div>
      <h1 class="text-[30px] font-medium leading-tight tracking-normal">Catalog</h1>
      <p class="mt-2 text-sm text-muted-foreground">{catalog?.pymss.total ?? 0} model(s), {catalog?.pymss.source ?? "source unset"}</p>
    </div>
    <Button form="catalog-search-form" type="submit" disabled={busyAction !== null}>
      {#if busyAction === "catalog"}
        <Loader2 class="size-4 animate-spin" />
      {/if}
      Search
    </Button>
  </div>

  <Card>
    <CardContent>
      <form id="catalog-search-form" class="grid gap-3 lg:grid-cols-[10rem_10rem_minmax(0,1fr)_minmax(0,1.2fr)]" on:submit|preventDefault={onApplyFilters}>
        <SelectField
          id="catalog-supported"
          label="Supported"
          value={catalogQuery.supported}
          options={supportedOptions}
          onValueChange={(value) => (catalogQuery.supported = value as CatalogQuery["supported"])}
        />
        <SelectField
          id="catalog-local"
          label="Local"
          value={catalogQuery.local}
          options={localOptions}
          onValueChange={(value) => (catalogQuery.local = value as CatalogQuery["local"])}
        />
        <div class="grid gap-1.5">
          <Label for="catalog-category">Category</Label>
          <Input id="catalog-category" class="h-9 text-sm" bind:value={catalogQuery.category} placeholder="vocal" />
        </div>
        <div class="grid gap-1.5">
          <Label for="catalog-search">Search</Label>
          <Input id="catalog-search" class="h-9 text-sm" bind:value={catalogQuery.q} placeholder="model, alias, stem" />
        </div>
      </form>
    </CardContent>
  </Card>

  <Card class="py-0">
    <CardContent class="px-0">
      <Table.Root>
        <Table.Body>
          {#each catalog?.data ?? [] as model}
            <Table.Row>
              <Table.Cell class="whitespace-normal p-3">
                <div class="max-w-xl">
                  <div class="font-medium">{model.id}</div>
                  <div class="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                    <span>{model.pymss.model_type ?? model.pymss.architecture}</span>
                    <span>{model.pymss.category}</span>
                    <span>{model.pymss.target_stem}</span>
                    <span>{formatBytes(model.pymss.size_bytes)}</span>
                  </div>
                  {#if !model.pymss.supported}
                    <div class="mt-1 text-xs text-destructive">{model.pymss.unsupported_reason}</div>
                  {/if}
                </div>
              </Table.Cell>
              <Table.Cell class="hidden text-right sm:table-cell">
                <Badge variant="outline">
                  {model.pymss.local.complete ? "local" : `${model.pymss.local.missing_count} missing`}
                </Badge>
              </Table.Cell>
              <Table.Cell>
                <div class="flex justify-end gap-1">
                  <Button variant="ghost" size="icon-sm" type="button" onclick={() => onShowDetail(model.id)} aria-label="Detail">
                    <Info class="size-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    type="button"
                    onclick={() => onDownload(model.id)}
                    disabled={busyAction !== null}
                    aria-label="Download"
                  >
                    <Download class="size-4" />
                  </Button>
                  <Button
                    size="sm"
                    type="button"
                    onclick={() => onSetLoadTarget(model.id)}
                    disabled={!model.pymss.supported}
                  >
                    Load
                  </Button>
                </div>
              </Table.Cell>
            </Table.Row>
          {/each}
        </Table.Body>
      </Table.Root>
    </CardContent>
  </Card>

  {#if selectedCatalog}
    <Card>
      <CardHeader class="flex-row items-start justify-between">
        <div>
          <CardTitle class="break-words text-xl">{selectedCatalog.id}</CardTitle>
          <p class="mt-1 text-sm text-muted-foreground">{selectedCatalog.pymss.aliases.join(", ") || "No aliases"}</p>
        </div>
        <Button variant="ghost" size="sm" type="button" onclick={onCloseDetail}>Close</Button>
      </CardHeader>
      <CardContent>
        <div class="divide-y divide-border">
          {#each selectedCatalog.pymss.files ?? [] as file}
            <div class="grid gap-2 py-3 text-sm md:grid-cols-[7rem_minmax(0,1fr)_7rem]">
              <span class="text-muted-foreground">{file.role}</span>
              <span class="truncate font-mono text-xs">{file.relpath}</span>
              <span class="text-right">{file.exists ? "exists" : "missing"} · {formatBytes(file.size_bytes)}</span>
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}
</section>
