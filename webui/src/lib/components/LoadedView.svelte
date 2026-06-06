<script lang="ts">
  import { Info, Loader2 } from "@lucide/svelte";
  import { Alert, AlertDescription } from "$lib/components/ui/alert/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/card/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Label } from "$lib/components/ui/label/index.js";
  import { Textarea } from "$lib/components/ui/textarea/index.js";
  import SelectField from "./SelectField.svelte";
  import type { DownloadSourceName, LoadedModel } from "../types";

  export let loadedModel: LoadedModel | null;
  export let busyAction: string | null;
  export let loadModelName: string;
  export let loadSourceOverride: DownloadSourceName | "";
  export let loadEndpointOverride: string;
  export let inferenceParamsText: string;
  export let onLoad: () => void;

  const sourceOptions = [
    { value: "", label: "current default" },
    { value: "modelscope", label: "modelscope" },
    { value: "huggingface", label: "huggingface" },
    { value: "hf-mirror", label: "hf-mirror" },
  ];
</script>

<section class="grid gap-8 xl:grid-cols-[minmax(0,1fr)_22rem]">
  <div class="space-y-5">
    <div>
      <h1 class="text-[30px] font-medium leading-tight tracking-normal">Loaded model</h1>
      <p class="mt-2 text-sm text-muted-foreground">
        {loadedModel ? `${loadedModel.pymss.sample_rate} Hz · ${loadedModel.pymss.category}` : "No model is loaded."}
      </p>
    </div>

    <Card>
      <CardContent>
      {#if loadedModel}
        <h2 class="break-words text-xl font-medium">{loadedModel.id}</h2>
        <div class="mt-4 flex flex-wrap gap-2">
          {#each loadedModel.pymss.instruments as stem}
            <Badge variant="outline">{stem}</Badge>
          {/each}
        </div>
        <div class="mt-6 divide-y divide-border text-sm">
          <div class="flex justify-between gap-4 py-3">
            <span class="text-muted-foreground">Type</span>
            <span>{loadedModel.pymss.model_type ?? loadedModel.pymss.architecture}</span>
          </div>
          <div class="flex justify-between gap-4 py-3">
            <span class="text-muted-foreground">Parameters</span>
            <span class="text-right">{Object.keys(loadedModel.pymss.supported_parameters).join(", ") || "-"}</span>
          </div>
        </div>
      {:else}
        <Alert>
          <Info class="size-5" />
          <AlertDescription>Choose a catalog model or enter an alias.</AlertDescription>
        </Alert>
      {/if}
      </CardContent>
    </Card>
  </div>

  <Card>
    <CardHeader>
      <CardTitle>Load</CardTitle>
    </CardHeader>
    <CardContent>
    <form class="space-y-4" on:submit|preventDefault={onLoad}>
      <div class="grid gap-1.5">
        <Label for="load-model">Model</Label>
        <Input id="load-model" class="h-9 text-sm" bind:value={loadModelName} placeholder="bs_roformer_voc_hyperacev2" />
      </div>
      <SelectField
        id="load-source"
        label="Source"
        value={loadSourceOverride}
        options={sourceOptions}
        onValueChange={(value) => (loadSourceOverride = value as DownloadSourceName | "")}
      />
      <div class="grid gap-1.5">
        <Label for="load-endpoint">Endpoint</Label>
        <Input id="load-endpoint" class="h-9 text-sm" bind:value={loadEndpointOverride} placeholder="optional" />
      </div>
      <div class="grid gap-1.5">
        <Label for="inference-params">Inference params</Label>
        <Textarea id="inference-params" class="min-h-28 font-mono text-sm" bind:value={inferenceParamsText} placeholder={"batch_size=2\nnormalize=true"} />
      </div>
      <Button class="w-full" type="submit" disabled={busyAction !== null}>
        {#if busyAction === "load"}
          <Loader2 class="size-4 animate-spin" />
        {/if}
        Load
      </Button>
    </form>
    </CardContent>
  </Card>
</section>
