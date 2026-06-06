<script lang="ts">
  import { Download, Loader2, Play, UploadCloud } from "@lucide/svelte";
  import { Alert, AlertDescription } from "$lib/components/ui/alert/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/card/index.js";
  import { Checkbox } from "$lib/components/ui/checkbox/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Label } from "$lib/components/ui/label/index.js";
  import SelectField from "./SelectField.svelte";
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

  const responseOptions = [
    { value: "zip", label: "zip" },
    { value: "json", label: "json" },
  ];

  const outputOptions = [
    { value: "wav", label: "wav" },
    { value: "flac", label: "flac" },
    { value: "pcm_f32le", label: "pcm_f32le" },
  ];
</script>

<section class="grid gap-8 xl:grid-cols-[minmax(0,1fr)_22rem]">
  <div class="space-y-5">
    <div>
      <h1 class="text-[30px] font-medium leading-tight tracking-normal">Separate audio</h1>
      <p class="mt-2 text-sm text-muted-foreground">
        {loadedModel ? `Using ${loadedModel.id}` : "Load a model before running separation."}
      </p>
    </div>

    <Card>
      <CardContent>
      <form class="space-y-5" on:submit|preventDefault={onSeparate}>
        <div class="grid gap-1.5">
          <Label for="audio-file">Audio file</Label>
          <Input
            id="audio-file"
            type="file"
            accept="audio/*"
            onchange={(event) => (audioFile = (event.currentTarget as HTMLInputElement).files?.[0] ?? null)}
          />
        </div>

        <div>
          <div class="mb-2 text-sm font-medium">Stems</div>
          <div class="flex flex-wrap gap-2">
            {#each loadedModel?.pymss.instruments ?? [] as stem}
              <div class="flex h-8 items-center gap-2 border border-input px-2.5">
                <Checkbox
                  id={`stem-${stem}`}
                  checked={selectedStems.includes(stem)}
                  onCheckedChange={(checked) => onToggleStem(stem, checked)}
                />
                <Label for={`stem-${stem}`} class="text-sm">{stem}</Label>
              </div>
            {/each}
          </div>
        </div>

        <div class="grid gap-3 md:grid-cols-2">
          <SelectField
            id="response-format"
            label="Response"
            value={responseFormat}
            options={responseOptions}
            onValueChange={(value) => (responseFormat = value as SeparationResponseFormat)}
          />
          <SelectField
            id="output-format"
            label="Output"
            value={outputAudioFormat}
            options={outputOptions}
            disabled={responseFormat === "json"}
            onValueChange={(value) => (outputAudioFormat = value as OutputAudioFormat)}
          />
        </div>

        <Button type="submit" disabled={busyAction !== null || !loadedModel}>
          {#if busyAction === "separate"}
            <Loader2 class="size-4 animate-spin" />
          {:else}
            <Play class="size-4" />
          {/if}
          Separate
        </Button>
      </form>
      </CardContent>
    </Card>
  </div>

  <Card>
    <CardHeader>
      <CardTitle>Result</CardTitle>
    </CardHeader>
    <CardContent>
    {#if resultUrl}
      <Button class="w-full" href={resultUrl} download={resultName}>
        <Download class="size-4" /> Download ZIP
      </Button>
    {:else if jsonResult}
      <div class="space-y-3">
        <div class="text-sm text-muted-foreground">{jsonResult.metadata.input_seconds.toFixed(1)} seconds · {jsonResult.outputs.length} output(s)</div>
        {#each jsonResult.outputs as output}
          <Button variant="outline" size="sm" class="w-full justify-between" type="button" onclick={() => onDownloadJsonOutput(output)}>
            <span>{output.stem}</span>
            <Download class="size-4" />
          </Button>
        {/each}
      </div>
    {:else}
      <Alert>
        <UploadCloud class="size-5" />
        <AlertDescription>No result yet.</AlertDescription>
      </Alert>
    {/if}
    </CardContent>
  </Card>
</section>
