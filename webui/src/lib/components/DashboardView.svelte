<script lang="ts">
  import {
    ArrowRight,
    FileAudio,
    Library,
    SlidersHorizontal,
  } from "@lucide/svelte";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Card, CardContent } from "$lib/components/ui/card/index.js";
  import { formatBytes } from "../params";
  import type {
    DownloadSource,
    HealthResponse,
    LoadedModel,
    ServerInfo,
    View,
  } from "../types";

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
    <h1
      class="text-[32px] font-medium leading-tight tracking-normal sm:text-4xl"
    >
      Model separation server
    </h1>
    <p class="mx-auto mt-3 max-w-xl text-base leading-7 text-muted-foreground">
      {loadedModel
        ? loadedModel.id
        : "Load a model, then send audio from the browser for separation."}
    </p>
  </div>

  <Card size="sm" class="mx-auto max-w-3xl">
    <CardContent>
      <div
        class="flex flex-col gap-1 font-mono text-sm sm:flex-row sm:items-center sm:justify-between"
      >
        <span>source={downloadSource?.source ?? "-"}</span>
        <span>device={health?.device ?? "-"}</span>
        <span
          >limit={formatBytes(serverInfo?.limits.max_request_bytes ?? 0)}</span
        >
      </div>
    </CardContent>
  </Card>

  <div class="mx-auto grid max-w-3xl gap-3 md:grid-cols-3">
    {#each actions as action, index}
      <Button
        variant={index === 0 ? "default" : "outline"}
        size="lg"
        class="h-12 justify-between px-4"
        type="button"
        onclick={() => onSelectView(action.view)}
      >
        <svelte:component this={action.icon} class="size-4" />
        {action.label}
        {#if index === 0}
          <ArrowRight class="size-4" />
        {/if}
      </Button>
    {/each}
  </div>
</section>
