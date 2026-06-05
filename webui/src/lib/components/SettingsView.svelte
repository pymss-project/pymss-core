<script lang="ts">
  import { Loader2 } from "@lucide/svelte";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/card/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Label } from "$lib/components/ui/label/index.js";
  import SelectField from "./SelectField.svelte";
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

  const sourceOptions = [
    { value: "modelscope", label: "modelscope" },
    { value: "huggingface", label: "huggingface" },
    { value: "hf-mirror", label: "hf-mirror" },
  ];

  const themeOptions = [
    { value: "light", label: "light" },
    { value: "dark", label: "dark" },
  ];
</script>

<section class="space-y-8">
  <div>
    <h1 class="text-[30px] font-medium leading-tight tracking-normal">Settings</h1>
    <p class="mt-2 text-sm text-muted-foreground">Download source, token and runtime limits.</p>
  </div>

  <div class="grid gap-6 xl:grid-cols-2">
    <Card>
      <CardHeader>
        <CardTitle>Download source</CardTitle>
      </CardHeader>
      <CardContent>
      <form class="space-y-4" on:submit|preventDefault={onSaveDownloadSource}>
        <SelectField
          id="settings-source"
          label="Source"
          value={downloadSourceForm}
          options={sourceOptions}
          onValueChange={(value) => (downloadSourceForm = value as DownloadSourceName)}
        />
        <div class="grid gap-1.5">
          <Label for="settings-endpoint">Endpoint</Label>
          <Input id="settings-endpoint" class="h-9 text-sm" bind:value={downloadEndpointForm} placeholder="optional" />
        </div>
        <Button type="submit" disabled={busyAction !== null}>
          {#if busyAction === "source"}
            <Loader2 class="size-4 animate-spin" />
          {/if}
          Save
        </Button>
      </form>
      </CardContent>
    </Card>

    <Card>
      <CardHeader>
        <CardTitle>Access</CardTitle>
      </CardHeader>
      <CardContent>
      <div class="space-y-4">
        <div class="flex items-center justify-between gap-3">
          <span class="text-sm text-muted-foreground">API token</span>
          <Badge variant="outline">{token ? "stored" : "empty"}</Badge>
        </div>
        <div class="flex flex-wrap gap-2">
          <Button variant="outline" type="button" onclick={onOpenToken}>Edit token</Button>
          <Button variant="ghost" type="button" onclick={onClearToken}>Clear</Button>
        </div>
        <SelectField
          id="settings-theme"
          label="Theme"
          value={theme}
          options={themeOptions}
          onValueChange={(value) => onThemeChange(value as Theme)}
        />
      </div>
      </CardContent>
    </Card>
  </div>

  <Card>
    <CardHeader>
      <CardTitle>Runtime</CardTitle>
    </CardHeader>
    <CardContent>
    <div class="divide-y divide-border text-sm">
      <div class="grid gap-2 py-3 md:grid-cols-[10rem_minmax(0,1fr)]">
        <span class="text-muted-foreground">Request limit</span>
        <span>{formatBytes(serverInfo?.limits.max_request_bytes ?? 0)}</span>
      </div>
      <div class="grid gap-2 py-3 md:grid-cols-[10rem_minmax(0,1fr)]">
        <span class="text-muted-foreground">Timeout</span>
        <span>{serverInfo?.limits.request_timeout_seconds ?? "-"} seconds</span>
      </div>
    </div>
    </CardContent>
  </Card>
</section>
