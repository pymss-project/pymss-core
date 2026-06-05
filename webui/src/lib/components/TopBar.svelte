<script lang="ts">
  import { KeyRound, Loader2, Moon, RefreshCw, Sun } from "@lucide/svelte";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import type { HealthResponse } from "../types";

  export let health: HealthResponse | null;
  export let busyAction: string | null;
  export let theme: "light" | "dark";
  export let onRefresh: () => void;
  export let onToggleTheme: () => void;
  export let onOpenToken: () => void;

  $: online = health?.status === "ok";
</script>

<header class="flex min-h-14 items-center border-b border-border bg-background px-4">
  <div class="min-w-0 flex-1">
    <div class="flex items-baseline gap-3">
      <div class="text-[18px] font-medium leading-none tracking-normal">pymss</div>
      <span class="hidden text-sm text-muted-foreground sm:inline">server</span>
    </div>
  </div>

  <div class="hidden flex-1 justify-center md:flex">
    <Badge variant="outline" class="h-7 gap-2 px-3 font-mono font-normal">
      <span class="size-1.5 rounded-full {online ? 'bg-primary' : 'bg-muted-foreground/40'}"></span>
      {online ? "server ok" : "server offline"}
    </Badge>
  </div>

  <div class="flex flex-1 justify-end gap-2">
    <Button
      variant="ghost"
      size="icon-sm"
      type="button"
      onclick={onRefresh}
      aria-label="Refresh"
    >
      {#if busyAction === "refresh"}
        <Loader2 class="size-4 animate-spin" />
      {:else}
        <RefreshCw class="size-4" />
      {/if}
    </Button>
    <Button
      variant="ghost"
      size="icon-sm"
      type="button"
      onclick={onToggleTheme}
      aria-label="Toggle theme"
    >
      {#if theme === "dark"}
        <Sun class="size-4" />
      {:else}
        <Moon class="size-4" />
      {/if}
    </Button>
    <Button variant="outline" size="sm" type="button" onclick={onOpenToken}>
      <KeyRound class="size-4" />
      <span class="hidden sm:inline">Token</span>
    </Button>
  </div>
</header>
