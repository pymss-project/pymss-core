<script lang="ts">
  import { KeyRound, Loader2, Moon, RefreshCw, Sun } from "@lucide/svelte";
  import type { HealthResponse } from "../types";

  export let health: HealthResponse | null;
  export let busyAction: string | null;
  export let theme: "light" | "dark";
  export let onRefresh: () => void;
  export let onToggleTheme: () => void;
  export let onOpenToken: () => void;

  $: online = health?.status === "ok";
</script>

<header class="navbar min-h-14 border-b border-base-300 bg-base-100 px-4">
  <div class="navbar-start min-w-0">
    <div class="flex items-baseline gap-3">
      <div class="text-[18px] font-medium leading-none tracking-normal">pymss</div>
      <span class="hidden text-sm text-base-content/50 sm:inline">server</span>
    </div>
  </div>

  <div class="navbar-center hidden md:flex">
    <span class="badge badge-outline gap-2 rounded-full px-3 py-3 font-mono text-xs font-normal">
      <span class="size-1.5 rounded-full {online ? 'bg-primary' : 'bg-base-content/30'}"></span>
      {online ? "server ok" : "server offline"}
    </span>
  </div>

  <div class="navbar-end gap-2">
    <button
      class="btn btn-ghost btn-circle btn-sm"
      type="button"
      on:click={onRefresh}
      aria-label="Refresh"
    >
      {#if busyAction === "refresh"}
        <Loader2 class="size-4 animate-spin" />
      {:else}
        <RefreshCw class="size-4" />
      {/if}
    </button>
    <button
      class="btn btn-ghost btn-circle btn-sm"
      type="button"
      on:click={onToggleTheme}
      aria-label="Toggle theme"
    >
      {#if theme === "dark"}
        <Sun class="size-4" />
      {:else}
        <Moon class="size-4" />
      {/if}
    </button>
    <button class="btn btn-outline btn-sm rounded-full" type="button" on:click={onOpenToken}>
      <KeyRound class="size-4" />
      <span class="hidden sm:inline">Token</span>
    </button>
  </div>
</header>
