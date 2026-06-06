<script lang="ts">
  import { Activity, FileAudio, Library, Settings, SlidersHorizontal } from "@lucide/svelte";
  import { Button } from "$lib/components/ui/button/index.js";
  import type { View } from "../types";

  export let activeView: View;
  export let onSelect: (view: View) => void;

  const navItems = [
    { id: "dashboard" as View, label: "Status", icon: Activity },
    { id: "catalog" as View, label: "Catalog", icon: Library },
    { id: "loaded" as View, label: "Loaded", icon: SlidersHorizontal },
    { id: "separate" as View, label: "Separate", icon: FileAudio },
    { id: "settings" as View, label: "Settings", icon: Settings },
  ];
</script>

<nav class="hidden border-r border-border bg-background px-3 py-6 lg:block">
  <div class="sticky top-20 flex flex-col gap-1">
    {#each navItems as item}
      <Button
        variant={activeView === item.id ? "secondary" : "ghost"}
        size="sm"
        class="justify-start"
        type="button"
        onclick={() => onSelect(item.id)}
      >
        <svelte:component this={item.icon} class="size-4" />
        {item.label}
      </Button>
    {/each}
  </div>
</nav>

<div class="border-b border-border bg-background p-2 lg:hidden">
  <div class="grid grid-cols-5 gap-1">
    {#each navItems as item}
      <Button
        variant={activeView === item.id ? "secondary" : "ghost"}
        size="icon-sm"
        type="button"
        onclick={() => onSelect(item.id)}
        aria-label={item.label}
      >
        <svelte:component this={item.icon} class="size-4" />
      </Button>
    {/each}
  </div>
</div>
