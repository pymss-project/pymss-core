<script lang="ts">
  import { Activity, FileAudio, Library, Settings, SlidersHorizontal } from "@lucide/svelte";
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

<nav class="hidden border-r border-base-300 bg-base-100 px-3 py-6 lg:block">
  <div class="sticky top-20 flex flex-col gap-1">
    {#each navItems as item}
      <button
        class="btn btn-ghost btn-sm justify-start rounded-full {activeView === item.id ? 'btn-active' : ''}"
        type="button"
        on:click={() => onSelect(item.id)}
      >
        <svelte:component this={item.icon} class="size-4" />
        {item.label}
      </button>
    {/each}
  </div>
</nav>

<div class="border-b border-base-300 bg-base-100 p-2 lg:hidden">
  <div class="grid grid-cols-5 gap-1">
    {#each navItems as item}
      <button
        class="btn btn-ghost btn-sm rounded-full {activeView === item.id ? 'btn-active' : ''}"
        type="button"
        on:click={() => onSelect(item.id)}
        aria-label={item.label}
      >
        <svelte:component this={item.icon} class="size-4" />
      </button>
    {/each}
  </div>
</div>
