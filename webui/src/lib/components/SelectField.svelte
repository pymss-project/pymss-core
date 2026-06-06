<script lang="ts">
  import { Label } from "$lib/components/ui/label/index.js";
  import * as Select from "$lib/components/ui/select/index.js";

  export interface SelectOption<T extends string = string> {
    value: T;
    label: string;
  }

  export let id: string;
  export let label: string;
  export let value: string;
  export let options: SelectOption[];
  export let disabled = false;
  export let placeholder = "Select";
  export let className = "";
  export let onValueChange: (value: string) => void = () => {};

  $: selectedLabel = options.find((option) => option.value === value)?.label ?? placeholder;
</script>

<div class={`grid gap-1.5 ${className}`}>
  <Label for={id}>{label}</Label>
  <Select.Root type="single" bind:value disabled={disabled} onValueChange={(next) => onValueChange(next)}>
    <Select.Trigger {id} class="h-9 w-full text-sm">
      <span data-slot="select-value">{selectedLabel}</span>
    </Select.Trigger>
    <Select.Content>
      {#each options as option}
        <Select.Item value={option.value} label={option.label}>{option.label}</Select.Item>
      {/each}
    </Select.Content>
  </Select.Root>
</div>
