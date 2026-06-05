<script lang="ts">
  import { Button } from "$lib/components/ui/button/index.js";
  import * as Dialog from "$lib/components/ui/dialog/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Label } from "$lib/components/ui/label/index.js";

  export let show: boolean;
  export let tokenInput: string;
  export let onSave: () => void;
  export let onClose: () => void;
</script>

<Dialog.Root bind:open={show} onOpenChange={(open) => !open && onClose()}>
  <Dialog.Content>
    <Dialog.Header>
      <Dialog.Title>API token</Dialog.Title>
      <Dialog.Description class="sr-only">Bearer token</Dialog.Description>
    </Dialog.Header>
    <form class="space-y-4" on:submit|preventDefault={onSave}>
      <div class="grid gap-1.5">
        <Label for="token-input">Bearer token</Label>
        <Input id="token-input" class="h-9 text-sm" bind:value={tokenInput} type="password" />
      </div>
      <Dialog.Footer>
        <Button variant="ghost" type="button" onclick={onClose}>Cancel</Button>
        <Button type="submit">Save</Button>
      </Dialog.Footer>
    </form>
  </Dialog.Content>
</Dialog.Root>
