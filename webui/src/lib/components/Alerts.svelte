<script lang="ts">
  import { AlertTriangle, CheckCircle2 } from "@lucide/svelte";
  import { Alert, AlertAction, AlertDescription } from "$lib/components/ui/alert/index.js";
  import { Button } from "$lib/components/ui/button/index.js";

  export let errorMessage: string;
  export let noticeMessage: string;
  export let onDismissError: () => void;
  export let onDismissNotice: () => void;
</script>

{#if errorMessage}
  <Alert variant="destructive" class="items-center">
    <AlertTriangle class="size-5" />
    <AlertDescription>{errorMessage}</AlertDescription>
    <AlertAction>
      <Button variant="ghost" size="xs" type="button" onclick={onDismissError}>Dismiss</Button>
    </AlertAction>
  </Alert>
{/if}

{#if noticeMessage}
  <Alert class="items-center">
    <CheckCircle2 class="size-5" />
    <AlertDescription>{noticeMessage}</AlertDescription>
    <AlertAction>
      <Button variant="ghost" size="xs" type="button" onclick={onDismissNotice}>Dismiss</Button>
    </AlertAction>
  </Alert>
{/if}
