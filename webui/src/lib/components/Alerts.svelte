<script lang="ts">
  import { AlertTriangle, CheckCircle2 } from "@lucide/svelte";

  export let errorMessage: string;
  export let noticeMessage: string;
  export let onDismissError: () => void;
  export let onDismissNotice: () => void;
</script>

{#if errorMessage}
  <div class="alert border border-error bg-base-100 text-error">
    <AlertTriangle class="size-5" />
    <span>{errorMessage}</span>
    <button class="btn btn-ghost btn-xs rounded-full" type="button" on:click={onDismissError}>Dismiss</button>
  </div>
{/if}

{#if noticeMessage}
  <div class="alert border border-base-300 bg-base-100">
    <CheckCircle2 class="size-5" />
    <span>{noticeMessage}</span>
    <button class="btn btn-ghost btn-xs rounded-full" type="button" on:click={onDismissNotice}>Dismiss</button>
  </div>
{/if}
