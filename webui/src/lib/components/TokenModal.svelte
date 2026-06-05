<script lang="ts">
  export let show: boolean;
  export let tokenInput: string;
  export let onSave: () => void;
  export let onClose: () => void;
</script>

{#if show}
  <div class="modal modal-open">
    <div class="modal-box rounded-box border border-base-300 bg-base-100 shadow-none">
      <h3 class="text-xl font-medium">API token</h3>
      <form class="mt-4 space-y-3" on:submit|preventDefault={onSave}>
        <label class="field">
          <span class="field-label">Bearer token</span>
          <input class="text-field" bind:value={tokenInput} type="password" />
        </label>
        <div class="modal-action">
          <button class="btn btn-ghost rounded-full" type="button" on:click={onClose}>Cancel</button>
          <button class="btn btn-primary rounded-full" type="submit">Save</button>
        </div>
      </form>
    </div>
    <button class="modal-backdrop" type="button" on:click={onClose}>Close</button>
  </div>
{/if}
