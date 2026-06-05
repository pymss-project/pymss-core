<script lang="ts">
  import { onMount } from "svelte";
  import { AppController } from "./lib/app-controller.svelte";
  import MainPanel from "./lib/components/MainPanel.svelte";
  import Navigation from "./lib/components/Navigation.svelte";
  import TokenModal from "./lib/components/TokenModal.svelte";
  import TopBar from "./lib/components/TopBar.svelte";

  const app = new AppController();

  onMount(() => app.mount());
</script>

<svelte:head>
  <title>pymss WebUI</title>
</svelte:head>

<main class="min-h-screen bg-base-100 text-base-content">
  <TopBar
    health={app.health}
    busyAction={app.busyAction}
    theme={app.theme}
    onRefresh={() => void app.refreshAll()}
    onToggleTheme={() => app.setTheme(app.theme === "dark" ? "light" : "dark")}
    onOpenToken={() => (app.showTokenModal = true)}
  />

  <div class="grid min-h-[calc(100vh-3.5rem)] grid-cols-1 lg:grid-cols-[12rem_minmax(0,1fr)]">
    <Navigation activeView={app.activeView} onSelect={(view) => (app.activeView = view)} />

    <MainPanel
      state={app.panelState}
      actions={app.panelActions}
      bind:catalogQuery={app.catalogQuery}
      bind:loadModelName={app.loadModelName}
      bind:loadSourceOverride={app.loadSourceOverride}
      bind:loadEndpointOverride={app.loadEndpointOverride}
      bind:inferenceParamsText={app.inferenceParamsText}
      bind:audioFile={app.audioFile}
      bind:responseFormat={app.responseFormat}
      bind:outputAudioFormat={app.outputAudioFormat}
      bind:theme={app.theme}
      bind:downloadSourceForm={app.downloadSourceForm}
      bind:downloadEndpointForm={app.downloadEndpointForm}
    />
  </div>

  <TokenModal
    bind:show={app.showTokenModal}
    bind:tokenInput={app.tokenInput}
    onSave={() => app.saveToken()}
    onClose={() => (app.showTokenModal = false)}
  />
</main>
