<script lang="ts">
	import type { LayoutData } from './$types';
	import ArtifactExplorer from '$lib/components/artifacts/ArtifactExplorer.svelte';
	import type { Artifact } from '$lib/types/artifact';
	import { goto } from '$app/navigation';

	export let data: LayoutData;
	let selectedArtifact: Artifact | null = null;

	$: if (selectedArtifact) {
		goto(`/artifacts/${selectedArtifact.type}/${selectedArtifact.id}`);
	}
</script>

<div class="flex h-screen flex-col overflow-hidden">
	<div class="flex-none border-b px-6 py-4">
		<h1 class="text-2xl font-bold">Artifact Explorer</h1>
	</div>

	<div class="flex min-h-0 flex-1">
		<!-- Left Panel: Explorer -->
		<div class="w-80 flex-none overflow-hidden border-r">
			<ArtifactExplorer
				artifacts={data.artifacts}
				artifactTypes={data.artifactTypes}
				bind:selectedArtifact
			/>
		</div>

		<!-- Right Panel: Detail -->
		<div class="flex-1 overflow-y-auto">
			<slot />
		</div>
	</div>
</div>
