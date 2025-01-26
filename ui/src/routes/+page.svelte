<script lang="ts">
	import type { PageData } from './$types';
	import ArtifactExplorer from '$lib/components/artifacts/ArtifactExplorer.svelte';
	import ArtifactDetail from '$lib/components/artifacts/ArtifactDetail.svelte';
	import type { Artifact } from '$lib/types/artifact';

	export let data: PageData;
	let selectedArtifact: Artifact | null = null;
</script>

<div class="container mx-auto p-4">
	<h1 class="mb-4 text-2xl font-bold">Artifact Explorer</h1>

	<div class="grid grid-cols-1 gap-4 lg:grid-cols-4">
		<!-- Left Panel: Explorer -->
		<div class="lg:col-span-1">
			<ArtifactExplorer
				artifacts={data.artifacts}
				artifactTypes={data.artifactTypes}
				bind:selectedArtifact
			/>
		</div>

		<!-- Right Panel: Detail -->
		<div class="lg:col-span-3">
			<ArtifactDetail artifact={selectedArtifact} />
		</div>
	</div>
</div>
