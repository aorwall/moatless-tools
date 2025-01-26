<script lang="ts">
	import type { PageData } from './$types';
	import type { Artifact } from '$lib/types/artifact';
	import ArtifactDetail from '$lib/components/artifacts/ArtifactDetail.svelte';

	export let data: PageData;
	$: artifact = data.artifact as Artifact;
	$: referencedArtifacts = data.referencedArtifacts || [];
</script>

<div class="flex h-full min-h-0 flex-col">
	<div class="flex-none border-b px-6 py-4">
		<div class="flex items-baseline gap-3">
			<h1 class="text-2xl font-bold">{artifact.name || artifact.id}</h1>
			<div class="text-sm text-gray-500">
				<span class="rounded-full bg-gray-100 px-2 py-1">{artifact.type}</span>
			</div>
		</div>
		<div class="mt-1 font-mono text-sm text-gray-500">{artifact.id}</div>
	</div>

	<div class="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[2fr,1fr]">
		<div class="overflow-y-auto border-r">
			<div class="p-6">
				<ArtifactDetail {artifact} />
			</div>
		</div>

		{#if referencedArtifacts.length > 0}
			<div class="overflow-y-auto">
				<div class="p-6">
					<h2 class="mb-4 text-xl font-semibold">Referenced Artifacts</h2>
					<div class="space-y-4">
						{#each referencedArtifacts as refArtifact}
							<ArtifactDetail artifact={refArtifact} isReference={true} />
						{/each}
					</div>
				</div>
			</div>
		{/if}
	</div>
</div>
