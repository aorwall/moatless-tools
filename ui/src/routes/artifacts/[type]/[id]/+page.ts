import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';
import type { Artifact, FileArtifact } from '$lib/types/artifact';
import { API_BASE_URL } from '$lib/config';

export const load = (async ({ params, fetch }) => {
	try {
		const url = new URL(`${API_BASE_URL}/artifacts/${params.type}/${params.id}`);
		console.log('Fetching artifact from:', url);
		const artifactResponse = await fetch(url);

		if (!artifactResponse.ok) {
			throw error(artifactResponse.status, 'Failed to load artifact');
		}

		const artifact: Artifact = await artifactResponse.json();

		// Fetch referenced artifacts if they exist
		const referencedArtifacts = await Promise.all(
			(artifact as FileArtifact).references?.map(async (ref) => {
				const refUrl = new URL(`${API_BASE_URL}/artifacts/${ref.type}/${ref.id}`);
				const refResponse = await fetch(refUrl);
				if (!refResponse.ok) return null;
				return await refResponse.json();
			}) || []
		).then((results) => results.filter(Boolean));

		return {
			artifact,
			referencedArtifacts
		};
	} catch (e) {
		console.error('Failed to load artifact:', e);
		throw error(500, 'Failed to load artifact');
	}
}) satisfies PageLoad;
