import type { LayoutLoad } from './$types';
import type { Artifact, ArtifactData } from '$lib/types/artifact';
import { API_BASE_URL } from '$lib/config';

export const load = (async ({ fetch }) => {
	try {
		const response = await fetch(`${API_BASE_URL}/artifacts`);
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}
		const artifacts: Artifact[] = await response.json();
		const artifactTypes = Array.from(new Set(artifacts.map((a) => a.type)));

		return {
			artifacts,
			artifactTypes
		} satisfies ArtifactData;
	} catch (error) {
		console.error('Failed to load artifacts:', error);
		return {
			artifacts: [],
			artifactTypes: []
		} satisfies ArtifactData;
	}
}) satisfies LayoutLoad;
