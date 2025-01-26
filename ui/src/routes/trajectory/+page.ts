import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';
import { API_BASE_URL } from '$lib/config';
import type { TrajectoryDTO } from '$lib/types/trajectory';

export const load = (async ({ url, fetch }) => {
	const filePath = url.searchParams.get('path') || '';

	if (!filePath) {
		return {
			filePath: ''
		};
	}

	try {
		const trajectoryUrl = new URL(`${API_BASE_URL}/trajectory`);
		trajectoryUrl.searchParams.set('file_path', filePath);

		const response = await fetch(trajectoryUrl);
		if (!response.ok) {
			throw error(response.status, 'Failed to load trajectory');
		}

		const trajectory: TrajectoryDTO = await response.json();
		return {
			filePath,
			trajectory
		};
	} catch (e) {
		console.error('Failed to load trajectory:', e);
		throw error(500, 'Failed to load trajectory');
	}
}) satisfies PageLoad;
