<script lang="ts">
	import { Card, CardTitle } from '$lib/components/ui/card';
	import { Input } from '$lib/components/ui/input';
	import { Button } from '$lib/components/ui/button';
	import { Maximize2, Minimize2, FolderOpen, Upload, FileText, Info } from 'lucide-svelte';
	import TrajectoryTimeline from '$lib/components/trajectory/TrajectoryTimeline.svelte';
	import { goto } from '$app/navigation';
	import { API_BASE_URL } from '$lib/config';
	import type { PageData } from './$types';

	export let data: PageData;

	let filePath = data.filePath;
	let timelineRef: TrajectoryTimeline | null = null;
	let fileInput: HTMLInputElement;
	let isLoading = false;
	let activeTab: 'path' | 'upload' = 'path';

	function loadTrajectory() {
		if (!filePath) {
			return;
		}
		// Use the current URL's pathname to maintain SPA routing
		goto(`/trajectory/?path=${encodeURIComponent(filePath)}`);
	}

	async function handleFileSelect() {
		if (fileInput.files && fileInput.files[0]) {
			const file = fileInput.files[0];
			
			if (activeTab === 'path') {
				filePath = file.name;
				loadTrajectory();
			} else {
				isLoading = true;
				try {
					const formData = new FormData();
					formData.append('file', file);
					
					const response = await fetch(`${API_BASE_URL}/trajectory/upload`, {
						method: 'POST',
						body: formData
					});
					
					if (!response.ok) {
						throw new Error('Failed to upload trajectory');
					}
					
					const trajectory = await response.json();
					data = {
						...data,
						trajectory
					};
				} catch (error) {
					console.error('Failed to upload trajectory:', error);
				} finally {
					isLoading = false;
				}
			}
		}
	}
</script>

<div class="container mx-auto space-y-6 py-6">
	<Card class="p-6">
		<div class="mb-6 flex space-x-4 border-b">
			<button
				class="px-4 py-2 -mb-px {activeTab === 'path' ? 'border-b-2 border-primary text-primary' : 'text-muted-foreground'}"
				on:click={() => (activeTab = 'path')}
			>
				<div class="flex items-center space-x-2">
					<FileText class="h-4 w-4" />
					<span>Load from Path</span>
				</div>
			</button>
			<button
				class="px-4 py-2 -mb-px {activeTab === 'upload' ? 'border-b-2 border-primary text-primary' : 'text-muted-foreground'}"
				on:click={() => (activeTab = 'upload')}
			>
				<div class="flex items-center space-x-2">
					<Upload class="h-4 w-4" />
					<span>Upload File</span>
				</div>
			</button>
		</div>

		{#if activeTab === 'path'}
			<div class="space-y-2">
				<div class="flex items-center gap-2 text-sm text-muted-foreground">
					<Info class="h-4 w-4" />
					<span>Enter the absolute path to a trajectory file on your local machine</span>
				</div>
				<div class="flex items-center gap-4">
					<Input
						type="text"
						placeholder="/absolute/path/to/trajectory.json"
						bind:value={filePath}
						class="flex-1 font-mono text-sm"
					/>
					<Button on:click={loadTrajectory}>Load Trajectory</Button>
				</div>
			</div>
		{:else}
			<div class="flex items-center gap-4">
				<input
					type="file"
					bind:this={fileInput}
					on:change={handleFileSelect}
					class="hidden"
					accept=".json,.jsonl"
				/>
				<Button variant="outline" class="flex-1" on:click={() => fileInput.click()}>
					<Upload class="mr-2 h-4 w-4" />
					{isLoading ? 'Uploading...' : 'Choose File to Upload'}
				</Button>
			</div>
		{/if}
	</Card>

	{#if data.trajectory}
		{#if data.trajectory.nodes.length > 0}
			<Card class="p-6">
				<div class="mb-6 flex items-center justify-between">
					<CardTitle>Trajectory</CardTitle>
					<button
						on:click={() => timelineRef?.toggleExpandAll()}
						class="inline-flex items-center rounded-md border border-gray-300 bg-white px-2 py-1 text-xs font-medium text-gray-700 hover:bg-gray-50 sm:px-3 sm:py-1.5 sm:text-sm"
					>
						{#if timelineRef && timelineRef.isExpandedAll()}
							<Minimize2 class="mr-1 h-3 w-3 sm:mr-2 sm:h-4 sm:w-4" />
							Collapse All
						{:else}
							<Maximize2 class="mr-1 h-3 w-3 sm:mr-2 sm:h-4 sm:w-4" />
							Expand All
						{/if}
					</button>
				</div>

				<div class="grid grid-cols-1 gap-8">
					<div class="w-full">
						<TrajectoryTimeline bind:this={timelineRef} nodes={data.trajectory.nodes} />
					</div>
				</div>
			</Card>
		{:else}
			<div class="rounded-lg border border-gray-200 bg-white p-6 text-center text-gray-600">
				No nodes available
			</div>
		{/if}
	{/if}
</div>
