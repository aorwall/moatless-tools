import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// Use static adapter for FastAPI serving
		adapter: adapter({
			pages: 'dist',
			assets: 'dist',
			fallback: 'index.html', // SPA fallback
			strict: false
		}),
		alias: {
			'@/*': './src/lib/*'
		}
	}
};

export default config;
