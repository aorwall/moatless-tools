import prettier from 'eslint-config-prettier';
import js from '@eslint/js';
import { includeIgnoreFile } from '@eslint/compat';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import { fileURLToPath } from 'node:url';
import ts from 'typescript-eslint';
import unusedImports from 'eslint-plugin-unused-imports';

const gitignorePath = fileURLToPath(new URL('./.gitignore', import.meta.url));

export default [
	includeIgnoreFile(gitignorePath),
	js.configs.recommended,
	...ts.configs.recommended,
	...svelte.configs['flat/recommended'],
	{
		plugins: {
			'unused-imports': unusedImports
		},
		rules: {
			'@typescript-eslint/no-unused-vars': 'off',
			'unused-imports/no-unused-imports': 'error',
			'unused-imports/no-unused-vars': [
				'warn',
				{
					vars: 'all',
					varsIgnorePattern: '^_|\\$\\$|^page$',
					args: 'after-used',
					argsIgnorePattern: '^_'
				}
			],
			'@typescript-eslint/no-explicit-any': 'off',
			'svelte/valid-compile': [
				'error',
				{
					ignoreWarnings: true
				}
			],
			'svelte/no-unused-svelte-ignore': 'off',
			'svelte/a11y-click-events-have-key-events': 'off',
			'svelte/a11y-missing-attribute': 'off',
			'svelte/a11y-no-static-element-interactions': 'off'
		}
	},
	prettier,
	...svelte.configs['flat/prettier'],
	{
		languageOptions: {
			globals: {
				...globals.browser,
				...globals.node
			}
		}
	},
	{
		files: ['**/*.svelte'],
		languageOptions: {
			parser: svelte.parser,
			parserOptions: {
				parser: ts.parser
			}
		}
	}
];
