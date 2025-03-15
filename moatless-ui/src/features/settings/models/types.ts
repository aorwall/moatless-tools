import { z } from 'zod';
import { ModelConfigSchema, ModelConfig, ModelTestResult } from '@/lib/types/model';

// Re-export types from lib
export type { ModelConfig, ModelTestResult };
export { ModelConfigSchema };

// Form schema for the settings form
export const ModelFormSchema = z.object({
    model_id: z.string().min(1, 'Model ID is required'),
    model: z.string().min(1, 'Model name is required'),
    completion_model_class: z.string().min(1, 'Completion model class is required'),
    model_base_url: z.string().nullable().optional(),
    model_api_key: z.string().nullable().optional(),
    temperature: z.number().min(0).max(1).optional(),
    max_tokens: z.number().int().positive().optional(),
    timeout: z.number().int().positive(),
    thoughts_in_action: z.boolean(),
    disable_thoughts: z.boolean(),
    merge_same_role_messages: z.boolean(),
    message_cache: z.boolean(),
    few_shot_examples: z.boolean()
});

export type ModelFormValues = z.infer<typeof ModelFormSchema>;

// Form schema sections for the settings form
export const modelFormSections = [
    {
        id: 'basic',
        title: 'Basic Settings',
        description: 'Configure the basic settings for your model',
        fields: [
            {
                id: 'model',
                type: 'text' as const,
                label: 'Model Name',
                tooltip: 'The LiteLLM model identifier to use',
                required: true,
                placeholder: 'e.g. anthropic/claude-3-sonnet-20240229'
            },
            {
                id: 'model_base_url',
                type: 'text' as const,
                label: 'Model Base URL',
                tooltip: 'Optional base URL for the model API',
                placeholder: 'e.g. http://localhost:8000/v1'
            },
            {
                id: 'model_api_key',
                type: 'text' as const,
                label: 'Model API Key',
                tooltip: 'Optional API key for the model',
                placeholder: 'Optional API key'
            },
            {
                id: 'timeout',
                type: 'number' as const,
                label: 'Timeout (seconds)',
                tooltip: 'Request timeout in seconds',
                required: true,
                min: 1,
                step: 1
            }
        ]
    },
    {
        id: 'parameters',
        title: 'Model Parameters',
        description: 'Configure the parameters for your model',
        fields: [
            {
                id: 'temperature',
                type: 'number' as const,
                label: 'Temperature',
                tooltip: 'Randomness in model output',
                min: 0,
                max: 1,
                step: 0.1
            },
            {
                id: 'max_tokens',
                type: 'number' as const,
                label: 'Max Tokens',
                tooltip: 'Maximum tokens to generate',
                min: 1,
                step: 1
            }
        ]
    },
    {
        id: 'features',
        title: 'Features',
        description: 'Configure the features for your model',
        fields: [
            {
                id: 'thoughts_in_action',
                type: 'toggle' as const,
                label: 'Thoughts in Action',
                description: 'Include thought generation in action steps'
            },
            {
                id: 'disable_thoughts',
                type: 'toggle' as const,
                label: 'Disable Thoughts',
                description: 'Disable thought generation completely'
            },
            {
                id: 'merge_same_role_messages',
                type: 'toggle' as const,
                label: 'Merge Same Role Messages',
                description: 'Combine consecutive messages from the same role'
            },
            {
                id: 'message_cache',
                type: 'toggle' as const,
                label: 'Message Cache',
                description: 'Enable caching of message responses'
            },
            {
                id: 'few_shot_examples',
                type: 'toggle' as const,
                label: 'Few Shot Examples',
                description: 'Include few-shot examples in prompts'
            }
        ]
    }
]; 