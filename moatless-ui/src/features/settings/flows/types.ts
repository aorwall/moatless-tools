import { z } from 'zod';
import { FlowConfigSchema, FlowConfig, ComponentSchema } from '@/lib/types/flow';

// Re-export types from lib
export type { FlowConfig, ComponentSchema };
export { FlowConfigSchema };

// Form schema sections for the settings form
export const flowFormSections = [
    {
        id: 'basic',
        title: 'Basic Settings',
        description: 'Configure the basic settings for your flow',
        fields: [
            {
                id: 'description',
                type: 'textarea' as const,
                label: 'Description',
                tooltip: 'A brief description of what this flow does',
                placeholder: 'Enter a description for this flow',
                rows: 3
            },
            {
                id: 'flow_type',
                type: 'select' as const,
                label: 'Flow Type',
                tooltip: 'The type of flow execution strategy',
                options: [
                    { value: 'tree', label: 'Tree' },
                    { value: 'loop', label: 'Loop' }
                ]
            },
            {
                id: 'agent_id',
                type: 'component-select' as const,
                label: 'Agent',
                tooltip: 'Select the agent to use for this flow',
                options: [], // This will be populated dynamically
                conditionalFields: {}
            }
        ]
    },
    {
        id: 'limits',
        title: 'Execution Limits',
        description: 'Configure the execution limits for your flow',
        fields: [
            {
                id: 'max_iterations',
                type: 'number' as const,
                label: 'Max Iterations',
                tooltip: 'Maximum number of iterations',
                min: 1,
                step: 1
            },
            {
                id: 'max_cost',
                type: 'number' as const,
                label: 'Max Cost ($)',
                tooltip: 'Maximum cost allowed for the flow in USD',
                min: 0,
                step: 0.1
            }
        ]
    },
    {
        id: 'tree_settings',
        title: 'Tree Settings',
        description: 'Configure the tree-specific settings for your flow',
        fields: [
            {
                id: 'max_expansions',
                type: 'number' as const,
                label: 'Max Expansions',
                tooltip: 'Maximum number of expansions per iteration',
                min: 1,
                step: 1
            },
            {
                id: 'max_depth',
                type: 'number' as const,
                label: 'Max Depth',
                tooltip: 'Maximum depth of the flow tree',
                min: 1,
                step: 1
            },
            {
                id: 'min_finished_nodes',
                type: 'number' as const,
                label: 'Min Finished Nodes',
                tooltip: 'Minimum number of finished nodes required',
                min: 0,
                step: 1
            },
            {
                id: 'max_finished_nodes',
                type: 'number' as const,
                label: 'Max Finished Nodes',
                tooltip: 'Maximum number of finished nodes allowed',
                min: 0,
                step: 1
            },
            {
                id: 'reward_threshold',
                type: 'number' as const,
                label: 'Reward Threshold',
                tooltip: 'Minimum reward threshold for accepting nodes',
                step: 0.1
            },
            {
                id: 'selector',
                type: 'component-select' as const,
                label: 'Selector',
                tooltip: 'Component that selects which nodes to expand',
                options: [], // This will be populated dynamically
                conditionalFields: {}
            },
            {
                id: 'value_function',
                type: 'component-select' as const,
                label: 'Value Function',
                tooltip: 'Component that evaluates node quality',
                options: [], // This will be populated dynamically
                conditionalFields: {}
            },
            {
                id: 'feedback_generator',
                type: 'component-select' as const,
                label: 'Feedback Generator',
                tooltip: 'Component that generates feedback for nodes',
                options: [], // This will be populated dynamically
                conditionalFields: {}
            }
        ]
    },
    {
        id: 'artifact_handlers',
        title: 'Artifact Handlers',
        description: 'Configure the artifact handlers for your flow',
        fields: [
            {
                id: 'artifact_handlers',
                type: 'dynamic-item-list' as const,
                label: 'Artifact Handlers',
                tooltip: 'Components that handle artifacts for this flow',
                availableItems: [], // This will be populated dynamically
                addButtonText: 'Add Artifact Handler'
            }
        ]
    }
]; 