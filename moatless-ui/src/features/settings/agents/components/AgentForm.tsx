import { useState, useEffect } from 'react';
import { Button } from '@/lib/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { Loader2, Copy, Trash2 } from 'lucide-react';
import { AgentConfig } from '@/lib/types/agent';
import { SectionCard } from '@/lib/components/form/section-card';
import { FormField } from '@/lib/components/form/form-field';
import { Input } from '@/lib/components/ui/input';
import { Textarea } from '@/lib/components/ui/textarea';
import { ComponentSelector } from '@/lib/components/form/component-selector';
import { DynamicItemList } from '@/lib/components/form/dynamic-item-list';
import { useActionStore } from '@/lib/stores/actionStore';
import { createActionConfigFromSchema } from '@/features/settings/agents/utils/actionUtils';
import { useMemory } from '@/lib/hooks/useFlowComponents';
import { DynamicListItem, Field, TextField, NumberField, ToggleField } from '@/lib/components/form/types';
import { useDeleteAgent } from '@/lib/hooks/useAgents';
import { useNavigate } from 'react-router-dom';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from '@/lib/components/ui/alert-dialog';

interface AgentFormProps {
    agent: AgentConfig;
    onSubmit: (data: AgentConfig) => Promise<void>;
    onDuplicate?: () => void;
    isNew?: boolean;
}

export function AgentForm({ agent, onSubmit, onDuplicate, isNew = false }: AgentFormProps) {
    const { data: memory } = useMemory();
    const { actions, fetchActions, getActionByClass } = useActionStore();
    const [formValues, setFormValues] = useState<Record<string, any>>(agent);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const navigate = useNavigate();
    const deleteAgentMutation = useDeleteAgent();

    // Fetch actions on component mount
    useEffect(() => {
        fetchActions();
    }, [fetchActions]);

    // Reset form when agent changes
    useEffect(() => {
        setFormValues(agent);
    }, [agent]);

    // Handle field changes
    const handleFieldChange = (id: string, value: any) => {
        setFormValues(prev => ({
            ...prev,
            [id]: value
        }));
    };

    // Handle form submission
    const handleFormSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSubmitting(true);

        try {
            // Process form data
            const formData = { ...formValues };

            // Submit the form with processed data
            await onSubmit(formData as AgentConfig);
        } catch (error) {
            console.error('Error submitting form:', error);
        } finally {
            setIsSubmitting(false);
        }
    };

    // Function to handle agent deletion
    const handleDeleteAgent = async () => {
        try {
            await deleteAgentMutation.mutateAsync(agent.agent_id);
            navigate('/settings/agents');
        } catch (error) {
            console.error('Error deleting agent:', error);
        }
    };

    // Convert actions to DynamicListItems for the DynamicItemList component
    const actionsToListItems = (): DynamicListItem[] => {
        if (!formValues.actions || !Array.isArray(formValues.actions)) {
            return [];
        }

        // Map actions to list items
        const items = formValues.actions.map(action => {
            const actionSchema = getActionByClass(action.action_class);
            if (!actionSchema) return null;

            // Create fields for each property in the action schema
            const fields = Object.entries(actionSchema.properties || {}).map(([propName, propSchema]) => {
                if (propSchema.type === 'integer') {
                    return {
                        id: propName,
                        type: 'number' as const,
                        label: propSchema.title || propName,
                        description: propSchema.description || '',
                        defaultValue: propSchema.default
                    } as NumberField;
                } else if (propSchema.type === 'boolean') {
                    return {
                        id: propName,
                        type: 'toggle' as const,
                        label: propSchema.title || propName,
                        description: propSchema.description || '',
                        defaultValue: propSchema.default
                    } as ToggleField;
                } else {
                    return {
                        id: propName,
                        type: 'text' as const,
                        label: propSchema.title || propName,
                        description: propSchema.description || '',
                        defaultValue: propSchema.default,
                        placeholder: `Enter ${propSchema.title || propName}`
                    } as TextField;
                }
            });

            // Extract all properties except action_class for values
            const { action_class, ...actionProperties } = action;

            return {
                id: action.action_class,
                name: actionSchema.title || action.action_class,
                description: actionSchema.description || '',
                fields,
                values: actionProperties // Use the extracted properties as values
            };
        }).filter(Boolean) as DynamicListItem[];

        // Sort items by name
        return items.sort((a, b) => a.name.localeCompare(b.name));
    };

    // Create available actions for the DynamicItemList
    const getAvailableActions = () => {
        if (!actions) return [];

        // Map actions to available items
        const items = Object.entries(actions).map(([actionClass, schema]) => {
            // Create fields for each property in the action schema
            const fields = Object.entries(schema.properties || {}).map(([propName, propSchema]) => {
                if (propSchema.type === 'integer') {
                    return {
                        id: propName,
                        type: 'number' as const,
                        label: propSchema.title || propName,
                        description: propSchema.description || '',
                        defaultValue: propSchema.default
                    } as NumberField;
                } else if (propSchema.type === 'boolean') {
                    return {
                        id: propName,
                        type: 'toggle' as const,
                        label: propSchema.title || propName,
                        description: propSchema.description || '',
                        defaultValue: propSchema.default
                    } as ToggleField;
                } else {
                    return {
                        id: propName,
                        type: 'text' as const,
                        label: propSchema.title || propName,
                        description: propSchema.description || '',
                        defaultValue: propSchema.default,
                        placeholder: `Enter ${propSchema.title || propName}`
                    } as TextField;
                }
            });

            return {
                id: actionClass,
                name: schema.title || actionClass,
                description: schema.description || '',
                fields
            };
        });

        // Sort available items by name
        return items.sort((a, b) => a.name.localeCompare(b.name));
    };

    // Handle actions change from DynamicItemList
    const handleActionsChange = (id: string, items: DynamicListItem[]) => {
        // Convert DynamicListItems back to ActionConfig objects
        const actions = items.map(item => ({
            action_class: item.id,
            ...item.values // Spread values directly onto the action object instead of nesting them
        }));

        handleFieldChange('actions', actions);
    };

    // If actions are still loading, show loading spinner
    if (!actions) {
        return (
            <div className="flex h-full w-full items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Form */}
            <form onSubmit={handleFormSubmit} className="space-y-6 pb-24">
                <div className="flex items-center justify-between">
                    <h1 className="text-2xl font-semibold">{isNew ? 'New Agent' : agent.agent_id}</h1>
                    <div className="flex gap-2">
                        {!isNew && onDuplicate && (
                            <Button
                                type="button"
                                variant="outline"
                                onClick={onDuplicate}
                                className="flex items-center gap-2"
                            >
                                <Copy className="h-4 w-4" />
                                Duplicate Agent
                            </Button>
                        )}
                        {!isNew && (
                            <Button
                                type="button"
                                variant="destructive"
                                onClick={() => setDeleteDialogOpen(true)}
                                disabled={deleteAgentMutation.isPending}
                                className="gap-2"
                            >
                                {deleteAgentMutation.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                    <Trash2 className="h-4 w-4" />
                                )}
                                {deleteAgentMutation.isPending ? "Deleting..." : "Delete Agent"}
                            </Button>
                        )}
                        <Button type="submit" disabled={isSubmitting}>
                            {isSubmitting ? "Saving..." : "Save Changes"}
                        </Button>
                    </div>
                </div>

                {/* Agent ID field for new agents */}
                {isNew && (
                    <SectionCard
                        title="Basic Settings"
                        description="Configure the basic settings for your agent"
                    >
                        <FormField label="Agent ID" htmlFor="agent_id" tooltip="A unique identifier for this agent">
                            <Input
                                id="agent_id"
                                value={formValues.agent_id || ''}
                                onChange={(e) => handleFieldChange('agent_id', e.target.value)}
                                placeholder="Enter a unique identifier for the agent"
                                required
                            />
                        </FormField>
                    </SectionCard>
                )}

                {/* System Prompt Section */}
                <SectionCard
                    title="System Prompt"
                    description="Configure the system prompt for your agent"
                >
                    <FormField label="System Prompt" htmlFor="system_prompt" tooltip="The system prompt that defines the agent's behavior">
                        <Textarea
                            id="system_prompt"
                            value={formValues.system_prompt || ''}
                            onChange={(e) => handleFieldChange('system_prompt', e.target.value)}
                            placeholder="Enter the system prompt for the agent..."
                            className="min-h-[200px] font-mono text-sm"
                            rows={10}
                        />
                    </FormField>
                </SectionCard>

                {/* Memory Section */}
                <SectionCard
                    title="Memory"
                    description="Configure the memory component for your agent"
                >
                    <ComponentSelector
                        componentType="memory"
                        value={formValues.memory}
                        onChange={(componentValue) => {
                            handleFieldChange('memory', componentValue);
                        }}
                    />
                </SectionCard>

                {/* Actions Section */}
                <SectionCard
                    title="Agent Actions"
                    description="Configure the actions available to this agent"
                >
                    <DynamicItemList
                        field={{
                            id: 'actions',
                            type: 'dynamic-item-list',
                            label: 'Actions',
                            addButtonText: 'Add Action',
                            availableItems: getAvailableActions()
                        }}
                        value={actionsToListItems()}
                        onChange={handleActionsChange}
                    />
                </SectionCard>
            </form>

            {/* Delete Confirmation Dialog */}
            <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Agent</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete the agent "{agent.agent_id}"? This action cannot be undone.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={handleDeleteAgent}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
} 