import { useEffect, useState } from 'react';
import { Button } from '@/lib/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { Loader2, Copy } from 'lucide-react';
import { FlowConfigSchema, type FlowConfig } from '../types';
import { SectionCard } from '@/lib/components/form/section-card';
import { FormField } from '@/lib/components/form/form-field';
import { Input } from '@/lib/components/ui/input';
import { Textarea } from '@/lib/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/lib/components/ui/select';
import { ComponentSelector } from '@/lib/components/form/component-selector';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { useAgents } from '@/lib/hooks/useAgents';

interface FlowFormProps {
    flow: FlowConfig;
    onSubmit: (data: FlowConfig) => Promise<void>;
    onDuplicate?: () => void;
    isNew?: boolean;
}

export function FlowForm({ flow, onSubmit, onDuplicate, isNew = false }: FlowFormProps) {
    // Fetch the list of agents
    const { data: agents, isLoading: isLoadingAgents } = useAgents();

    const form = useForm<FlowConfig>({
        resolver: zodResolver(FlowConfigSchema),
        defaultValues: flow,
    });

    // Reset form when flow changes
    useEffect(() => {
        form.reset(flow);
    }, [form, flow]);


    const [formValues, setFormValues] = useState<Record<string, any>>(flow);

    // Handle field changes
    const handleFieldChange = (id: string, value: any) => {
        setFormValues(prev => ({
            ...prev,
            [id]: value
        }));
    };

    // Handle form submission
    const handleFormSubmit = () => {
        // Process form data
        const formData = { ...formValues };

        // Submit the form with processed data
        onSubmit(formData as FlowConfig);
    };

    // Custom action buttons for the form
    const actionButtons = onDuplicate ? (
        <Button
            type="button"
            variant="outline"
            onClick={onDuplicate}
            className="flex items-center gap-2"
        >
            <Copy className="h-4 w-4" />
            Duplicate Flow
        </Button>
    ) : null;

    // If form is loading, show loading spinner
    if (form.formState.isValidating && !form.formState.isDirty) {
        return (
            <div className="flex h-full w-full items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Form */}
            <form onSubmit={(e) => { e.preventDefault(); handleFormSubmit(); }} className="space-y-6 pb-24">
                <div className="flex items-center justify-between">
                    <h1 className="text-2xl font-semibold">{isNew ? 'New Flow' : flow.id}</h1>
                    <div className="flex gap-2">
                        {actionButtons}
                        <Button type="submit" disabled={form.formState.isSubmitting}>
                            {form.formState.isSubmitting ? "Saving..." : "Save Changes"}
                        </Button>
                    </div>
                </div>

                {/* Basic Settings Section */}
                <SectionCard
                    title="Basic Settings"
                    description="Configure the basic settings for your flow"
                >
                    <div className="space-y-4">
                        <FormField label="Description" htmlFor="description" tooltip="A brief description of what this flow does">
                            <Textarea
                                id="description"
                                value={formValues.description || ''}
                                onChange={(e) => handleFieldChange('description', e.target.value)}
                                placeholder="Enter a description for this flow"
                                rows={3}
                            />
                        </FormField>

                        <FormField label="Flow Type" htmlFor="flow_type" tooltip="The type of flow execution strategy">
                            <Select
                                value={formValues.flow_type || 'tree'}
                                onValueChange={(value) => handleFieldChange('flow_type', value)}
                            >
                                <SelectTrigger id="flow_type">
                                    <SelectValue placeholder="Select flow type" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="tree">Tree</SelectItem>
                                    <SelectItem value="loop">Loop</SelectItem>
                                </SelectContent>
                            </Select>
                        </FormField>

                        <FormField label="Agent" htmlFor="agent_id" tooltip="The agent to use for this flow">
                            <Select
                                value={formValues.agent_id || ''}
                                onValueChange={(value) => handleFieldChange('agent_id', value)}
                                disabled={isLoadingAgents}
                            >
                                <SelectTrigger id="agent_id">
                                    <SelectValue placeholder="Select an agent" />
                                </SelectTrigger>
                                <SelectContent>
                                    {agents?.map((agent) => (
                                        <SelectItem key={agent.agent_id} value={agent.agent_id}>
                                            {agent.agent_id}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </FormField>
                    </div>
                </SectionCard>

                {/* Execution Limits Section */}
                <SectionCard
                    title="Execution Limits"
                    description="Configure the execution limits for your flow"
                >
                    <div className="space-y-4">
                        <FormField label="Max Iterations" htmlFor="max_iterations" tooltip="Maximum number of iterations">
                            <Input
                                id="max_iterations"
                                type="number"
                                value={formValues.max_iterations || 10}
                                onChange={(e) => handleFieldChange('max_iterations', Number(e.target.value))}
                                min={1}
                            />
                        </FormField>

                        <FormField label="Max Cost" htmlFor="max_cost" tooltip="Maximum cost in USD">
                            <Input
                                id="max_cost"
                                type="number"
                                value={formValues.max_cost || 1.0}
                                onChange={(e) => handleFieldChange('max_cost', Number(e.target.value))}
                                min={0}
                                step={0.1}
                            />
                        </FormField>
                    </div>
                </SectionCard>

                {/* Tree Settings Section (only show if flow_type is 'tree') */}
                {formValues.flow_type === 'tree' && (
                    <>
                        <SectionCard
                            title="Selector Component"
                            description="Configure the component that selects which nodes to expand"
                        >
                            <ComponentSelector
                                componentType="selectors"
                                value={formValues.selector}
                                onChange={(componentValue) => {
                                    handleFieldChange('selector', componentValue);
                                }}
                            />
                        </SectionCard>

                        <SectionCard
                            title="Value Function Component"
                            description="Configure the component that evaluates node quality"
                        >
                            <ComponentSelector
                                componentType="value-functions"
                                value={formValues.value_function}
                                onChange={(componentValue) => {
                                    handleFieldChange('value_function', componentValue);
                                }}
                            />
                        </SectionCard>

                        <SectionCard
                            title="Feedback Generator Component"
                            description="Configure the component that generates feedback for nodes"
                        >
                            <ComponentSelector
                                componentType="feedback-generators"
                                value={formValues.feedback_generator}
                                onChange={(componentValue) => {
                                    handleFieldChange('feedback_generator', componentValue);
                                }}
                            />
                        </SectionCard>

                        <SectionCard
                            title="Tree Limits"
                            description="Configure the limits for tree exploration"
                        >
                            <div className="space-y-4">
                                <FormField label="Max Expansions" htmlFor="max_expansions" tooltip="Maximum number of node expansions">
                                    <Input
                                        id="max_expansions"
                                        type="number"
                                        value={formValues.max_expansions || 10}
                                        onChange={(e) => handleFieldChange('max_expansions', Number(e.target.value))}
                                        min={1}
                                    />
                                </FormField>

                                <FormField label="Max Depth" htmlFor="max_depth" tooltip="Maximum tree depth">
                                    <Input
                                        id="max_depth"
                                        type="number"
                                        value={formValues.max_depth || 5}
                                        onChange={(e) => handleFieldChange('max_depth', Number(e.target.value))}
                                        min={1}
                                    />
                                </FormField>
                            </div>
                        </SectionCard>
                    </>
                )}

            </form>
        </div>
    );
} 