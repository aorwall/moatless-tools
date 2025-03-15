import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormDescription,
  FormMessage,
} from "@/lib/components/ui/form";
import { Input } from "@/lib/components/ui/input";
import { Button } from "@/lib/components/ui/button";
import { Textarea } from "@/lib/components/ui/textarea";
import { FlowConfigSchema, type FlowConfig } from "@/lib/types/flow";
import { useState, useEffect } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { useAgents, useAgent } from "@/lib/hooks/useAgents";
import { Loader2 } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/lib/components/ui/collapsible";
import { ChevronDown, ChevronUp } from "lucide-react";
import { Badge } from "@/lib/components/ui/badge";
import { Card } from "@/lib/components/ui/card";
import {
  useSelectors,
  useValueFunctions,
  useFeedbackGenerators,
  useArtifactHandlers,
} from "@/lib/hooks/useFlowComponents";
import { ComponentSchema, ComponentProperty } from "@/lib/types/flow";
import { ComponentSelect } from "@/features/settings/components/ComponentSelect";
import { ArtifactHandlersSelect } from "./ArtifactHandlersSelect";

interface FlowDetailProps {
  flow: FlowConfig;
  onSubmit: (data: FlowConfig) => Promise<void>;
  isNew?: boolean;
}

// Add type for component fields
type ComponentField = "selector" | "value_function" | "feedback_generator";

type ComponentsResponse = {
  components: Record<string, ComponentSchema>;
};

export function FlowDetail({ flow, onSubmit, isNew }: FlowDetailProps) {
  const form = useForm<FlowConfig>({
    resolver: zodResolver(FlowConfigSchema),
    defaultValues: flow,
  });

  useEffect(() => {
    form.reset(flow);
  }, [flow, form]);

  const flowType = form.watch("flow_type");
  const { data: agents, isLoading: isLoadingAgents } = useAgents();
  const selectedAgentId = form.watch("agent_id");
  const { data: selectedAgent } = useAgent(selectedAgentId || "");
  const [isAgentInfoOpen, setIsAgentInfoOpen] = useState(false);

  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data: selectors } = useSelectors();
  const { data: valueFunctions } = useValueFunctions();
  const { data: feedbackGenerators } = useFeedbackGenerators();
  const { data: artifactHandlers } = useArtifactHandlers();

  const handleSubmit = async (data: FlowConfig) => {
    try {
      setIsSaving(true);
      setError(null);

      await onSubmit(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "An error occurred");
      throw e;
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
        {/* First row: ID and Flow Type */}
        <div className="grid grid-cols-2 gap-4">
          <FormField
            control={form.control}
            name="id"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Flow ID</FormLabel>
                <FormControl>
                  <Input
                    {...field}
                    disabled={!isNew}
                    placeholder={
                      isNew ? "Enter a unique identifier" : undefined
                    }
                  />
                </FormControl>
                <FormDescription>
                  {isNew
                    ? "Enter a unique identifier for this flow"
                    : "The unique identifier for this flow"}
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="flow_type"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Flow Type</FormLabel>
                <Select onValueChange={field.onChange} value={field.value}>
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select flow type" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value="tree">Tree</SelectItem>
                    <SelectItem value="loop">Loop</SelectItem>
                  </SelectContent>
                </Select>
                <FormDescription>
                  The type of flow execution strategy
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <FormField
          control={form.control}
          name="description"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Description</FormLabel>
              <FormControl>
                <Textarea
                  {...field}
                  placeholder="Enter a description for this flow"
                />
              </FormControl>
              <FormDescription>
                A brief description of what this flow does
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        {/* Numeric inputs in a 2x2 grid */}
        <div className="grid grid-cols-2 gap-4">
          <FormField
            control={form.control}
            name="max_iterations"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Max Iterations</FormLabel>
                <FormControl>
                  <Input
                    {...field}
                    type="number"
                    min={1}
                    onChange={(e) => field.onChange(Number(e.target.value))}
                  />
                </FormControl>
                <FormDescription>Maximum number of iterations</FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="max_cost"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Max Cost ($)</FormLabel>
                <FormControl>
                  <Input
                    {...field}
                    type="number"
                    step="0.1"
                    min={0}
                    value={field.value || ""}
                    onChange={(e) => {
                      const value = e.target.value;
                      const cleanValue = value ? Number(value).toString() : "";
                      field.onChange(
                        cleanValue ? Number(cleanValue) : undefined,
                      );
                    }}
                  />
                </FormControl>
                <FormDescription>
                  Maximum cost allowed for the flow in USD
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        {/* Agent selection remains full width */}
        <FormField
          control={form.control}
          name="agent_id"
          render={({ field }) => (
            <FormItem className="space-y-1">
              <FormLabel>Agent</FormLabel>
              <Select onValueChange={field.onChange} value={field.value || ""}>
                <FormControl>
                  <SelectTrigger>
                    {isLoadingAgents ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <SelectValue placeholder="Select an agent" />
                    )}
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  {agents?.map((agent) => (
                    <SelectItem key={agent.id} value={agent.id}>
                      {agent.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <FormDescription>
                Select the agent to use for this flow
              </FormDescription>
              {selectedAgent && (
                <Collapsible
                  open={isAgentInfoOpen}
                  onOpenChange={setIsAgentInfoOpen}
                  className="mt-2"
                >
                  <CollapsibleTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex w-full items-center justify-between p-2 font-normal"
                    >
                      Agent Details
                      {isAgentInfoOpen ? (
                        <ChevronUp className="h-4 w-4" />
                      ) : (
                        <ChevronDown className="h-4 w-4" />
                      )}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <Card className="mt-2 p-4">
                      <div className="space-y-4">
                        <div>
                          <div className="text-sm font-medium">Model</div>
                          <div className="text-sm text-muted-foreground">
                            {selectedAgent.model_id}
                          </div>
                        </div>
                        <div>
                          <div className="text-sm font-medium">
                            Response Format
                          </div>
                          <Badge variant="secondary">
                            {selectedAgent.response_format}
                          </Badge>
                        </div>
                        {selectedAgent.system_prompt && (
                          <div>
                            <div className="text-sm font-medium">
                              System Prompt
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {selectedAgent.system_prompt}
                            </div>
                          </div>
                        )}
                        {selectedAgent.actions.length > 0 && (
                          <div>
                            <div className="text-sm font-medium">Actions</div>
                            <div className="mt-1 flex flex-wrap gap-1">
                              {selectedAgent.actions.map((action) => (
                                <Badge key={action.title} variant="outline">
                                  {action.title}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </Card>
                  </CollapsibleContent>
                </Collapsible>
              )}
              <FormMessage />
            </FormItem>
          )}
        />

        {/* Artifact Handlers */}
        <ArtifactHandlersSelect
          control={form.control}
          componentsResponse={artifactHandlers}
          label="Artifact Handlers"
          description="Components that handle artifacts for this flow"
        />

        {/* Tree-specific fields in a 2x2 grid */}
        {flowType === "tree" && (
          <>
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="max_expansions"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Max Expansions</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        type="number"
                        min={1}
                        onChange={(e) => field.onChange(Number(e.target.value))}
                      />
                    </FormControl>
                    <FormDescription>
                      Maximum number of expansions per iteration
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="max_depth"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Max Depth</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        type="number"
                        min={1}
                        onChange={(e) => field.onChange(Number(e.target.value))}
                      />
                    </FormControl>
                    <FormDescription>
                      Maximum depth of the flow tree
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="min_finished_nodes"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Min Finished Nodes</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        type="number"
                        min={0}
                        onChange={(e) =>
                          field.onChange(
                            e.target.value ? Number(e.target.value) : undefined,
                          )
                        }
                      />
                    </FormControl>
                    <FormDescription>
                      Minimum number of finished nodes required
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="max_finished_nodes"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Max Finished Nodes</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        type="number"
                        min={0}
                        onChange={(e) =>
                          field.onChange(
                            e.target.value ? Number(e.target.value) : undefined,
                          )
                        }
                      />
                    </FormControl>
                    <FormDescription>
                      Maximum number of finished nodes allowed
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <FormField
              control={form.control}
              name="reward_threshold"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Reward Threshold</FormLabel>
                  <FormControl>
                    <Input
                      {...field}
                      type="number"
                      step="0.1"
                      onChange={(e) =>
                        field.onChange(
                          e.target.value ? Number(e.target.value) : undefined,
                        )
                      }
                    />
                  </FormControl>
                  <FormDescription>
                    Minimum reward threshold for accepting nodes
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <ComponentSelect
              name="selector"
              control={form.control}
              componentsResponse={selectors}
              label="Selector"
              description="Component that selects which nodes to expand"
            />
            <ComponentSelect
              name="value_function"
              control={form.control}
              componentsResponse={valueFunctions}
              label="Value Function"
              description="Component that evaluates node quality"
            />
            <ComponentSelect
              name="feedback_generator"
              control={form.control}
              componentsResponse={feedbackGenerators}
              label="Feedback Generator"
              description="Component that generates feedback for nodes"
            />
          </>
        )}

        <div className="mt-8 flex items-center justify-end gap-4">
          {error && <p className="text-sm text-red-500">{error}</p>}
          <Button type="submit" disabled={isSaving}>
            {isSaving
              ? isNew
                ? "Creating..."
                : "Saving..."
              : isNew
                ? "Create Flow"
                : "Save Changes"}
          </Button>
        </div>
      </form>
    </Form>
  );
}
