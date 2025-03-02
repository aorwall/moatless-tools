import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useDeleteAgent } from "@/lib/hooks/useAgents";
import { Button } from "@/lib/components/ui/button";
import { Textarea } from "@/lib/components/ui/textarea";
import { toast } from "sonner";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/lib/components/ui/tabs";
import { ActionSelector } from "../[id]/components/ActionSelector";
import { SelectedActions } from "../[id]/components/SelectedActions";
import { ActionConfig, AgentConfig } from "@/lib/types/agent";
import { useActionStore } from "@/lib/stores/actionStore";
import { createActionConfigFromSchema } from "../[id]/utils/actionUtils";
import { Loader2 } from "lucide-react";

interface AgentDetailProps {
  agent: AgentConfig;
  onSubmit: (data: AgentConfig) => Promise<void>;
}

export function AgentDetail({ agent, onSubmit }: AgentDetailProps) {
  const navigate = useNavigate();
  const deleteAgent = useDeleteAgent();
  const [selectedActions, setSelectedActions] = useState<ActionConfig[]>([]);
  const { getActionByTitle } = useActionStore();
  const [systemPrompt, setSystemPrompt] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (agent) {
      setSelectedActions(agent.actions || []);
      setSystemPrompt(agent.system_prompt || "");
    }
  }, [agent]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    try {
      setIsSaving(true);
      setError(null);

      const agentData = {
        ...agent,
        system_prompt: systemPrompt,
        actions: selectedActions,
      };

      await onSubmit(agentData);
    } catch (e) {
      const errorMessage =
        e instanceof Error
          ? e.message
          : (e as any)?.response?.data?.detail ||
            "An unexpected error occurred";

      console.error(errorMessage);
      setError(errorMessage);
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm("Are you sure you want to delete this agent?")) return;

    try {
      await deleteAgent.mutateAsync(agent.id);
      toast.success("Agent deleted successfully");
      navigate("/settings/agents");
    } catch (error) {
      console.error("Error deleting agent:", error);
      toast.error("Failed to delete agent");
    }
  };

  const handlePropertyChange = (
    className: string,
    property: string,
    value: any,
  ) => {
    const newActions = selectedActions.map((action) => {
      if (action.title === className) {
        return {
          ...action,
          properties: {
            ...action.properties,
            [property]: value,
          },
        };
      }
      return action;
    });
    setSelectedActions(newActions);
  };

  const handleActionRemove = (actionClassName: string) => {
    const newActions = selectedActions.filter(
      (a) => a.title !== actionClassName,
    );
    setSelectedActions(newActions);
  };

  const handleActionAdd = (actionClassName: string) => {
    const actionSchema = getActionByTitle(actionClassName);
    if (!actionSchema) return;

    const newAction = createActionConfigFromSchema(actionSchema);
    const newActions = [...selectedActions, newAction];
    setSelectedActions(newActions);
  };

  if (!agent) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="flex-none border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Agent: {agent.id}</h1>
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="destructive"
              onClick={handleDelete}
              disabled={deleteAgent.isPending}
            >
              {deleteAgent.isPending ? "Deleting..." : "Delete Agent"}
            </Button>
            <Button type="submit" form="agent-form" disabled={isSaving}>
              {isSaving ? "Saving..." : "Save Changes"}
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <form
          id="agent-form"
          onSubmit={handleSubmit}
          className="h-full flex flex-col"
        >
          <Tabs defaultValue="actions" className="flex-1 flex flex-col min-h-0">
            <div className="flex-none border-b bg-muted">
              <TabsList>
                <TabsTrigger value="actions">Actions</TabsTrigger>
                <TabsTrigger value="prompt">System Prompt</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="actions" className="flex-1 p-6 min-h-0">
              <div className="flex gap-6 h-full">
                <div className="flex-1 min-h-0 h-full">
                  <SelectedActions
                    actions={selectedActions}
                    onRemove={handleActionRemove}
                    onPropertyChange={handlePropertyChange}
                  />
                </div>
                <div className="flex-1 min-h-0 h-full">
                  <ActionSelector
                    selectedActions={selectedActions.map((a) => a.title)}
                    onSelect={handleActionAdd}
                  />
                </div>
              </div>
            </TabsContent>

            <TabsContent
              value="prompt"
              className="flex-1 p-6 min-h-0 overflow-hidden"
            >
              <div className="flex flex-col h-full min-h-0">
                <div className="flex-none space-y-2 mb-2">
                  <h2 className="text-lg font-semibold">System Prompt</h2>
                  <p className="text-sm text-muted-foreground">
                    The complete system prompt that defines the agent's
                    behavior, role, and guidelines.
                  </p>
                </div>
                <div className="flex-1 min-h-0">
                  <Textarea
                    id="system_prompt"
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    placeholder="Enter the system prompt for the agent..."
                    className="h-full w-full font-mono text-sm leading-relaxed resize-none !min-h-full"
                  />
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </form>
      </div>
    </div>
  );
}
