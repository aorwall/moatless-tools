import { useNavigate, useParams } from "react-router-dom";
import { useAgents } from "@/lib/hooks/useAgents";
import { Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/lib/components/ui/card";
import { SettingsHeader } from "@/features/settings/components/SettingsHeader";

export function AgentsPage() {
    const navigate = useNavigate();
    const { id } = useParams();
    const { data: agents = [], isLoading, error } = useAgents();

    if (isLoading) {
        return (
            <div className="flex h-full w-full items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex h-full w-full items-center justify-center p-4">
                <Alert variant="destructive" className="max-w-md">
                    <AlertTitle>Error Loading Agents</AlertTitle>
                    <AlertDescription>
                        {error instanceof Error ? error.message : "Failed to load agents"}
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    if (id) {
        // If there's an ID in the URL, the AgentDetailPage will be rendered by the router
        return null;
    }

    return (
        <div className="space-y-6">
            <SettingsHeader
                title="Agents"
                description="Manage your AI agents and configurations"
                addButtonPath="/settings/agents/new"
                addButtonLabel="Add Agent"
            />

            {agents.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {agents.map((agent) => (
                        <Card
                            key={agent.agent_id}
                            className="cursor-pointer hover:bg-accent/50 transition-colors"
                            onClick={() => navigate(`/settings/agents/${encodeURIComponent(agent.agent_id)}`)}
                        >
                            <CardHeader className="pb-2">
                                <CardTitle>{agent.agent_id}</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-muted-foreground">Model:</span>
                                        <span>{agent.model_id || "Not set"}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-muted-foreground">System Prompt:</span>
                                        <span>{agent.system_prompt ? "Set" : "Not set"}</span>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center p-12 text-center border rounded-lg bg-background">
                    <h3 className="text-lg font-medium mb-2">No agents configured</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                        Add an agent to get started with custom AI configurations
                    </p>
                </div>
            )}
        </div>
    );
} 