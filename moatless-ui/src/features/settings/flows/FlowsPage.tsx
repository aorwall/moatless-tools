import { useNavigate, useParams } from "react-router-dom";
import { useFlows } from "@/lib/hooks/useFlows";
import { Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/lib/components/ui/card";
import { SettingsHeader } from "@/features/settings/components/SettingsHeader";

export function FlowsPage() {
    const navigate = useNavigate();
    const { id } = useParams();
    const { data: flows = [], isLoading, error } = useFlows();

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
                    <AlertTitle>Error Loading Flows</AlertTitle>
                    <AlertDescription>
                        {error instanceof Error ? error.message : "Failed to load flows"}
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    if (id) {
        // If there's an ID in the URL, the FlowDetailPage will be rendered by the router
        return null;
    }

    return (
        <div className="space-y-6">
            <SettingsHeader
                title="Flows"
                description="Manage your workflow configurations"
                addButtonPath="/settings/flows/new"
                addButtonLabel="Add Flow"
            />

            {flows.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {flows.map((flow) => (
                        <Card
                            key={flow.id}
                            className="cursor-pointer hover:bg-accent/50 transition-colors"
                            onClick={() => navigate(`/settings/flows/${encodeURIComponent(flow.id)}`)}
                        >
                            <CardHeader className="pb-2">
                                <CardTitle>{flow.id}</CardTitle>
                                <CardDescription>{flow.description || "No description"}</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-muted-foreground">Type:</span>
                                        <span>{flow.flow_type}</span>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center p-12 text-center border rounded-lg bg-background">
                    <h3 className="text-lg font-medium mb-2">No flows configured</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                        Add a flow to get started with custom workflow configurations
                    </p>
                </div>
            )}
        </div>
    );
} 