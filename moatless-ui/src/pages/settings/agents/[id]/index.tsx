import { useParams } from "react-router-dom";
import { useAgent } from "@/lib/hooks/useAgents";
import { Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { AgentForm } from "../components/AgentForm";

export function AgentDetailPage() {
  const { id } = useParams();
  const { data: agent, isLoading, error } = useAgent(id ?? "");

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Alert variant="destructive" className="max-w-md">
          <AlertTitle>Error Loading Agent</AlertTitle>
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load agent"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!agent) {
    return (
      <div className="flex h-full items-center justify-center">
        <Alert className="max-w-md">
          <AlertTitle>Agent Not Found</AlertTitle>
          <AlertDescription>
            The requested agent could not be found.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="h-full min-h-0">
      <AgentForm agent={agent} />
    </div>
  );
}
