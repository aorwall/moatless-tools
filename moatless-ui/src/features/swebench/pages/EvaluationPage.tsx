import { Button } from "@/lib/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/lib/components/ui/collapsible";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import debounce from "lodash/debounce";
import { useWebSocketStore } from "@/lib/stores/websocketStore";
import { Evaluation } from "../api/evaluation";
import { EvaluationInstancesTable } from "../components/EvaluationInstancesTable";
import { EvaluationStatus } from "../components/EvaluationStatus";
import { EvaluationTimeline } from "../components/EvaluationTimeline";
import { EvaluationToolbar } from "../components/EvaluationToolbar";
import { useRealtimeEvaluation } from "../hooks/useEvaluation";

const formatDate = (date: string) => {
  return new Date(date).toLocaleString();
};

// Component that renders the evaluation details
function EvaluationContent({ evaluation }: { evaluation: Evaluation }) {
  const [timelineExpanded, setTimelineExpanded] = useState(false);

  const canStart = (() => {
    if (evaluation.status.toLowerCase() === "running") return false;
    if (evaluation.status.toLowerCase() === "completed") {
      // Allow restart if there are any error instances
      return evaluation.instances.some(
        (instance) =>
          instance.status.toLowerCase() === "error" ||
          instance.status.toLowerCase() === "failed",
      );
    }
    return true;
  })();

  return (
    <div className="p-6 space-y-6">
      {/* Header Section */}
      <div className="border-b pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">
              {evaluation.evaluation_name}
            </h1>
            <p className="text-xs text-muted-foreground mt-1">
              Created {formatDate(evaluation.created_at)}
            </p>
          </div>
          <EvaluationToolbar
            evaluation={evaluation}
            canStart={canStart}
          />
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-3 gap-4">
        {/* Model Info */}
        <div className="rounded-lg border p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium">Model</h3>
            <span className="text-xs text-muted-foreground">
              ID: {evaluation.model.id}
            </span>
          </div>
          <div className="space-y-1 text-sm">
            <p>
              <span className="text-muted-foreground">Name:</span>{" "}
              {evaluation.model.model}
            </p>
            <p>
              <span className="text-muted-foreground">Response Format:</span>{" "}
              {evaluation.model.response_format}
            </p>
            <p>
              <span className="text-muted-foreground">Temperature:</span>{" "}
              {evaluation.model.temperature || "N/A"}
            </p>
          </div>
        </div>

        {/* Flow Info */}
        <div className="rounded-lg border p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium">Flow</h3>
            <span className="text-xs text-muted-foreground">
              ID: {evaluation.flow.id}
            </span>
          </div>
          <div className="space-y-1 text-sm">
            <p>
              <span className="text-muted-foreground">Type:</span>{" "}
              {evaluation.flow.flow_type}
            </p>
            <p>
              <span className="text-muted-foreground">Max Cost:</span> $
              {evaluation.flow.max_cost}
            </p>
            <p>
              <span className="text-muted-foreground">Max Iterations:</span>{" "}
              {evaluation.flow.max_iterations}
            </p>
          </div>
        </div>

        {/* Dataset Info */}
        <div className="rounded-lg border p-4">
          <h3 className="font-medium mb-2">Dataset</h3>
          <div className="space-y-1 text-sm">
            <p>
              <span className="text-muted-foreground">Name:</span>{" "}
              {evaluation.dataset_name}
            </p>
            <p>
              <span className="text-muted-foreground">Instances:</span>{" "}
              {evaluation.instances.length}
            </p>
            <p>
              <span className="text-muted-foreground">Workers:</span>{" "}
              {evaluation.num_workers}
            </p>
          </div>
        </div>
      </div>

      {/* Status Section */}
      <EvaluationStatus evaluation={evaluation} />

      <Collapsible
        open={timelineExpanded}
        onOpenChange={setTimelineExpanded}
        className="border rounded-md"
      >
        <div className="flex items-center justify-between px-4 py-2 border-b">
          <p className="font-medium">Evaluation Timeline</p>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
              {timelineExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
              <span className="sr-only">Toggle timeline</span>
            </Button>
          </CollapsibleTrigger>
        </div>
        <CollapsibleContent className="p-4">
          <EvaluationTimeline evaluation={evaluation} />
        </CollapsibleContent>
      </Collapsible>

      <div>
        <p className="font-medium mb-2">Instance Details</p>
        <EvaluationInstancesTable evaluation={evaluation} />
      </div>
    </div>
  );
}

// Main page component that fetches data and handles errors
export function EvaluationPage() {
  const { evaluationId } = useParams();
  const navigate = useNavigate();

  // Use the real-time evaluation hook
  const {
    data: evaluation,
    isLoading,
    error,
    connectionState
  } = useRealtimeEvaluation(evaluationId!);

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-destructive">
            Failed to load evaluation
          </h2>
          <p className="mt-2 text-muted-foreground">
            {error instanceof Error
              ? error.message
              : "An unexpected error occurred"}
          </p>
          <Button
            variant="outline"
            className="mt-4"
            onClick={() => navigate("/swebench/evaluation")}
          >
            Back to Evaluations
          </Button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold">Loading evaluation...</h2>
        </div>
      </div>
    );
  }

  if (!evaluation) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold">Evaluation not found</h2>
          <Button
            variant="outline"
            className="mt-4"
            onClick={() => navigate("/swebench/evaluation")}
          >
            Back to Evaluations
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 min-h-0 overflow-auto">
        <EvaluationContent evaluation={evaluation} />
      </div>
    </div>
  );
}
