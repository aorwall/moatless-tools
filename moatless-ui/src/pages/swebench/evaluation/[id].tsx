import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/lib/components/ui/button";
import { Badge } from "@/lib/components/ui/badge";
import { Skeleton } from "@/lib/components/ui/skeleton";
import { useEvaluation } from "@/features/swebench/hooks/useEvaluation";
import { EvaluationInstance } from "@/features/swebench/api/evaluation";
import { SplitLayout } from "@/lib/components/layouts/SplitLayout";
import { DataExplorer } from "@/lib/components/DataExplorer";
import { useState, useRef, useEffect } from "react";
import { EvaluationOverview } from "@/features/swebench/components/EvaluationOverview";
import { InstanceDetails } from "@/features/swebench/components/InstanceDetails";
import { EvaluationDetails } from "@/features/swebench/components/EvaluationDetails";
import { useWebSocketStore } from "@/lib/stores/websocketStore";
import { useQueryClient } from "@tanstack/react-query";
import debounce from "lodash/debounce";

type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

export function EvaluationDetailsPage() {
  const { evaluationId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { subscribe } = useWebSocketStore();
  const lastUpdateTime = useRef<number>(0);
  const { data: evaluation, isLoading, error } = useEvaluation(evaluationId!);
  const [selectedInstance, setSelectedInstance] = useState<EvaluationInstance | null>(null);

  // Create debounced refresh function
  const debouncedRefresh = debounce(() => {
    const now = Date.now();
    // Only refresh if more than 1 second has passed since last update
    if (now - lastUpdateTime.current >= 1000) {
      queryClient.invalidateQueries({ queryKey: ["evaluation", evaluationId] });
      lastUpdateTime.current = now;
    }
  }, 1000, { maxWait: 1000 }); // Maximum wait of 1 second between updates

  useEffect(() => {
    if (!evaluationId) return;

    // Subscribe to project events for this evaluation
    const unsubscribe = subscribe(`project.${evaluationId}`, (message) => {
      console.log("Received evaluation update:", message);
      debouncedRefresh();
    });

    return () => {
      unsubscribe();
      debouncedRefresh.cancel();
    };
  }, [evaluationId, subscribe, queryClient]);

  const getStatusColor = (status: string): BadgeVariant => {
    switch (status.toLowerCase()) {
      case "running": return "default";
      case "completed": return "secondary";
      case "error": return "destructive";
      case "pending": return "outline";
      default: return "secondary";
    }
  };

  const formatDate = (date: string) => {
    return new Date(date).toLocaleString();
  };

  const calculateProgress = () => {
    if (!evaluation) return 0;
    const terminalStates = ["resolved", "failed", "evaluated", "error"];
    const completed = evaluation.instances.filter((i: EvaluationInstance) => 
      terminalStates.includes(i.status.toLowerCase())
    ).length;
    return (completed / evaluation.instances.length) * 100;
  };

  const getResolutionStats = () => {
    if (!evaluation) return { resolved: 0, total: 0, percentage: 0 };
    const resolved = evaluation.instances.filter(i => i.status.toLowerCase() === "resolved").length;
    const terminalStates = ["resolved", "failed", "evaluated", "error"];
    const total = evaluation.instances.filter(i => terminalStates.includes(i.status.toLowerCase())).length;
    return {
      resolved,
      total,
      percentage: total > 0 ? (resolved / total) * 100 : 0
    };
  };

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-destructive">Failed to load evaluation</h2>
          <p className="mt-2 text-muted-foreground">
            {error instanceof Error ? error.message : 'An unexpected error occurred'}
          </p>
          <Button variant="outline" className="mt-4" onClick={() => navigate("/swebench/evaluation")}>
            Back to Evaluations
          </Button>
        </div>
      </div>
    );
  }

  if (!evaluation) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold">Evaluation not found</h2>
          <Button variant="outline" className="mt-4" onClick={() => navigate("/swebench/evaluation")}>
            Back to Evaluations
          </Button>
        </div>
      </div>
    );
  }


  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 min-h-0 overflow-auto">
        <EvaluationDetails
          evaluation={evaluation}
          getStatusColor={getStatusColor}
          formatDate={formatDate}
          calculateProgress={calculateProgress}
          getResolutionStats={getResolutionStats}
        />
      </div>
    </div>
  );
} 