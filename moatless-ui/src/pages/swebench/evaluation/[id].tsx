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
    const completed = evaluation.instances.filter((i: EvaluationInstance) => 
      i.status === "completed" || i.status === "error"
    ).length;
    return (completed / evaluation.instances.length) * 100;
  };

  const getInstanceDisplay = (instance: EvaluationInstance) => ({
    title: instance.instance_id,
    subtitle: (
      <div className="flex items-center gap-2">
        <Badge variant={getStatusColor(instance.status)}>
          {instance.status}
        </Badge>
        {instance.error && (
          <span className="text-destructive">{instance.error}</span>
        )}
      </div>
    ),
  });

  const handleInstanceSelect = (instance: EvaluationInstance) => {
    setSelectedInstance(instance);
    navigate(`/swebench/evaluation/${evaluationId}/${instance.instance_id}`);
  };

  if (isLoading) {
    return (
      <div className="h-full">
        <div className="flex h-14 items-center border-b px-4">
          <Skeleton className="h-8 w-64" />
        </div>
        <SplitLayout
          left={<Skeleton className="h-full" />}
          right={<Skeleton className="h-full" />}
        />
      </div>
    );
  }

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
        />
      </div>
    </div>
  );
} 