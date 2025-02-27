import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/lib/components/ui/button";
import { Badge } from "@/lib/components/ui/badge";
import { Skeleton } from "@/lib/components/ui/skeleton";
import { useEvaluation } from "@/features/swebench/hooks/useEvaluation";
import { EvaluationInstance } from "@/features/swebench/api/evaluation";
import { SplitLayout } from "@/lib/components/layouts/SplitLayout";
import { DataExplorer } from "@/lib/components/DataExplorer";
import { useState, useRef, useEffect } from "react";
import { useWebSocketStore } from "@/lib/stores/websocketStore";
import { useQueryClient } from "@tanstack/react-query";
import debounce from "lodash/debounce";
import { EvaluationPage } from "@/features/swebench/pages/EvaluationPage";

type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

export function EvaluationDetailsPage() {
  const { evaluationId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { subscribe } = useWebSocketStore();
  const lastUpdateTime = useRef<number>(0);
  const { data: evaluation, isLoading, error } = useEvaluation(evaluationId!);

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
        <EvaluationPage
          evaluation={evaluation}
        />
      </div>
    </div>
  );
} 