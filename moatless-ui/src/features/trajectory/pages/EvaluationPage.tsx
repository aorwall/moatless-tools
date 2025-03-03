import { useEvaluation } from "@/features/swebench/hooks/useEvaluation";
import { EvaluationPage } from "@/features/swebench/pages/EvaluationPage";
import { Button } from "@/lib/components/ui/button";
import { useWebSocketStore } from "@/lib/stores/websocketStore";
import { useQueryClient } from "@tanstack/react-query";
import debounce from "lodash/debounce";
import { useEffect, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";

export function EvaluationDetailsPage() {
  const { evaluationId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { subscribe } = useWebSocketStore();
  const lastUpdateTime = useRef<number>(0);
  const { data: evaluation, isLoading, error } = useEvaluation(evaluationId!);

  // Create debounced refresh function
  const debouncedRefresh = debounce(
    () => {
      const now = Date.now();
      // Only refresh if more than 1 second has passed since last update
      if (now - lastUpdateTime.current >= 1000) {
        queryClient.invalidateQueries({
          queryKey: ["evaluation", evaluationId],
        });
        lastUpdateTime.current = now;
      }
    },
    1000,
    { maxWait: 1000 },
  ); // Maximum wait of 1 second between updates

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
        <EvaluationPage evaluation={evaluation} />
      </div>
    </div>
  );
}
