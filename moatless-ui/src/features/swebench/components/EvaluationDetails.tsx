import { Badge } from "@/lib/components/ui/badge";
import { Button } from "@/lib/components/ui/button";
import { Progress } from "@/lib/components/ui/progress";
import { Play } from "lucide-react";
import { Evaluation } from "../api/evaluation";
import { useStartEvaluation } from "../hooks/useStartEvaluation";
import { toast } from "sonner";

interface EvaluationDetailsProps {
  evaluation: Evaluation;
  getStatusColor: (status: string) => string;
  formatDate: (date: string) => string;
  calculateProgress: () => number;
}

export function EvaluationDetails({ 
  evaluation, 
  getStatusColor, 
  formatDate,
  calculateProgress 
}: EvaluationDetailsProps) {
  const { mutate: startEvaluation, isPending } = useStartEvaluation();

  const handleStart = () => {
    startEvaluation(evaluation.evaluation_name, {
      onSuccess: () => {
        toast.success("Evaluation started successfully");
      },
      onError: (error) => {
        toast.error("Failed to start evaluation", {
          description: error instanceof Error ? error.message : "Unknown error",
        });
      },
    });
  };

  const canStart = !["running", "completed"].includes(evaluation.status.toLowerCase());

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-semibold">Dataset</h2>
          <p className="text-sm text-muted-foreground">{evaluation.dataset_name}</p>
        </div>
        <div className="flex items-center gap-2">
          {canStart && (
            <Button 
              size="sm" 
              onClick={handleStart} 
              disabled={isPending}
            >
              <Play className="h-4 w-4 mr-1" />
              Start
            </Button>
          )}
          <Badge variant={getStatusColor(evaluation.status)}>
            {evaluation.status}
          </Badge>
        </div>
      </div>

      <div>
        <p className="font-medium mb-2">Progress</p>
        <div className="space-y-2">
          <Progress value={calculateProgress()} />
          <div className="flex justify-between text-sm text-muted-foreground">
            <span>Total Instances: {evaluation.instances.length}</span>
            <span>{Math.round(calculateProgress())}% Complete</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="font-medium">Started At</p>
          <p className="text-sm text-muted-foreground">
            {formatDate(evaluation.start_time)}
          </p>
        </div>
        {evaluation.finish_time && (
          <div>
            <p className="font-medium">Completed At</p>
            <p className="text-sm text-muted-foreground">
              {formatDate(evaluation.finish_time)}
            </p>
          </div>
        )}
      </div>

      <div>
        <p className="font-medium mb-2">Instance Status Summary</p>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(getStatusCounts(evaluation.instances)).map(([status, count]) => (
            <div key={status} className="flex items-center justify-between p-2 rounded-md border">
              <Badge variant={getStatusColor(status)}>
                {status}
              </Badge>
              <span className="text-sm font-medium">{count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function getStatusCounts(instances: Evaluation['instances']) {
  return instances.reduce((acc, instance) => {
    const status = instance.status.toLowerCase();
    acc[status] = (acc[status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
} 