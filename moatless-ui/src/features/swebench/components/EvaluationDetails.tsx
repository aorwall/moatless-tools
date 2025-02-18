import { Badge } from "@/lib/components/ui/badge";
import { Button } from "@/lib/components/ui/button";
import { Progress } from "@/lib/components/ui/progress";
import { Play } from "lucide-react";
import { Slider } from "@/lib/components/ui/slider";
import { Label } from "@/lib/components/ui/label";
import { useState } from "react";
import { Evaluation } from "../api/evaluation";
import { useStartEvaluation } from "../hooks/useStartEvaluation";
import { toast } from "sonner";
import { format, formatDuration, intervalToDuration } from "date-fns";
import { EvaluationTimeline } from "./EvaluationTimeline";
import { Input } from "@/lib/components/ui/input";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/lib/components/ui/tooltip";

interface EvaluationDetailsProps {
  evaluation: Evaluation;
  formatDate: (date: string) => string;
  calculateProgress: () => number;
}

type BadgeVariant = "default" | "destructive" | "outline" | "secondary";

const getStatusColor = (status: string): BadgeVariant => {
  const statusMap: Record<string, BadgeVariant> = {
    running: "secondary",
    completed: "default",
    error: "destructive",
    pending: "outline",
    failed: "destructive",
    resolved: "default",
    evaluating: "secondary"
  };
  return statusMap[status.toLowerCase()] || "default";
};

function getDuration(start?: string, end?: string, isRunning?: boolean): string {
  if (!start) return '-';
  if (isRunning) {
    const duration = intervalToDuration({
      start: new Date(start),
      end: new Date()
    });
    return formatDuration(duration, { format: ['minutes', 'seconds'] });
  }
  if (!end) return '-';
  const duration = intervalToDuration({
    start: new Date(start),
    end: new Date(end)
  });
  return formatDuration(duration, { format: ['minutes', 'seconds'] });
}

export function EvaluationDetails({ 
  evaluation, 
  formatDate,
  calculateProgress 
}: EvaluationDetailsProps) {
  const [concurrentInstances, setConcurrentInstances] = useState(1);
  const { mutate: startEvaluation, isPending } = useStartEvaluation();

  const handleStart = () => {
    startEvaluation({
      evaluationId: evaluation.evaluation_name,
      numConcurrentInstances: concurrentInstances
    }, {
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

  const canStart = (() => {
    if (evaluation.status.toLowerCase() === 'running') return false;
    if (evaluation.status.toLowerCase() === 'completed') {
      // Allow restart if there are any error instances
      return evaluation.instances.some(instance => 
        instance.status.toLowerCase() === 'error' || 
        instance.status.toLowerCase() === 'failed'
      );
    }
    return true;
  })();

  const handleNumberChange = (value: string) => {
    const num = parseInt(value);
    if (!isNaN(num) && num >= 1 && num <= 10) {
      setConcurrentInstances(num);
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-semibold">Dataset</h2>
          <p className="text-sm text-muted-foreground">{evaluation.dataset_name}</p>
        </div>
        <div className="flex items-center gap-2">
          {canStart && (
            <>
              <div className="flex items-center gap-2">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-2">
                        <Input
                          type="number"
                          min={1}
                          max={10}
                          value={concurrentInstances}
                          onChange={(e) => handleNumberChange(e.target.value)}
                          className="w-16 h-9"
                        />
                        <span className="text-xs text-muted-foreground">workers</span>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Number of concurrent evaluation workers</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Button 
                  size="sm" 
                  onClick={handleStart} 
                  disabled={isPending}
                >
                  <Play className="h-4 w-4 mr-1" />
                  Start
                </Button>
              </div>
            </>
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
            {formatDate(evaluation.started_at)}
          </p>
        </div>
        {evaluation.completed_at && (
          <div>
            <p className="font-medium">Completed At</p>
            <p className="text-sm text-muted-foreground">
              {formatDate(evaluation.completed_at)}
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

      <div>
        <p className="font-medium mb-2">Evaluation Timeline</p>
        <EvaluationTimeline 
          evaluation={evaluation}
          getStatusColor={getStatusColor}
        />
      </div>

      <div>
        <p className="font-medium mb-2">Instance Details</p>
        <div className="rounded-md border">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="p-2 text-left font-medium">Instance ID</th>
                  <th className="p-2 text-left font-medium">Status</th>
                  <th className="p-2 text-left font-medium">Started At</th>
                  <th className="p-2 text-left font-medium">Run Duration</th>
                  <th className="p-2 text-left font-medium">Eval Duration</th>
                  <th className="p-2 text-left font-medium">Result</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.instances.map((instance) => (
                  <tr key={instance.instance_id} className="border-b">
                    <td className="p-2 font-mono text-xs">{instance.instance_id}</td>
                    <td className="p-2">
                      <Badge variant={getStatusColor(instance.status)} className="text-[10px] px-1.5 py-0">
                        {instance.status}
                      </Badge>
                    </td>
                    <td className="p-2 text-xs text-muted-foreground">
                      {instance.started_at ? format(new Date(instance.started_at), 'MMM d, HH:mm:ss') : '-'}
                    </td>
                    <td className="p-2 text-xs text-muted-foreground">
                      {getDuration(
                        instance.started_at, 
                        instance.completed_at, 
                        instance.status.toLowerCase() === 'running'
                      )}
                    </td>
                    <td className="p-2 text-xs text-muted-foreground">
                      {getDuration(instance.completed_at, instance.evaluated_at)}
                    </td>
                    <td className="p-2">
                      {instance.status === "completed" && (
                        <Badge 
                          variant={instance.resolved ? "default" : "destructive"} 
                          className="text-[10px] px-1.5 py-0"
                        >
                          {instance.resolved ? "✓" : "✗"}
                        </Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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