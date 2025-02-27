import { Badge } from "@/lib/components/ui/badge";
import { Button } from "@/lib/components/ui/button";
import { Play, Copy, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import { Evaluation } from "../api/evaluation";
import { useStartEvaluation } from "../hooks/useStartEvaluation";
import { useCloneEvaluation } from "../hooks/useCloneEvaluation";
import { useProcessEvaluationResults } from "../hooks/useProcessEvaluationResults";
import { toast } from "sonner";
import { format, formatDuration, intervalToDuration } from "date-fns";
import { EvaluationTimeline } from "../components/EvaluationTimeline";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/lib/components/ui/tooltip";
import { EvaluationStatus } from "../components/EvaluationStatus";
import { Link, useNavigate } from "react-router-dom";
import { EvaluationToolbar } from "../components/EvaluationToolbar";
import { EvaluationInstancesTable } from "../components/EvaluationInstancesTable";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/lib/components/ui/collapsible";

interface EvaluationPageProps {
  evaluation: Evaluation;
}

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

const formatDate = (date: string) => {
  return new Date(date).toLocaleString();
};

export function EvaluationPage({ 
  evaluation
}: EvaluationPageProps) {
  const navigate = useNavigate();
  const [concurrentInstances, setConcurrentInstances] = useState(1);
  const [timelineExpanded, setTimelineExpanded] = useState(false);
  const { mutate: startEvaluation, isPending: isStartPending } = useStartEvaluation();
  const { mutate: cloneEvaluation, isPending: isClonePending } = useCloneEvaluation();
  const { mutate: processResults, isPending: isProcessPending } = useProcessEvaluationResults();

  const handleStart = () => {
    startEvaluation({
      evaluationId: evaluation.evaluation_name,
      numConcurrentInstances: concurrentInstances
    }, {
      onSuccess: () => {
        toast.success('Evaluation started successfully');
      },
      onError: (error) => {
        toast.error(`Failed to start evaluation: ${error.message}`);
      }
    });
  };

  const handleClone = () => {
    cloneEvaluation(evaluation.evaluation_name, {
      onSuccess: (data) => {
        toast.success('Evaluation cloned successfully');
        navigate(`/swebench/evaluations/${data.evaluation_name}`);
      },
      onError: (error) => {
        toast.error(`Failed to clone evaluation: ${error.message}`);
      }
    });
  };

  const handleProcessResults = () => {
    processResults(evaluation.evaluation_name, {
      onSuccess: () => {
        toast.success('Evaluation results synchronized successfully');
      },
      onError: (error) => {
        toast.error(`Failed to synchronize evaluation results: ${error.message}`);
      }
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

  return (
    <div className="p-6 space-y-6">
      {/* Header Section */}
      <div className="border-b pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">{evaluation.evaluation_name}</h1>
            <p className="text-xs text-muted-foreground mt-1">
              Created {formatDate(evaluation.created_at)}
            </p>
          </div>
          <EvaluationToolbar
            evaluation={evaluation}
            onClone={handleClone}
            onStart={handleStart}
            onProcess={handleProcessResults}
            isStartPending={isStartPending}
            isClonePending={isClonePending}
            isProcessPending={isProcessPending}
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
            <span className="text-xs text-muted-foreground">ID: {evaluation.model.id}</span>
          </div>
          <div className="space-y-1 text-sm">
            <p><span className="text-muted-foreground">Name:</span> {evaluation.model.model}</p>
            <p><span className="text-muted-foreground">Response Format:</span> {evaluation.model.response_format}</p>
            <p><span className="text-muted-foreground">Temperature:</span> {evaluation.model.temperature || 'N/A'}</p>
          </div>
        </div>

        {/* Flow Info */}
        <div className="rounded-lg border p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium">Flow</h3>
            <span className="text-xs text-muted-foreground">ID: {evaluation.flow.id}</span>
          </div>
          <div className="space-y-1 text-sm">
            <p><span className="text-muted-foreground">Type:</span> {evaluation.flow.flow_type}</p>
            <p><span className="text-muted-foreground">Max Cost:</span> ${evaluation.flow.max_cost}</p>
            <p><span className="text-muted-foreground">Max Iterations:</span> {evaluation.flow.max_iterations}</p>
          </div>
        </div>

        {/* Dataset Info */}
        <div className="rounded-lg border p-4">
          <h3 className="font-medium mb-2">Dataset</h3>
          <div className="space-y-1 text-sm">
            <p><span className="text-muted-foreground">Name:</span> {evaluation.dataset_name}</p>
            <p><span className="text-muted-foreground">Instances:</span> {evaluation.instances.length}</p>
            <p><span className="text-muted-foreground">Workers:</span> {evaluation.num_workers}</p>
          </div>
        </div>
      </div>

      {/* Status Section */}
      <EvaluationStatus
        evaluation={evaluation}
      />

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
          <EvaluationTimeline 
            evaluation={evaluation}
          />
        </CollapsibleContent>
      </Collapsible>

      <div>
        <p className="font-medium mb-2">Instance Details</p>
        <EvaluationInstancesTable evaluation={evaluation} />
      </div>
    </div>
  );
}
