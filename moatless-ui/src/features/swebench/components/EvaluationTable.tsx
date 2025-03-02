import React from "react";
import { useNavigate } from "react-router-dom";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/lib/components/ui/table";
import { Badge } from "@/lib/components/ui/badge";
import { Skeleton } from "@/lib/components/ui/skeleton";
import { Progress } from "@/lib/components/ui/progress";
import { Coins, Cpu, MessageSquare, Zap, ClipboardList } from "lucide-react";
import { formatNumber } from "@/lib/utils/format";
import { EvaluationListItem } from "@/features/swebench/api/evaluation";

type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

interface EvaluationTableProps {
  evaluations?: EvaluationListItem[];
  isLoading: boolean;
}

export function EvaluationTable({
  evaluations,
  isLoading,
}: EvaluationTableProps) {
  const navigate = useNavigate();

  const getStatusColor = (status: string): BadgeVariant => {
    switch (status.toLowerCase()) {
      case "running":
        return "default";
      case "completed":
        return "secondary";
      case "error":
        return "destructive";
      case "pending":
        return "outline";
      default:
        return "secondary";
    }
  };

  const formatDate = (date: string) => {
    return new Date(date).toLocaleString();
  };

  const getProgressPercentage = (evaluation: EvaluationListItem) => {
    if (!evaluation.status_summary) return 0;
    const total = evaluation.instance_count;
    const completed =
      evaluation.status_summary.error +
      evaluation.status_summary.resolved +
      evaluation.status_summary.completed;
    return (completed / total) * 100;
  };

  const getResolvedPercentage = (evaluation: EvaluationListItem) => {
    if (!evaluation.status_summary) return 0;
    const total = evaluation.instance_count;
    const resolved = evaluation.status_summary.resolved;
    return (resolved / total) * 100;
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-16 w-full" />
        ))}
      </div>
    );
  }

  if (!evaluations?.length) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <ClipboardList className="h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-lg font-semibold">No evaluations yet</h3>
        <p className="text-sm text-muted-foreground mt-2">
          Start a new evaluation to see your results here
        </p>
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-[250px]">Evaluation</TableHead>
          <TableHead className="w-[250px]">Status & Progress</TableHead>
          <TableHead>Started</TableHead>
          <TableHead>Finished</TableHead>
          <TableHead>Metrics</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {evaluations?.map((evaluation) => (
          <TableRow
            key={evaluation.evaluation_name}
            className="cursor-pointer"
            onClick={() => navigate(evaluation.evaluation_name)}
          >
            <TableCell>
              <div className="font-medium">{evaluation.evaluation_name}</div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>{evaluation.flow_id}</span>
                <span>·</span>
                <span>{evaluation.model_id}</span>
                <span>·</span>
                <span>{evaluation.dataset_name}</span>
              </div>
            </TableCell>

            <TableCell>
              <div className="flex flex-col space-y-2">
                <div className="flex items-center justify-between">
                  <Badge variant={getStatusColor(evaluation.status)}>
                    {evaluation.status}
                  </Badge>
                  <span className="text-xs font-medium">
                    {evaluation.status.toLowerCase() === "completed"
                      ? `${Math.round(getResolvedPercentage(evaluation))}% resolved`
                      : `${Math.round(getProgressPercentage(evaluation))}% complete`}
                  </span>
                </div>

                <Progress
                  value={
                    evaluation.status.toLowerCase() === "completed"
                      ? getResolvedPercentage(evaluation)
                      : getProgressPercentage(evaluation)
                  }
                  className={`h-2 ${evaluation.status.toLowerCase() === "completed" ? "bg-green-100" : ""}`}
                />

                {evaluation.status_summary && (
                  <div className="flex flex-wrap gap-1 text-xs">
                    <span className="text-destructive">
                      {evaluation.status_summary.error} error
                    </span>
                    <span className="text-muted-foreground mx-1">|</span>
                    <span>{evaluation.status_summary.resolved} resolved</span>
                    <span className="text-muted-foreground mx-1">|</span>
                    <span>{evaluation.status_summary.completed} completed</span>
                    <span className="text-muted-foreground mx-1">|</span>
                    <span>{evaluation.status_summary.pending} pending</span>
                  </div>
                )}
              </div>
            </TableCell>

            <TableCell>
              <div className="text-sm">{formatDate(evaluation.started_at)}</div>
            </TableCell>

            <TableCell>
              <div className="text-sm">
                {evaluation.completed_at
                  ? formatDate(evaluation.completed_at)
                  : "-"}
              </div>
            </TableCell>

            <TableCell>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1" title="Cost">
                  <Coins className="h-3 w-3 text-muted-foreground" />
                  <div className="text-xs">
                    ${formatNumber(evaluation.total_cost, 2)}
                  </div>
                </div>
                <div className="flex items-center gap-1" title="Prompt Tokens">
                  <MessageSquare className="h-3 w-3 text-muted-foreground" />
                  <div className="text-xs">
                    {formatNumber(evaluation.prompt_tokens)}
                  </div>
                </div>
                <div
                  className="flex items-center gap-1"
                  title="Completion Tokens"
                >
                  <Cpu className="h-3 w-3 text-muted-foreground" />
                  <div className="text-xs">
                    {formatNumber(evaluation.completion_tokens)}
                  </div>
                </div>
                <div className="flex items-center gap-1" title="Cached Tokens">
                  <Zap className="h-3 w-3 text-muted-foreground" />
                  <div className="text-xs">
                    {formatNumber(evaluation.cached_tokens)}
                  </div>
                </div>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
