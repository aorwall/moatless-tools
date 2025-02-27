import { Badge } from "@/lib/components/ui/badge";
import { Progress } from "@/lib/components/ui/progress";
import { Button } from "@/lib/components/ui/button";
import { Play } from "lucide-react";
import { Evaluation } from "../api/evaluation";
import { useStartEvaluation } from "../hooks/useStartEvaluation";
import { toast } from "sonner";

type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

interface EvaluationOverviewProps {
  evaluation: Evaluation;
  calculateProgress: () => number;
  getStatusColor: (status: string) => BadgeVariant;
}

export function EvaluationOverview({ evaluation, calculateProgress, getStatusColor }: EvaluationOverviewProps) {

  return (
    <div className="border-b bg-muted/40 p-4">
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <div className="min-w-0 flex-1">
            <h2 className="font-semibold truncate">{evaluation.dataset_name}</h2>
            <p className="text-sm text-muted-foreground truncate">
              {evaluation.evaluation_name}
            </p>
          </div>
          <div className="flex-shrink-0">
            <Badge variant={getStatusColor(evaluation.status)}>
              {evaluation.status}
            </Badge>
          </div>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Progress</span>
            <span>{Math.round(calculateProgress())}%</span>
          </div>
          <Progress value={calculateProgress()} />
        </div>
      </div>
    </div>
  );
} 