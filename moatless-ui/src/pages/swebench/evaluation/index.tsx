"use client";

import { toast } from "sonner";
import { PageLayout } from "@/lib/components/layouts/PageLayout";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { useNavigate } from "react-router-dom";
import { Button } from "@/lib/components/ui/button";
import { Plus, Coins, Cpu, MessageSquare, Zap } from "lucide-react";
import { useEvaluationsList } from "@/features/swebench/hooks/useEvaluationsList";
import { Card, CardContent, CardHeader, CardTitle } from "@/lib/components/ui/card";
import { Badge } from "@/lib/components/ui/badge";
import { Skeleton } from "@/lib/components/ui/skeleton";
import React from "react";
import { Progress } from "@/lib/components/ui/progress";
import { formatNumber } from "@/lib/utils/format";

type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

export function EvaluationsPage() {
  const navigate = useNavigate();
  const { data: evaluations, isLoading, error } = useEvaluationsList();

  React.useEffect(() => {
    if (error) {
      toast.error('Failed to fetch evaluations', {
        description: error.message || 'Please try again later'
      });
    }
  }, [error]);

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

  const getProgressPercentage = (evaluation: EvaluationListItem) => {
    if (!evaluation.status_summary) return 0;
    const total = evaluation.instance_count;
    const completed = evaluation.status_summary.error + 
                     evaluation.status_summary.resolved + 
                     evaluation.status_summary.completed;
    return (completed / total) * 100;
  };

  return (
    <PageLayout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Evaluations</h1>
        <Button onClick={() => navigate("create")}>
          <Plus className="mr-2 h-4 w-4" />
          New Evaluation
        </Button>
      </div>

      <ScrollArea className="h-[calc(100vh-56px-8rem)]">
        <div className="space-y-4 pr-4">
          {isLoading ? (
            Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-[200px] w-full" />
            ))
          ) : evaluations?.evaluations.map((evaluation) => (
            <Card 
              key={evaluation.evaluation_name} 
              className="hover:bg-muted/50 transition-colors cursor-pointer"
              onClick={() => navigate(evaluation.evaluation_name)}
            >
              <CardHeader className="pb-3">
                <div className="flex justify-between items-start">
                  <div className="space-y-2">
                    <CardTitle className="text-lg">
                      {evaluation.evaluation_name}
                    </CardTitle>
                    <div className="flex items-center gap-2 text-sm">
                      <span className="font-medium">{evaluation.flow_id}</span>
                      <span className="text-muted-foreground">·</span>
                      <span className="font-medium">{evaluation.model_id}</span>
                      <span className="text-muted-foreground">·</span>
                      <span className="font-medium">{evaluation.dataset_name}</span>
                    </div>
                  </div>
                  <Badge variant={getStatusColor(evaluation.status)}>
                    {evaluation.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium">Started</p>
                      <p className="text-sm text-muted-foreground">
                        {formatDate(evaluation.started_at)}
                      </p>
                    </div>
                    {evaluation.completed_at && (
                      <div>
                        <p className="text-sm font-medium">Finished</p>
                        <p className="text-sm text-muted-foreground">
                          {formatDate(evaluation.completed_at)}
                        </p>
                      </div>
                    )}
                  </div>

                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="font-medium">Progress</span>
                      <span className="text-muted-foreground">
                        {Math.round(getProgressPercentage(evaluation))}%
                      </span>
                    </div>
                    <Progress value={getProgressPercentage(evaluation)} className="h-2" />
                  </div>

                  {evaluation.status_summary && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="col-span-2">
                        <div className="grid grid-cols-4 gap-2">
                          <Badge 
                            variant="outline" 
                            className="justify-center text-destructive border-destructive"
                          >
                            {evaluation.status_summary.error} error
                          </Badge>
                          <Badge variant="outline" className="justify-center">
                            {evaluation.status_summary.resolved} resolved
                          </Badge>
                          <Badge variant="outline" className="justify-center">
                            {evaluation.status_summary.completed} completed
                          </Badge>
                          <Badge variant="outline" className="justify-center">
                            {evaluation.status_summary.pending} pending
                          </Badge>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-4 gap-2 pt-2 border-t">
                    <div className="flex items-center gap-2">
                      <Coins className="h-4 w-4 text-muted-foreground" />
                      <div className="text-sm">
                        ${formatNumber(evaluation.total_cost, 2)}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <MessageSquare className="h-4 w-4 text-muted-foreground" />
                      <div className="text-sm">
                        {formatNumber(evaluation.prompt_tokens)}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Cpu className="h-4 w-4 text-muted-foreground" />
                      <div className="text-sm">
                        {formatNumber(evaluation.completion_tokens)}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-muted-foreground" />
                      <div className="text-sm">
                        {formatNumber(evaluation.cached_tokens)}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </PageLayout>
  );
} 