"use client";

import { toast } from "sonner";
import { PageLayout } from "@/lib/components/layouts/PageLayout";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import EvaluationForm from "@/features/swebench/components/EvaluationForm";
import EvaluationStatus from "@/features/swebench/components/EvaluationStatus";
import { useEvaluationStart } from "@/features/swebench/hooks/useEvaluationCreate";
import type { EvaluationRequest } from "@/features/swebench/api/evaluation";
import { useNavigate } from "react-router-dom";
import { Button } from "@/lib/components/ui/button";
import { Plus } from "lucide-react";
import { useEvaluationsList } from "@/features/swebench/hooks/useEvaluationsList";
import { Card, CardContent, CardHeader, CardTitle } from "@/lib/components/ui/card";
import { Badge } from "@/lib/components/ui/badge";
import { Skeleton } from "@/lib/components/ui/skeleton";

type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

export function EvaluationsPage() {
  const navigate = useNavigate();
  const { data: evaluations, isLoading } = useEvaluationsList();

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
                  <div>
                    <CardTitle className="text-lg">{evaluation.dataset_name}</CardTitle>
                    <p className="text-sm text-muted-foreground mt-1">
                      {evaluation.evaluation_name}
                    </p>
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
                        {formatDate(evaluation.start_time)}
                      </p>
                    </div>
                    {evaluation.finish_time && (
                      <div>
                        <p className="text-sm font-medium">Finished</p>
                        <p className="text-sm text-muted-foreground">
                          {formatDate(evaluation.finish_time)}
                        </p>
                      </div>
                    )}
                  </div>

                  {evaluation.status_summary && (
                    <div>
                      <p className="text-sm font-medium mb-2">Progress</p>
                      <div className="grid grid-cols-4 gap-2">
                        <div>
                          <Badge variant="outline" className="w-full justify-center">
                            {evaluation.status_summary.pending} pending
                          </Badge>
                        </div>
                        <div>
                          <Badge variant="outline" className="w-full justify-center">
                            {evaluation.status_summary.started} running
                          </Badge>
                        </div>
                        <div>
                          <Badge variant="outline" className="w-full justify-center">
                            {evaluation.status_summary.completed} completed
                          </Badge>
                        </div>
                        <div>
                          <Badge variant="outline" className="w-full justify-center">
                            {evaluation.status_summary.error} failed
                          </Badge>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </PageLayout>
  );
} 