import { EvaluationTable } from "@/features/swebench/components/EvaluationTable";
import { useEvaluationsList } from "@/features/swebench/hooks/useEvaluationsList";
import { PageLayout } from "@/lib/components/layouts/PageLayout";
import { Button } from "@/lib/components/ui/button";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { Plus } from "lucide-react";
import React from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

export function EvaluationsPage() {
  const navigate = useNavigate();
  const { data: evaluations, isLoading, error } = useEvaluationsList();

  React.useEffect(() => {
    if (error) {
      toast.error("Failed to fetch evaluations", {
        description: error.message || "Please try again later",
      });
    }
  }, [error]);

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
        <div className="pr-4">
          <EvaluationTable
            evaluations={evaluations?.evaluations}
            isLoading={isLoading}
          />
        </div>
      </ScrollArea>
    </PageLayout>
  );
}
