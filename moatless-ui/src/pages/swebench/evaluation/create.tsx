"use client";

import { toast } from "sonner";
import { PageLayout } from "@/lib/components/layouts/PageLayout";
import { useNavigate } from "react-router-dom";
import EvaluationForm from "@/features/swebench/components/EvaluationForm";
import { useEvaluationCreate } from "@/features/swebench/hooks/useEvaluationCreate";
import type { EvaluationRequest } from "@/features/swebench/api/evaluation";

export function CreateEvaluationPage() {
  const navigate = useNavigate();
  const { mutate: createEvaluation, isPending } = useEvaluationCreate();

  const handleSubmit = (formData: EvaluationRequest) => {
    createEvaluation(formData, {
      onSuccess: (data) => {
        toast.success("Evaluation created successfully");
        // Navigate to the evaluation details page
        navigate(`/swebench/evaluation/${data.evaluation_name}`);
      },
      onError: (error) => {
        toast.error("Failed to create evaluation", {
          description: error instanceof Error ? error.message : "Unknown error",
        });
      },
    });
  };

  return (
    <PageLayout>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">New Evaluation</h1>
      </div>

      <div className="space-y-8">
        <EvaluationForm onSubmit={handleSubmit} isLoading={isPending} />
      </div>
    </PageLayout>
  );
} 