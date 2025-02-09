"use client";

import { useState } from "react";
import { Box, Container, VStack, useToast } from "@chakra-ui/react";
import { EvaluationForm } from "@/lib/components/swebench/evaluation/EvaluationForm";
import { EvaluationStatus } from "@/lib/components/swebench/evaluation/EvaluationStatus";

export default function EvaluationPage() {
  const [evaluationName, setEvaluationName] = useState<string | null>(null);
  const toast = useToast();

  const handleSubmit = async (formData: any) => {
    try {
      const response = await fetch("/api/swebench/evaluate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setEvaluationName(data.evaluation_name);
      toast({
        title: "Evaluation started",
        description: `Evaluation ID: ${data.evaluation_name}`,
        status: "success",
      });
    } catch (error) {
      toast({
        title: "Failed to start evaluation",
        description: error instanceof Error ? error.message : "Unknown error",
        status: "error",
      });
    }
  };

  return (
    <Container maxW="container.xl" py={8}>
      <VStack spacing={8}>
        <EvaluationForm onSubmit={handleSubmit} />
        {evaluationName && <EvaluationStatus evaluationName={evaluationName} />}
      </VStack>
    </Container>
  );
} 