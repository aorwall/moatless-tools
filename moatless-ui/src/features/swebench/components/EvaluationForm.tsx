import { Card, CardContent, CardHeader, CardTitle } from "@/lib/components/ui/card";
import { Button } from "@/lib/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormDescription, FormMessage } from "@/lib/components/ui/form";
import { Input } from "@/lib/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/lib/components/ui/select";
import { Separator } from "@/lib/components/ui/separator";
import { AgentSelector } from "@/lib/components/selectors/AgentSelector";
import { ModelSelector } from "@/lib/components/selectors/ModelSelector";
import { useDatasetsList } from "../hooks/useDatasetsList";
import type { EvaluationRequest } from "../api/evaluation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { toast } from "sonner";
import { useLastUsedStore } from "@/lib/stores/lastUsedStore";

const evaluationSchema = z.object({
  dataset: z.string().min(1, "Please select a dataset"),
  agent_id: z.string().min(1, "Please select an agent"),
  model_id: z.string().min(1, "Please select a model"),
  num_workers: z.number()
    .min(1, "Must have at least 1 worker")
    .max(10, "Maximum 10 workers allowed"),
  max_iterations: z.number()
    .min(1, "Must have at least 1 iteration")
    .max(50, "Maximum 50 iterations allowed"),
  max_expansions: z.number()
    .min(1, "Must have at least 1 expansion")
    .max(10, "Maximum 10 expansions allowed"),
});

type EvaluationFormData = z.infer<typeof evaluationSchema>;

interface EvaluationFormProps {
  onSubmit: (data: EvaluationRequest) => void;
  isLoading?: boolean;
}

export default function EvaluationForm({ onSubmit, isLoading }: EvaluationFormProps) {
  const { data: datasetsResponse, isError } = useDatasetsList();
  const { lastUsedAgent, lastUsedModel } = useLastUsedStore();
  
  const form = useForm<EvaluationFormData>({
    resolver: zodResolver(evaluationSchema),
    defaultValues: {
      agent_id: lastUsedAgent,
      model_id: lastUsedModel,
      dataset: "",
      num_workers: 1,
      max_iterations: 15,
      max_expansions: 1,
    },
  });

  if (isError) {
    toast.error("Failed to load datasets");
  }

  const handleSubmit = (data: EvaluationFormData) => {
    onSubmit({
      ...data,
      num_workers: Number(data.num_workers),
      max_iterations: Number(data.max_iterations),
      max_expansions: Number(data.max_expansions),
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Start New Evaluation</CardTitle>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-8">
            {/* Dataset Selection Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Dataset</h3>
              <FormField
                control={form.control}
                name="dataset"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Select Dataset</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Choose a dataset to evaluate" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {datasetsResponse?.datasets.map((dataset) => (
                          <SelectItem key={dataset.name} value={dataset.name}>
                            {dataset.name} ({dataset.instance_count} instances)
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormDescription>
                      The dataset containing instances to evaluate
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <Separator />

            {/* Agent and Model Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Agent & Model Configuration</h3>
              <div className="grid gap-6 sm:grid-cols-2">
                <FormField
                  control={form.control}
                  name="agent_id"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Agent</FormLabel>
                      <FormControl>
                        <AgentSelector
                          selectedAgentId={field.value}
                          onAgentSelect={field.onChange}
                        />
                      </FormControl>
                      <FormDescription>
                        The agent that will process the instances
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="model_id"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Model</FormLabel>
                      <FormControl>
                        <ModelSelector
                          selectedModelId={field.value}
                          onModelSelect={field.onChange}
                        />
                      </FormControl>
                      <FormDescription>
                        The language model to use for evaluation
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            <Separator />

            {/* Evaluation Settings Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Evaluation Settings</h3>
              <div className="grid gap-6 sm:grid-cols-3">
                <FormField
                  control={form.control}
                  name="num_workers"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Workers</FormLabel>
                      <FormControl>
                        <Input 
                          type="number" 
                          min={1} 
                          max={10}
                          {...field}
                          onChange={e => field.onChange(Number(e.target.value))}
                        />
                      </FormControl>
                      <FormDescription>
                        Number of parallel workers (1-10)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="max_iterations"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Max Iterations</FormLabel>
                      <FormControl>
                        <Input 
                          type="number" 
                          min={1} 
                          max={50}
                          {...field}
                          onChange={e => field.onChange(Number(e.target.value))}
                        />
                      </FormControl>
                      <FormDescription>
                        Maximum iterations per instance (1-50)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="max_expansions"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Max Expansions</FormLabel>
                      <FormControl>
                        <Input 
                          type="number" 
                          min={1} 
                          max={10}
                          {...field}
                          onChange={e => field.onChange(Number(e.target.value))}
                        />
                      </FormControl>
                      <FormDescription>
                        Maximum expansions per iteration (1-10)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            <div className="flex justify-end">
              <Button 
                type="submit" 
                disabled={isLoading || !form.formState.isValid} 
                className="w-[200px]"
              >
                {isLoading ? "Creating Evaluation..." : "Create Evaluation"}
              </Button>
            </div>
          </form>
        </Form>
      </CardContent>
    </Card>
  );
} 