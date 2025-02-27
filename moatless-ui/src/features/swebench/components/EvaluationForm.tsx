import { Card, CardContent, CardHeader, CardTitle } from "@/lib/components/ui/card";
import { Button } from "@/lib/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormDescription, FormMessage } from "@/lib/components/ui/form";
import { Input } from "@/lib/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/lib/components/ui/select";
import { Separator } from "@/lib/components/ui/separator";
import { ModelSelector } from "@/lib/components/selectors/ModelSelector";
import { useDatasetsList } from "../hooks/useDatasetsList";
import type { EvaluationRequest } from "../api/evaluation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { toast } from "sonner";
import { useLastUsedStore } from "@/lib/stores/lastUsedStore";
import { FlowSelector } from "@/lib/components/selectors/FlowSelector";
import { format } from "date-fns";
import React from "react";

const evaluationSchema = z.object({
  name: z.string().min(1, "Evaluation name is required"),
  dataset: z.string().min(1, "Please select a dataset"),
  flow_id: z.string().min(1, "Please select a flow"),
  model_id: z.string().min(1, "Please select a model")
});

type EvaluationFormData = z.infer<typeof evaluationSchema>;

interface EvaluationFormProps {
  onSubmit: (data: EvaluationRequest) => void;
  isLoading?: boolean;
}

export default function EvaluationForm({ onSubmit, isLoading }: EvaluationFormProps) {
  const { data: datasetsResponse, isError } = useDatasetsList();
  const { lastUsedModel } = useLastUsedStore();
  const [isNameManuallyEdited, setIsNameManuallyEdited] = React.useState(false);
  
  const generateDefaultName = (values: Partial<EvaluationFormData>) => {
    const date = format(new Date(), 'yyyyMMdd');
    return `${date}_${values.flow_id || 'flow'}_${values.model_id || 'model'}_${values.dataset || 'dataset'}`;
  };
  
  const form = useForm<EvaluationFormData>({
    resolver: zodResolver(evaluationSchema),
    defaultValues: {
      name: generateDefaultName({}),
      flow_id: "",
      model_id: lastUsedModel,
      dataset: ""
    },
  });

  // Watch for changes in relevant fields to update the name
  const watchedFields = form.watch(['flow_id', 'model_id', 'dataset']);
  React.useEffect(() => {
    if (!isNameManuallyEdited) {
      const newName = generateDefaultName({
        flow_id: watchedFields[0],
        model_id: watchedFields[1],
        dataset: watchedFields[2],
      });
      form.setValue('name', newName);
    }
  }, [watchedFields, isNameManuallyEdited]);

  if (isError) {
    toast.error("Failed to load datasets");
  }

  const handleSubmit = (data: EvaluationFormData) => {
    onSubmit({
      ...data
    });
  };

  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>Start New Evaluation</CardTitle>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-8">
            
            {/* Flow and Model Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Flow & Model Configuration</h3>
              <div className="grid gap-6 sm:grid-cols-2">
                <FormField
                  control={form.control}
                  name="flow_id"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Flow</FormLabel>
                      <FormControl>
                        <FlowSelector
                          selectedFlowId={field.value}
                          onFlowSelect={field.onChange}
                        />
                      </FormControl>
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
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            <Separator />

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

            
            {/* Evaluation Name Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Evaluation Name</h3>
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Name</FormLabel>
                    <FormControl>
                      <Input 
                        {...field} 
                        onChange={(e) => {
                          setIsNameManuallyEdited(true);
                          field.onChange(e);
                        }}
                      />
                    </FormControl>
                    <FormDescription>
                      A unique identifier for this evaluation
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
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