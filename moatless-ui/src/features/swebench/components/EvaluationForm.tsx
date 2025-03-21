import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/lib/components/ui/card";
import { Button } from "@/lib/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormDescription,
  FormMessage,
} from "@/lib/components/ui/form";
import { Input } from "@/lib/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
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
import { RadioGroup, RadioGroupItem } from "@/lib/components/ui/radio-group";
import { Badge } from "@/lib/components/ui/badge";
import { Label } from "@/lib/components/ui/label";
import { Slider } from "@/lib/components/ui/slider";
import { InstanceSelectionDialog } from "./InstanceSelectionDialog";

const evaluationSchema = z.object({
  name: z.string().min(1, "Evaluation name is required"),
  selectionType: z.enum(["dataset", "instances"]),
  dataset: z.string().optional(),
  flow_id: z.string().min(1, "Please select a flow"),
  model_id: z.string().min(1, "Please select a model"),
  num_concurrent_instances: z.number().min(1).default(1),
  instance_ids: z.array(z.string()).optional(),
});

type EvaluationFormData = z.infer<typeof evaluationSchema>;

interface EvaluationFormProps {
  onSubmit: (data: EvaluationRequest) => void;
  isLoading?: boolean;
}

export default function EvaluationForm({
  onSubmit,
  isLoading,
}: EvaluationFormProps) {
  const { data: datasetsResponse, isError } = useDatasetsList();
  const { lastUsedModel } = useLastUsedStore();
  const [isNameManuallyEdited, setIsNameManuallyEdited] = React.useState(false);
  const [isInstanceDialogOpen, setIsInstanceDialogOpen] = React.useState(false);

  const generateDefaultName = (values: Partial<EvaluationFormData>) => {
    const date = format(new Date(), "yyyyMMdd");
    return `${date}_${values.flow_id || "flow"}_${values.model_id || "model"}_${values.selectionType === "dataset"
      ? values.dataset || "dataset"
      : "custom_instances"
      }`;
  };

  const form = useForm<EvaluationFormData>({
    resolver: zodResolver(evaluationSchema),
    defaultValues: {
      name: generateDefaultName({}),
      flow_id: "",
      model_id: lastUsedModel,
      selectionType: "dataset",
      dataset: "",
      num_concurrent_instances: 1,
      instance_ids: [],
    },
  });

  // Watch for changes in relevant fields to update the name
  const watchedFields = form.watch([
    "flow_id",
    "model_id",
    "dataset",
    "selectionType",
  ]);
  React.useEffect(() => {
    if (!isNameManuallyEdited) {
      const newName = generateDefaultName({
        flow_id: watchedFields[0],
        model_id: watchedFields[1],
        dataset: watchedFields[2],
        selectionType: watchedFields[3] as "dataset" | "instances",
      });
      form.setValue("name", newName);
    }
  }, [watchedFields, isNameManuallyEdited]);

  if (isError) {
    toast.error("Failed to load datasets");
  }

  const handleSubmit = (data: EvaluationFormData) => {
    // If instances are selected, make sure we have at least one
    if (data.selectionType === "instances" && (!data.instance_ids || data.instance_ids.length === 0)) {
      toast.error("Please select at least one instance");
      return;
    }

    // If dataset is selected, make sure we have one
    if (data.selectionType === "dataset" && !data.dataset) {
      toast.error("Please select a dataset");
      return;
    }

    onSubmit({
      ...data,
      dataset: data.selectionType === "dataset" ? data.dataset! : "",
      instance_ids: data.selectionType === "instances" ? data.instance_ids! : [],
    });
  };

  const selectionType = form.watch("selectionType");
  const selectedInstanceIds = form.watch("instance_ids") || [];

  // Handler for when instances are selected in the dialog
  const handleInstancesSelect = (instanceIds: string[]) => {
    form.setValue("instance_ids", instanceIds);
  };

  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>Start New Evaluation</CardTitle>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit(handleSubmit)}
            className="space-y-8"
          >
            {/* Flow and Model Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">
                Flow & Model Configuration
              </h3>
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

            {/* Instance Selection Type */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Evaluation Scope</h3>
              <FormField
                control={form.control}
                name="selectionType"
                render={({ field }) => (
                  <FormItem className="space-y-3">
                    <FormLabel>Select By</FormLabel>
                    <FormControl>
                      <RadioGroup
                        onValueChange={field.onChange}
                        defaultValue={field.value}
                        className="flex flex-col space-y-1"
                      >
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="dataset" id="dataset" />
                          <Label htmlFor="dataset">Dataset</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="instances" id="instances" />
                          <Label htmlFor="instances">Specific Instances</Label>
                        </div>
                      </RadioGroup>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            {/* Dataset Selection Section */}
            {selectionType === "dataset" && (
              <div className="space-y-4">
                <FormField
                  control={form.control}
                  name="dataset"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Select Dataset</FormLabel>
                      <Select
                        onValueChange={field.onChange}
                        value={field.value}
                      >
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
            )}

            {/* Instance Selection Section */}
            {selectionType === "instances" && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-md font-medium">Selected Instances</h4>
                    <p className="text-sm text-muted-foreground">
                      {selectedInstanceIds.length === 0
                        ? "No instances selected"
                        : `${selectedInstanceIds.length} instance(s) selected`}
                    </p>
                  </div>
                  <Button onClick={() => setIsInstanceDialogOpen(true)}>
                    Select Instances
                  </Button>
                </div>

                {/* Show selected instances summary */}
                {selectedInstanceIds.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {selectedInstanceIds.slice(0, 10).map((id) => (
                      <Badge key={id} variant="outline" className="text-xs">
                        {id}
                      </Badge>
                    ))}
                    {selectedInstanceIds.length > 10 && (
                      <Badge variant="outline" className="text-xs">
                        +{selectedInstanceIds.length - 10} more
                      </Badge>
                    )}
                  </div>
                )}

                {/* Instance Selection Dialog */}
                <InstanceSelectionDialog
                  open={isInstanceDialogOpen}
                  onOpenChange={setIsInstanceDialogOpen}
                  onInstancesSelect={handleInstancesSelect}
                  selectedInstanceIds={selectedInstanceIds}
                />
              </div>
            )}

            <Separator />

            {/* Concurrent Instances */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Performance Settings</h3>
              <FormField
                control={form.control}
                name="num_concurrent_instances"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Concurrent Instances: {field.value}</FormLabel>
                    <FormControl>
                      <Slider
                        defaultValue={[field.value]}
                        min={1}
                        max={10}
                        step={1}
                        onValueChange={(value) => field.onChange(value[0])}
                      />
                    </FormControl>
                    <FormDescription>
                      Number of instances to evaluate concurrently
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
