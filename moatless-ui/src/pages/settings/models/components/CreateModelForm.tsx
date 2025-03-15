import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useNavigate } from "react-router-dom";
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
import { Button } from "@/lib/components/ui/button";
import { Checkbox } from "@/lib/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { useCreateModel } from "@/lib/hooks/useModels";
import { toast } from "sonner";
import { ModelConfigSchema } from "@/lib/types/model";

// Create a schema for the form with required fields
const createModelSchema = ModelConfigSchema.extend({
  model_id: z.string().min(1, "Model ID is required"),
  model: z.string().min(1, "Model name is required"),
});

// Default values for the form
const defaultValues = {
  model_id: "",
  model: "",
  model_base_url: "",
  model_api_key: "",
  temperature: 0.7,
  max_tokens: 4096,
  timeout: 120,
  thoughts_in_action: false,
  disable_thoughts: false,
  merge_same_role_messages: false,
  message_cache: true,
  few_shot_examples: true
};

export function CreateModelForm() {
  const navigate = useNavigate();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const createModelMutation = useCreateModel();

  const form = useForm<z.infer<typeof createModelSchema>>({
    resolver: zodResolver(createModelSchema),
    defaultValues,
  });

  const onSubmit = async (data: z.infer<typeof createModelSchema>) => {
    try {
      setIsSubmitting(true);
      await createModelMutation.mutateAsync(data);
      toast.success("Model created successfully");
      navigate(`/settings/models/${encodeURIComponent(data.id)}`);
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : (error as any)?.response?.data?.detail || "Failed to create model";
      toast.error(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto py-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Create New Model</h1>
      </div>

      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left Column - Basic Settings */}
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">Basic Settings</h2>

              <FormField
                control={form.control}
                name="model_id"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Model ID</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        placeholder="Enter a unique identifier for your model"
                        autoFocus
                      />
                    </FormControl>
                    <FormDescription>
                      A unique identifier for your model
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="model"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Model Name</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        placeholder="e.g. anthropic/claude-3-sonnet-20240229"
                      />
                    </FormControl>
                    <FormDescription>
                      The LiteLLM model identifier to use. See{" "}
                      <a
                        href="https://docs.litellm.ai/docs/providers"
                        target="_blank"
                        className="text-primary hover:underline"
                      >
                        LiteLLM providers
                      </a>
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="model_base_url"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Model Base URL</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        value={field.value ?? ""}
                        placeholder="e.g. http://localhost:8000/v1"
                      />
                    </FormControl>
                    <FormDescription>
                      Optional base URL for the model API
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="model_api_key"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Model API Key</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        value={field.value ?? ""}
                        type="password"
                        placeholder="Optional API key"
                      />
                    </FormControl>
                    <FormDescription>
                      Optional API key for the model
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="timeout"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Timeout (seconds)</FormLabel>
                    <FormControl>
                      <Input {...field} type="number" min="1" step="1" />
                    </FormControl>
                    <FormDescription>
                      Request timeout in seconds
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            {/* Right Column - Model Parameters */}
            <div className="space-y-4">
              <h2 className="text-lg font-semibold">Model Parameters</h2>


              <FormField
                control={form.control}
                name="temperature"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Temperature</FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        type="number"
                        min="0"
                        max="1"
                        step="0.1"
                      />
                    </FormControl>
                    <FormDescription>
                      Randomness in model output
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="max_tokens"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Max Tokens</FormLabel>
                    <FormControl>
                      <Input {...field} type="number" />
                    </FormControl>
                    <FormDescription>
                      Maximum tokens to generate
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          </div>

          {/* Features Section - Full Width */}
          <div className="mt-6">
            <h2 className="text-lg font-semibold mb-4">Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="thoughts_in_action"
                render={({ field }) => (
                  <FormItem className="flex items-start space-x-3 space-y-0">
                    <FormControl>
                      <Checkbox
                        checked={field.value as boolean}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel>Thoughts in Action</FormLabel>
                      <FormDescription>
                        Include thought generation in action steps
                      </FormDescription>
                    </div>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="disable_thoughts"
                render={({ field }) => (
                  <FormItem className="flex items-start space-x-3 space-y-0">
                    <FormControl>
                      <Checkbox
                        checked={field.value as boolean}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel>Disable Thoughts</FormLabel>
                      <FormDescription>
                        Disable thought generation completely
                      </FormDescription>
                    </div>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="merge_same_role_messages"
                render={({ field }) => (
                  <FormItem className="flex items-start space-x-3 space-y-0">
                    <FormControl>
                      <Checkbox
                        checked={field.value as boolean}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel>Merge Same Role Messages</FormLabel>
                      <FormDescription>
                        Combine consecutive messages from the same role
                      </FormDescription>
                    </div>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="message_cache"
                render={({ field }) => (
                  <FormItem className="flex items-start space-x-3 space-y-0">
                    <FormControl>
                      <Checkbox
                        checked={field.value as boolean}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel>Message Cache</FormLabel>
                      <FormDescription>
                        Enable caching of message responses
                      </FormDescription>
                    </div>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="few_shot_examples"
                render={({ field }) => (
                  <FormItem className="flex items-start space-x-3 space-y-0">
                    <FormControl>
                      <Checkbox
                        checked={field.value as boolean}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel>Few Shot Examples</FormLabel>
                      <FormDescription>
                        Include few-shot examples in prompts
                      </FormDescription>
                    </div>
                  </FormItem>
                )}
              />
            </div>
          </div>

          <div className="mt-8 flex items-center justify-end gap-4">
            <Button
              type="button"
              variant="outline"
              onClick={() => navigate("/settings/models")}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Creating..." : "Create Model"}
            </Button>
          </div>
        </form>
      </Form>
    </div>
  );
}
