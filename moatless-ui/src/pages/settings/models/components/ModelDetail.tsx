import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
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
import { ModelConfigSchema, type ModelConfig } from "@/lib/types/model";
import { useState, useEffect } from "react";
import { Loader2, PlayCircle, CheckCircle2, XCircle } from "lucide-react";
import { useTestModel } from "@/lib/hooks/useModels";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";

interface ModelDetailProps {
  model: ModelConfig;
  onSubmit: (data: ModelConfig) => Promise<void>;
}

export function ModelDetail({ model, onSubmit }: ModelDetailProps) {
  const form = useForm<ModelConfig>({
    resolver: zodResolver(ModelConfigSchema),
    defaultValues: model,
  });

  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const testModelMutation = useTestModel();

  // Reset form when model changes
  useEffect(() => {
    form.reset(model);
  }, [form, model]);

  const handleSubmit = async (data: ModelConfig) => {
    try {
      setIsSaving(true);
      setError(null);
      await onSubmit(data);
    } catch (e) {
      const errorMessage = e instanceof Error 
        ? e.message 
        : (e as any)?.response?.data?.detail || "An unexpected error occurred";
      
      console.error(errorMessage);
      setError(errorMessage);
      throw e;
    } finally {
      setIsSaving(false);
    }
  };

  const handleTestModel = async () => {
    try {
      await testModelMutation.mutateAsync(model.id);
    } catch (e) {
      const errorMessage = e instanceof Error 
        ? e.message 
        : (e as any)?.response?.data?.detail || "Failed to test model";
      console.error(errorMessage);
    }
  };

  // Only show loading when form is validating and not dirty
  if (form.formState.isValidating && !form.formState.isDirty) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
        {/* Test Results Alert - Show if test was run */}
        {testModelMutation.data && (
          <Alert
            variant={testModelMutation.data.success ? "default" : "destructive"}
            className="mb-6"
          >
            <div className="flex items-center gap-2">
              {testModelMutation.data.success ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500" />
              )}
              <AlertTitle>
                {testModelMutation.data.success ? "Test Passed" : "Test Failed"}
              </AlertTitle>
            </div>
            <AlertDescription className="mt-2 space-y-2">
              <p>{testModelMutation.data.message}</p>
              {testModelMutation.data.response_time_ms && (
                <p className="text-sm text-muted-foreground">
                  Response time: {(testModelMutation.data.response_time_ms / 1000).toFixed(2)}s
                </p>
              )}
              {testModelMutation.data.error_details && (
                <p className="text-sm text-red-500">
                  Error: {testModelMutation.data.error_details}
                </p>
              )}
            </AlertDescription>
          </Alert>
        )}

        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">Model Configuration</h1>
          <Button
            type="button"
            variant="outline"
            onClick={handleTestModel}
            disabled={testModelMutation.isPending}
            className="gap-2"
          >
            {testModelMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <PlayCircle className="h-4 w-4" />
            )}
            {testModelMutation.isPending ? "Testing..." : "Test Model"}
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left Column - Basic Settings */}
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Basic Settings</h2>
            
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
                      value={field.value ?? ''}
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
                      value={field.value ?? ''}
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
              name="response_format"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Response Format</FormLabel>
                  <Select
                    onValueChange={field.onChange}
                    value={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select response format" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="react">React</SelectItem>
                      <SelectItem value="tool_call">Tool Call</SelectItem>
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    Format for model responses
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="message_history_type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Message History Type</FormLabel>
                  <Select
                    onValueChange={field.onChange}
                    value={field.value}
                  >
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select history type" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="react">React</SelectItem>
                      <SelectItem value="messages">Messages</SelectItem>
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    Message history format
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="temperature"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Temperature</FormLabel>
                  <FormControl>
                    <Input {...field} type="number" min="0" max="1" step="0.1" />
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
          {error && (
            <p className="text-sm text-red-500 flex-1 break-words">
              {error}
            </p>
          )}
          <Button type="submit" disabled={isSaving || testModelMutation.isPending}>
            {isSaving ? "Saving..." : "Save Changes"}
          </Button>
        </div>
      </form>
    </Form>
  );
}
