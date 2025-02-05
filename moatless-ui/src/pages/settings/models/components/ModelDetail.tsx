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
import { useState } from "react";

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

  const handleSubmit = async (data: ModelConfig) => {
    try {
      setIsSaving(true);
      setError(null);
      await onSubmit(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "An error occurred");
      throw e;
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
        <div className="space-y-4">
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
                  </a>{" "}
                  for available models.
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
                    placeholder="e.g. http://localhost:8000/v1"
                  />
                </FormControl>
                <FormDescription>
                  Optional base URL for the model API. Leave empty to use the
                  default provider URL.
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
                    type="password"
                    placeholder="Optional, will use CUSTOM_LLM_API_KEY env var if not set"
                  />
                </FormControl>
                <FormDescription>
                  Optional API key for the model. If not set, will use
                  CUSTOM_LLM_API_KEY environment variable.
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="responseFormat"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Response Format</FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
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
                  Format for model responses - 'react' uses structured ReACT
                  format with thought/action/params in XML/JSON, 'tool_call'
                  uses function calling
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="messageHistoryType"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Message History Type</FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                >
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue placeholder="Select message history type" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value="react">React</SelectItem>
                    <SelectItem value="messages">Messages</SelectItem>
                  </SelectContent>
                </Select>
                <FormDescription>
                  How message history is formatted in completions - 'messages'
                  keeps full message list unchanged, 'react' uses ReACT format
                  with optimized history to reduce tokens
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
                  Temperature for model sampling - higher values make output
                  more random, lower values more deterministic
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="maxTokens"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Max Tokens</FormLabel>
                <FormControl>
                  <Input {...field} type="number" />
                </FormControl>
                <FormDescription>
                  Maximum number of tokens to generate in the response
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
                  Timeout in seconds for model requests
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <div className="space-y-4 mt-6">
          <h2 className="text-lg font-semibold">Features</h2>
          <div className="space-y-4">
            <FormField
              control={form.control}
              name="disableThoughts"
              render={({ field }) => (
                <FormItem className="flex items-center space-x-2">
                  <FormControl>
                    <Checkbox
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                  <div className="space-y-1">
                    <FormLabel>Disable Thoughts</FormLabel>
                    <FormDescription>
                      Disable thought generation completely. Works better with
                      reasoning models like Claude-1 and Deepseek R1
                    </FormDescription>
                  </div>
                </FormItem>
              )}
            />

            {/* Add other feature checkboxes similarly */}
          </div>
        </div>

        <div className="mt-8 flex items-center justify-end gap-4">
          {error && <p className="text-sm text-red-500">{error}</p>}
          <Button type="submit" disabled={isSaving}>
            {isSaving ? "Saving..." : "Save Changes"}
          </Button>
        </div>
      </form>
    </Form>
  );
}
