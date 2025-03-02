import { useState } from "react";
import { useBaseModels } from "@/lib/hooks/useModels";
import type { ModelConfig } from "@/lib/types/model";
import { Loader2, Plus, Search, Settings } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { Button } from "@/lib/components/ui/button";
import { Input } from "@/lib/components/ui/input";
import { AddModelDialog } from "./AddModelDialog";
import { useNavigate } from "react-router-dom";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";

export function BaseModelsList() {
  const navigate = useNavigate();
  const { data: baseModels, isLoading, error } = useBaseModels();
  const [searchQuery, setSearchQuery] = useState("");
  const [formatFilter, setFormatFilter] = useState<string>("all");
  const [selectedModel, setSelectedModel] = useState<ModelConfig | null>(null);

  if (isLoading) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="m-4">
        <AlertTitle>Error Loading Models</AlertTitle>
        <AlertDescription>
          {error instanceof Error
            ? error.message
            : "Failed to load base models"}
        </AlertDescription>
      </Alert>
    );
  }

  const filteredModels = baseModels?.models.filter((model) => {
    const matchesSearch = model.model
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const matchesFormat =
      formatFilter === "all" || model.response_format === formatFilter;
    return matchesSearch && matchesFormat;
  });

  return (
    <div className="flex flex-col h-full">
      <div className="flex-none border-b px-6 py-4">
        <h1 className="text-2xl font-bold">Add Model</h1>
        <p className="mt-1 text-sm text-gray-500">
          Create a new model configuration
        </p>
      </div>

      <div className="flex flex-col gap-6 p-6 border-b bg-gray-50/50">
        <div className="flex flex-col gap-2">
          <h2 className="text-lg font-semibold">Create from scratch</h2>
          <p className="text-sm text-gray-500">
            Configure a new model with custom settings
          </p>
          <Button
            className="mt-2 w-full sm:w-auto"
            onClick={() => navigate("/settings/models/create")}
          >
            <Settings className="h-4 w-4 mr-2" />
            Create Custom Model
          </Button>
        </div>

        <div className="flex flex-col gap-2">
          <h2 className="text-lg font-semibold">Start from a base model</h2>
          <p className="text-sm text-gray-500">
            Choose a pre-configured base model as a starting point
          </p>
          <div className="flex items-center gap-4 mt-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
              <Input
                placeholder="Search models..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={formatFilter} onValueChange={setFormatFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Response format" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All formats</SelectItem>
                <SelectItem value="tool_call">Tool Call</SelectItem>
                <SelectItem value="react">React</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredModels?.map((model) => (
            <div
              key={model.id}
              className="flex flex-col rounded-lg border bg-card text-card-foreground shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="flex flex-col space-y-1.5 p-6">
                <h3 className="font-semibold text-lg">{model.model}</h3>
                <p className="text-sm text-gray-500">ID: {model.id}</p>
              </div>
              <div className="p-6 pt-0 space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500">Response Format</span>
                    <span className="font-medium">{model.response_format}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500">History Type</span>
                    <span className="font-medium">
                      {model.message_history_type}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500">Temperature</span>
                    <span className="font-medium">
                      {model.temperature ?? "Default"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500">Max Tokens</span>
                    <span className="font-medium">
                      {model.max_tokens ?? "Default"}
                    </span>
                  </div>
                </div>
                <Button
                  className="w-full"
                  onClick={() => setSelectedModel(model)}
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add to Models
                </Button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {selectedModel && (
        <AddModelDialog
          baseModel={selectedModel}
          open={!!selectedModel}
          onOpenChange={(open) => !open && setSelectedModel(null)}
        />
      )}
    </div>
  );
}
