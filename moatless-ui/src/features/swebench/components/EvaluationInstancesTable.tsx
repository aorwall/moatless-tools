import { Badge } from "@/lib/components/ui/badge";
import { format, formatDuration, intervalToDuration } from "date-fns";
import { Link } from "react-router-dom";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/lib/components/ui/tooltip";
import { Evaluation } from "../api/evaluation";
import {
  ArrowDown,
  ArrowUp,
  Star,
  Users,
  Search,
  Coins,
  FileText,
  ChevronDown,
  ChevronUp,
  SlidersHorizontal,
} from "lucide-react";
import { useState, useMemo } from "react";
import { Input } from "@/lib/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { cn } from "@/lib/utils";
import { Button } from "@/lib/components/ui/button";
import { Label } from "@/lib/components/ui/label";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/lib/components/ui/collapsible";

interface EvaluationInstancesTableProps {
  evaluation: Evaluation;
}

function getDuration(
  start?: string,
  end?: string,
  isRunning?: boolean,
): string {
  if (!start) return "-";
  if (isRunning) {
    const duration = intervalToDuration({
      start: new Date(start),
      end: new Date(),
    });
    return formatDuration(duration, { format: ["minutes", "seconds"] });
  }
  if (!end) return "-";
  const duration = intervalToDuration({
    start: new Date(start),
    end: new Date(end),
  });
  return formatDuration(duration, { format: ["minutes", "seconds"] });
}

// Helper function to get appropriate background and text colors based on status
function getStatusColors(status: string) {
  const lowerStatus = status.toLowerCase();
  switch (lowerStatus) {
    case "running":
      return "bg-blue-100 text-blue-800 border-blue-200";
    case "pending":
      return "bg-yellow-100 text-yellow-800 border-yellow-200";
    case "completed":
      return "bg-green-100 text-green-800 border-green-200";
    case "error":
      return "bg-red-100 text-red-800 border-red-200";
    case "canceled":
      return "bg-gray-100 text-gray-800 border-gray-200";
    case "evaluated":
      return "bg-purple-100 text-purple-800 border-purple-200";
    case "created":
      return "bg-indigo-100 text-indigo-800 border-indigo-200";
    case "resolved":
      return "bg-green-100 text-green-800 border-green-200";
    case "failed":
      return "bg-red-100 text-red-800 border-red-200";
    default:
      return "bg-gray-100 text-gray-800 border-gray-200";
  }
}

type SortField =
  | "instance_id"
  | "status"
  | "started_at"
  | "reward"
  | "resolved_by"
  | "cost"
  | "total_tokens"
  | "prompt_tokens"
  | "completion_tokens"
  | "cache_tokens";
type SortDirection = "asc" | "desc";

interface AdvancedFilters {
  minTotalTokens: number | null;
  maxTotalTokens: number | null;
  minPromptTokens: number | null;
  maxPromptTokens: number | null;
  minCompletionTokens: number | null;
  maxCompletionTokens: number | null;
  minCacheTokens: number | null;
  maxCacheTokens: number | null;
  minReward: number | null;
  maxReward: number | null;
  minResolvedBy: number | null;
  maxResolvedBy: number | null;
}

function getDefaultAdvancedFilters(): AdvancedFilters {
  return {
    minTotalTokens: null,
    maxTotalTokens: null,
    minPromptTokens: null,
    maxPromptTokens: null,
    minCompletionTokens: null,
    maxCompletionTokens: null,
    minCacheTokens: null,
    maxCacheTokens: null,
    minReward: null,
    maxReward: null,
    minResolvedBy: null,
    maxResolvedBy: null,
  };
}

export function EvaluationInstancesTable({
  evaluation,
}: EvaluationInstancesTableProps) {
  const [sortField, setSortField] = useState<SortField>("instance_id");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");
  const [filters, setFilters] = useState<{
    status: string;
    instanceId: string;
  }>({
    status: "all",
    instanceId: "",
  });
  const [advancedFiltersOpen, setAdvancedFiltersOpen] = useState(false);
  const [advancedFilters, setAdvancedFilters] = useState<AdvancedFilters>(
    getDefaultAdvancedFilters(),
  );

  // Get unique statuses from instances
  const uniqueStatuses = useMemo(() => {
    const statuses = new Set<string>();
    evaluation.instances.forEach((instance) => {
      statuses.add(instance.status.toLowerCase());
      // Also add resolved/failed for completed instances
      if (instance.status.toLowerCase() === "completed") {
        if (instance.resolved === true) {
          statuses.add("resolved");
        } else if (instance.resolved === false) {
          statuses.add("failed");
        }
      }
    });
    return Array.from(statuses).sort();
  }, [evaluation.instances]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("asc");
    }
  };

  // Find max values for sliders
  const maxValues = useMemo(() => {
    let maxTotalTokens = 0;
    let maxPromptTokens = 0;
    let maxCompletionTokens = 0;
    let maxCacheTokens = 0;
    let maxReward = 0;
    let maxResolvedBy = 0;

    evaluation.instances.forEach((instance) => {
      const total =
        (instance.usage?.prompt_tokens || 0) +
        (instance.usage?.completion_tokens || 0);
      maxTotalTokens = Math.max(maxTotalTokens, total);
      maxPromptTokens = Math.max(
        maxPromptTokens,
        instance.usage?.prompt_tokens || 0,
      );
      maxCompletionTokens = Math.max(
        maxCompletionTokens,
        instance.usage?.completion_tokens || 0,
      );
      maxCacheTokens = Math.max(
        maxCacheTokens,
        instance.usage?.cache_read_tokens || 0,
      );
      maxReward = Math.max(maxReward, instance.reward || 0);
      maxResolvedBy = Math.max(maxResolvedBy, instance.resolved_by || 0);
    });

    return {
      maxTotalTokens: maxTotalTokens || 1000,
      maxPromptTokens: maxPromptTokens || 1000,
      maxCompletionTokens: maxCompletionTokens || 1000,
      maxCacheTokens: maxCacheTokens || 1000,
      maxReward: maxReward || 5,
      maxResolvedBy: maxResolvedBy || 10,
    };
  }, [evaluation.instances]);

  const filteredInstances = useMemo(() => {
    return evaluation.instances.filter((instance) => {
      // Basic filters
      const matchesStatus =
        filters.status === "all" ||
        instance.status.toLowerCase() === filters.status ||
        // Special handling for resolved/failed
        (filters.status === "resolved" &&
          instance.status.toLowerCase() === "completed" &&
          instance.resolved === true) ||
        (filters.status === "failed" &&
          instance.status.toLowerCase() === "completed" &&
          instance.resolved === false);

      const matchesId =
        !filters.instanceId ||
        instance.instance_id
          .toLowerCase()
          .includes(filters.instanceId.toLowerCase());

      // Advanced filters
      const totalTokens =
        (instance.usage?.prompt_tokens || 0) +
        (instance.usage?.completion_tokens || 0);

      const matchesAdvancedFilters =
        // Total tokens
        (advancedFilters.minTotalTokens === null ||
          totalTokens >= advancedFilters.minTotalTokens) &&
        (advancedFilters.maxTotalTokens === null ||
          totalTokens <= advancedFilters.maxTotalTokens) &&
        // Prompt tokens
        (advancedFilters.minPromptTokens === null ||
          (instance.usage?.prompt_tokens || 0) >=
          advancedFilters.minPromptTokens) &&
        (advancedFilters.maxPromptTokens === null ||
          (instance.usage?.prompt_tokens || 0) <=
          advancedFilters.maxPromptTokens) &&
        // Completion tokens
        (advancedFilters.minCompletionTokens === null ||
          (instance.usage?.completion_tokens || 0) >=
          advancedFilters.minCompletionTokens) &&
        (advancedFilters.maxCompletionTokens === null ||
          (instance.usage?.completion_tokens || 0) <=
          advancedFilters.maxCompletionTokens) &&
        // Cache tokens
        (advancedFilters.minCacheTokens === null ||
          (instance.usage?.cache_read_tokens || 0) >=
          advancedFilters.minCacheTokens) &&
        (advancedFilters.maxCacheTokens === null ||
          (instance.usage?.cache_read_tokens || 0) <=
          advancedFilters.maxCacheTokens) &&
        // Reward
        (advancedFilters.minReward === null ||
          (instance.reward || 0) >= advancedFilters.minReward) &&
        (advancedFilters.maxReward === null ||
          (instance.reward || 0) <= advancedFilters.maxReward) &&
        // Resolved by
        (advancedFilters.minResolvedBy === null ||
          (instance.resolved_by || 0) >= advancedFilters.minResolvedBy) &&
        (advancedFilters.maxResolvedBy === null ||
          (instance.resolved_by || 0) <= advancedFilters.maxResolvedBy);

      return matchesStatus && matchesId && matchesAdvancedFilters;
    });
  }, [evaluation.instances, filters, advancedFilters]);

  // Reset advanced filters to defaults
  const resetAdvancedFilters = () => {
    setAdvancedFilters(getDefaultAdvancedFilters());
  };

  // Update a specific filter while keeping others unchanged
  const updateAdvancedFilter = (
    key: keyof AdvancedFilters,
    value: number | null,
  ) => {
    setAdvancedFilters((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const sortedAndFilteredInstances = useMemo(() => {
    return [...filteredInstances].sort((a, b) => {
      // Sort based on the selected field
      switch (sortField) {
        case "instance_id":
          return sortDirection === "asc"
            ? a.instance_id.localeCompare(b.instance_id)
            : b.instance_id.localeCompare(a.instance_id);

        case "status":
          return sortDirection === "asc"
            ? a.status.localeCompare(b.status)
            : b.status.localeCompare(a.status);

        case "started_at": {
          if (!a.started_at) return sortDirection === "asc" ? 1 : -1;
          if (!b.started_at) return sortDirection === "asc" ? -1 : 1;
          return sortDirection === "asc"
            ? new Date(a.started_at).getTime() -
            new Date(b.started_at).getTime()
            : new Date(b.started_at).getTime() -
            new Date(a.started_at).getTime();
        }

        case "reward": {
          const aReward = a.reward ?? -1;
          const bReward = b.reward ?? -1;
          return sortDirection === "asc"
            ? aReward - bReward
            : bReward - aReward;
        }

        case "resolved_by": {
          const aResolved = a.resolved_by ?? -1;
          const bResolved = b.resolved_by ?? -1;
          return sortDirection === "asc"
            ? aResolved - bResolved
            : bResolved - aResolved;
        }

        case "cost": {
          const aCost = a.usage?.completion_cost ?? 0;
          const bCost = b.usage?.completion_cost ?? 0;
          return sortDirection === "asc" ? aCost - bCost : bCost - aCost;
        }

        case "total_tokens": {
          const aTokens =
            (a.usage?.prompt_tokens ?? 0) + (a.usage?.completion_tokens ?? 0);
          const bTokens =
            (b.usage?.prompt_tokens ?? 0) + (b.usage?.completion_tokens ?? 0);
          return sortDirection === "asc"
            ? aTokens - bTokens
            : bTokens - aTokens;
        }

        case "prompt_tokens": {
          const aTokens = a.usage?.prompt_tokens ?? 0;
          const bTokens = b.usage?.prompt_tokens ?? 0;
          return sortDirection === "asc"
            ? aTokens - bTokens
            : bTokens - aTokens;
        }

        case "completion_tokens": {
          const aTokens = a.usage?.completion_tokens ?? 0;
          const bTokens = b.usage?.completion_tokens ?? 0;
          return sortDirection === "asc"
            ? aTokens - bTokens
            : bTokens - aTokens;
        }

        case "cache_tokens": {
          const aTokens = a.usage?.cache_read_tokens ?? 0;
          const bTokens = b.usage?.cache_read_tokens ?? 0;
          return sortDirection === "asc"
            ? aTokens - bTokens
            : bTokens - aTokens;
        }

        default:
          return 0;
      }
    });
  }, [filteredInstances, sortField, sortDirection]);

  const renderSortHeader = (label: string, field: SortField) => (
    <div
      className="flex items-center gap-1 cursor-pointer"
      onClick={() => handleSort(field)}
    >
      <span>{label}</span>
      {sortField === field &&
        (sortDirection === "asc" ? (
          <ArrowUp className="h-3 w-3" />
        ) : (
          <ArrowDown className="h-3 w-3" />
        ))}
    </div>
  );

  // Helper function to format cost to 4 decimal places
  const formatCost = (cost?: number | null) => {
    if (cost === undefined || cost === null) return "-";
    return `$${cost.toFixed(4)}`;
  };

  // Helper function to format tokens with comma separators
  const formatTokens = (tokens?: number | null) => {
    if (tokens === undefined || tokens === null) return "-";
    return tokens.toLocaleString();
  };

  // Create a numeric input with label
  const NumericFilter = ({
    label,
    minKey,
    maxKey,
    min = 0,
    max,
    step = 1,
  }: {
    label: string;
    minKey: keyof AdvancedFilters;
    maxKey: keyof AdvancedFilters;
    min?: number;
    max: number;
    step?: number;
  }) => (
    <div className="space-y-2 mb-4">
      <Label>{label}</Label>
      <div className="flex items-center gap-2">
        <Input
          type="number"
          placeholder="Min"
          className="w-24"
          min={min}
          max={max}
          step={step}
          value={
            advancedFilters[minKey] === null
              ? ""
              : (advancedFilters[minKey] as number)
          }
          onChange={(e) =>
            updateAdvancedFilter(
              minKey,
              e.target.value === "" ? null : Number(e.target.value),
            )
          }
        />
        <span>to</span>
        <Input
          type="number"
          placeholder="Max"
          className="w-24"
          min={min}
          max={max}
          step={step}
          value={
            advancedFilters[maxKey] === null
              ? ""
              : (advancedFilters[maxKey] as number)
          }
          onChange={(e) =>
            updateAdvancedFilter(
              maxKey,
              e.target.value === "" ? null : Number(e.target.value),
            )
          }
        />
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4">
        <div className="flex gap-4">
          <div className="flex-1">
            <Select
              value={filters.status}
              onValueChange={(value) =>
                setFilters((prev) => ({ ...prev, status: value }))
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All statuses</SelectItem>
                {uniqueStatuses.map((status) => (
                  <SelectItem key={status} value={status}>
                    {status.charAt(0).toUpperCase() + status.slice(1)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex-1 relative">
            <Input
              type="text"
              placeholder="Search instance ID..."
              value={filters.instanceId}
              onChange={(e) =>
                setFilters((prev) => ({ ...prev, instanceId: e.target.value }))
              }
              className="w-full pl-8"
            />
            <Search className="h-4 w-4 absolute left-2.5 top-1/2 transform -translate-y-1/2 text-muted-foreground" />
          </div>
        </div>

        <Collapsible
          open={advancedFiltersOpen}
          onOpenChange={setAdvancedFiltersOpen}
          className="border rounded-md"
        >
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="flex w-full justify-between p-4">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="h-4 w-4" />
                <span>Advanced Filters</span>
              </div>
              {advancedFiltersOpen ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="px-4 pb-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <NumericFilter
                label="Total Tokens"
                minKey="minTotalTokens"
                maxKey="maxTotalTokens"
                max={maxValues.maxTotalTokens}
              />
              <NumericFilter
                label="Prompt Tokens"
                minKey="minPromptTokens"
                maxKey="maxPromptTokens"
                max={maxValues.maxPromptTokens}
              />
              <NumericFilter
                label="Completion Tokens"
                minKey="minCompletionTokens"
                maxKey="maxCompletionTokens"
                max={maxValues.maxCompletionTokens}
              />
              <NumericFilter
                label="Cache Tokens"
                minKey="minCacheTokens"
                maxKey="maxCacheTokens"
                max={maxValues.maxCacheTokens}
              />
              <NumericFilter
                label="Reward"
                minKey="minReward"
                maxKey="maxReward"
                max={maxValues.maxReward}
              />
              <NumericFilter
                label="Resolved By"
                minKey="minResolvedBy"
                maxKey="maxResolvedBy"
                max={maxValues.maxResolvedBy}
              />
            </div>
            <div className="flex justify-end mt-4">
              <Button
                variant="outline"
                size="sm"
                onClick={resetAdvancedFilters}
              >
                Reset Filters
              </Button>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>

      <div className="rounded-md border">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Instance ID", "instance_id")}
                </th>
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Status", "status")}
                </th>
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Started At", "started_at")}
                </th>
                <th className="p-2 text-left font-medium">Run Duration</th>
                <th className="p-2 text-left font-medium">Eval Duration</th>
                <th className="p-2 text-left font-medium">Iterations</th>
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Reward", "reward")}
                </th>
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Resolved By", "resolved_by")}
                </th>
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Cost", "cost")}
                </th>
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Total Tokens", "total_tokens")}
                </th>
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Prompt", "prompt_tokens")}
                </th>
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Completion", "completion_tokens")}
                </th>
                <th className="p-2 text-left font-medium">
                  {renderSortHeader("Cache", "cache_tokens")}
                </th>
                <th className="p-2 text-left font-medium">Result</th>
              </tr>
            </thead>
            <tbody>
              {sortedAndFilteredInstances.map((instance, index) => (
                <tr
                  key={instance.instance_id}
                  className={`border-b transition-colors hover:bg-muted/50 ${index % 2 === 0 ? "bg-muted/10" : ""}`}
                >
                  <td className="p-2 font-mono text-xs">
                    <Link
                      to={`/swebench/evaluation/${evaluation.evaluation_name}/${instance.instance_id}`}
                      className="text-xs font-mono hover:text-primary transition-colors"
                    >
                      {instance.instance_id}
                    </Link>
                  </td>
                  <td className="p-2">
                    <Badge
                      className={cn(
                        "text-[10px] px-1.5 py-0 border",
                        getStatusColors(instance.status),
                      )}
                    >
                      {instance.status}
                    </Badge>
                  </td>
                  <td className="p-2 text-xs text-muted-foreground">
                    {instance.started_at
                      ? format(new Date(instance.started_at), "MMM d, HH:mm:ss")
                      : "-"}
                  </td>
                  <td className="p-2 text-xs text-muted-foreground">
                    {getDuration(
                      instance.started_at,
                      instance.completed_at || instance.error_at,
                      instance.status.toLowerCase() === "running",
                    )}
                  </td>
                  <td className="p-2 text-xs text-muted-foreground">
                    {getDuration(instance.completed_at, instance.evaluated_at)}
                  </td>
                  <td className="p-2 text-xs text-muted-foreground">
                    {instance.iterations !== null &&
                      instance.iterations !== undefined ? (
                      <span className="text-xs font-medium">
                        {instance.iterations}
                      </span>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="p-2">
                    {instance.reward !== null &&
                      instance.reward !== undefined ? (
                      <div className="flex items-center space-x-1">
                        <Star className="h-3 w-3 text-yellow-500 fill-yellow-500" />
                        <span className="text-xs font-medium">
                          {instance.reward}
                        </span>
                      </div>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="p-2">
                    {instance.resolved_by !== null &&
                      instance.resolved_by !== undefined ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex items-center space-x-1">
                              <Users className="h-3 w-3 text-blue-500" />
                              <span className="text-xs font-medium">
                                {instance.resolved_by}
                              </span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent side="top">
                            <p className="text-xs">
                              Number of agents that have resolved this instance
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="p-2">
                    {instance.usage?.completion_cost ? (
                      <div className="flex items-center space-x-1">
                        <Coins className="h-3 w-3 text-emerald-500" />
                        <span className="text-xs font-medium">
                          {formatCost(instance.usage.completion_cost)}
                        </span>
                      </div>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="p-2">
                    {instance.usage ? (
                      <div className="flex items-center space-x-1">
                        <FileText className="h-3 w-3 text-indigo-500" />
                        <span className="text-xs font-medium">
                          {formatTokens(
                            (instance.usage.prompt_tokens || 0) +
                            (instance.usage.completion_tokens || 0),
                          )}
                        </span>
                      </div>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="p-2">
                    {instance.usage?.prompt_tokens ? (
                      <div className="flex items-center space-x-1">
                        <span className="text-xs font-medium">
                          {formatTokens(instance.usage.prompt_tokens)}
                        </span>
                      </div>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="p-2">
                    {instance.usage?.completion_tokens ? (
                      <div className="flex items-center space-x-1">
                        <span className="text-xs font-medium">
                          {formatTokens(instance.usage.completion_tokens)}
                        </span>
                      </div>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="p-2">
                    {instance.usage?.cache_read_tokens ? (
                      <div className="flex items-center space-x-1">
                        <span className="text-xs font-medium">
                          {formatTokens(instance.usage.cache_read_tokens)}
                        </span>
                      </div>
                    ) : (
                      "-"
                    )}
                  </td>
                  <td className="p-2">
                    {instance.status === "completed" &&
                      instance.resolved != null && (
                        <Badge
                          className={cn(
                            "text-[10px] px-1.5 py-0 border",
                            instance.resolved
                              ? "bg-green-100 text-green-800 border-green-200"
                              : "bg-red-100 text-red-800 border-red-200",
                          )}
                        >
                          {instance.resolved ? "✓" : "✗"}
                        </Badge>
                      )}
                    {instance.status === "error" && (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span>
                              <Badge className="text-[10px] px-1.5 py-0 border bg-red-100 text-red-800 border-red-200">
                                Error
                              </Badge>
                            </span>
                          </TooltipTrigger>
                          <TooltipContent side="top" className="max-w-xs">
                            <p className="text-xs whitespace-normal break-words">
                              {instance.error}
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
