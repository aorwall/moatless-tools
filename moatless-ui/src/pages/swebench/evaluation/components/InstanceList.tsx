import { Link } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Evaluation, EvaluationInstance } from "@/features/swebench/api/evaluation";
import { Badge } from "@/lib/components/ui/badge";
import { format } from "date-fns";
import { Input } from "@/lib/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { useState, ElementType } from "react";

interface InstanceListProps {
  evaluation: Evaluation;
  selectedInstanceId?: string;
}

export function InstanceList({ evaluation, selectedInstanceId }: InstanceListProps) {
  const [filters, setFilters] = useState<{ status: string; instanceId: string }>({
    status: "all",
    instanceId: "",
  });

  const filteredInstances = evaluation.instances.filter((instance) => {
    const matchesStatus = filters.status === "all" || instance.status.toLowerCase() === filters.status;
    const matchesId = !filters.instanceId || 
      instance.instance_id.toLowerCase().includes(filters.instanceId.toLowerCase());
    return matchesStatus && matchesId;
  });

  const getRelevantTimestamp = (instance: EvaluationInstance) => {
    if (instance.error) {
      return instance.error_at;
    } else if (instance.evaluated_at) {
      return instance.evaluated_at;
    } else if (instance.completed_at) {
      return instance.completed_at;
    } else if (instance.started_at) {
      return instance.started_at;
    } else {
      return instance.created_at;
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex-none border-b bg-gray-50/50 px-3 py-3">
        <div className="space-y-2">
          <Select
            value={filters.status}
            onValueChange={(value) => setFilters(prev => ({ ...prev, status: value }))}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="running">Running</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="error">Error</SelectItem>
            </SelectContent>
          </Select>
          <Input
            type="text"
            placeholder="Search instance ID..."
            value={filters.instanceId}
            onChange={(e) => setFilters(prev => ({ ...prev, instanceId: e.target.value }))}
            className="w-full"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {filteredInstances.map((instance) => {
          const isPending = !instance.started_at;
          const Container = isPending ? 'div' as ElementType : Link;
          const props = {
            key: instance.instance_id,
            ...(isPending ? {} : { to: `/swebench/evaluation/${evaluation.evaluation_name}/${instance.instance_id}` }),
            className: cn(
              "block border-b px-4 py-3 transition-colors",
              !isPending && "hover:bg-gray-50",
              selectedInstanceId === instance.instance_id && "bg-blue-50 hover:bg-blue-50",
              isPending && "opacity-50 cursor-not-allowed"
            )
          };

          return (
            <Container {...props}>
              <div className="flex items-start gap-3">
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium text-sm">{instance.instance_id}</div>
                  <div className="mt-1 flex items-center gap-2">
                    <div className={`status-badge status-bg-${instance.status.toLowerCase()} status-text-${instance.status.toLowerCase()}`}>
                      {instance.status}
                    </div>
                    {instance.status === "completed" && (
                      <div className={`status-badge ${instance.resolved ? "status-bg-resolved status-text-resolved" : "status-bg-failed status-text-failed"}`}>
                        {instance.resolved ? "✓" : "✗"}
                      </div>
                    )}
                    <span className="text-[10px] text-muted-foreground">
                      {getRelevantTimestamp(instance) && 
                        format(new Date(getRelevantTimestamp(instance)!), 'MMM d, HH:mm')}
                    </span>
                  </div>
                </div>
              </div>
            </Container>
          );
        })}
      </div>
    </div>
  );
} 