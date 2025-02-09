import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/lib/components/ui/card";
import { Input } from "@/lib/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { useSWEBenchInstances } from "@/lib/hooks/useSWEBench";
import { useState, useEffect } from "react";

interface InstanceSelectorProps {
  selectedInstanceId: string;
  onInstanceSelect: (id: string) => void;
  defaultSearchQuery?: string;
}

export function InstanceSelector({
  selectedInstanceId,
  onInstanceSelect,
  defaultSearchQuery = "",
}: InstanceSelectorProps) {
  const [searchQuery, setSearchQuery] = useState(defaultSearchQuery);
  const {
    data: instancesData,
    isLoading,
    refetch,
  } = useSWEBenchInstances(1, 100);

  useEffect(() => {
    refetch();
  }, [searchQuery, refetch]);

  if (isLoading) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Select SWE-Bench Instance</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="Search instances..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />

        <Select value={selectedInstanceId} onValueChange={onInstanceSelect}>
          <SelectTrigger>
            <SelectValue placeholder="Select an instance" />
          </SelectTrigger>
          <SelectContent>
            {instancesData?.instances?.map((instance) => (
              <SelectItem
                key={instance.instance_id}
                value={instance.instance_id}
              >
                {instance.instance_id} (Resolved by: {instance.resolved_count})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {selectedInstanceId && instancesData && (
          <div className="mt-4 space-y-2 text-sm">
            <p className="font-medium">Problem Statement:</p>
            <p className="text-muted-foreground">
              {instancesData.instances
                .find((i) => i.instance_id === selectedInstanceId)
                ?.problem_statement.slice(0, 300)}
              {instancesData.instances.find(
                (i) => i.instance_id === selectedInstanceId,
              )?.problem_statement.length > 300
                ? "..."
                : ""}
            </p>
            <p>
              <span className="font-medium">Resolved by: </span>
              {
                instancesData.instances.find(
                  (i) => i.instance_id === selectedInstanceId,
                )?.resolved_count
              }{" "}
              agents
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
