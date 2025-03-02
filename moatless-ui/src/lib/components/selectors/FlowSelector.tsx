import { useFlows } from "@/lib/hooks/useFlows";
import { GenericSelector, OptionType } from "@/lib/components/GenericSelector";
import { Skeleton } from "@/lib/components/ui/skeleton";
import type { FlowConfig } from "@/lib/types/flow";

interface FlowSelectorProps {
  selectedFlowId: string;
  onFlowSelect: (id: string) => void;
}

export function FlowSelector({
  selectedFlowId,
  onFlowSelect,
}: FlowSelectorProps) {
  const { data: flows, isLoading } = useFlows();

  if (isLoading) {
    return <Skeleton className="h-10 w-full" />;
  }

  if (!flows?.length) {
    return (
      <div className="text-sm text-muted-foreground">No flows available</div>
    );
  }

  const options: OptionType[] = flows.map((flow) => ({
    id: flow.id,
    label: flow.id,
  }));

  const renderInfo = (selected: OptionType | undefined) => {
    if (!selected) return null;
    const flow = flows.find((f) => f.id === selected.id);
    if (!flow) return null;
    return (
      <>
        {flow.description && (
          <p className="text-sm text-muted-foreground mb-2">
            {flow.description}
          </p>
        )}
        <div className="text-sm grid grid-cols-2 gap-x-4 gap-y-1">
          <p>
            <span className="font-medium">Type:</span> {flow.flow_type}
          </p>
          <p>
            <span className="font-medium">Max Iterations:</span>{" "}
            {flow.max_iterations}
          </p>
          {flow.max_expansions && (
            <p>
              <span className="font-medium">Max Expansions:</span>{" "}
              {flow.max_expansions}
            </p>
          )}
          {flow.max_depth && (
            <p>
              <span className="font-medium">Max Depth:</span> {flow.max_depth}
            </p>
          )}
          <p>
            <span className="font-medium">Max Cost:</span> {flow.max_cost}
          </p>
        </div>
      </>
    );
  };

  return (
    <GenericSelector
      title="Select Flow"
      value={selectedFlowId}
      onValueChange={onFlowSelect}
      placeholder="Select a flow"
      options={options}
      renderAdditionalInfo={renderInfo}
    />
  );
}
