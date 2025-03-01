import { useAgents } from "@/lib/hooks/useAgents";
import { GenericSelector, OptionType } from "@/lib/components/GenericSelector";
import { useLastUsedStore } from "@/lib/stores/lastUsedStore";
import { Skeleton } from "@/lib/components/ui/skeleton";

interface AgentSelectorProps {
  selectedAgentId: string;
  onAgentSelect: (id: string) => void;
}

export function AgentSelector({
  selectedAgentId,
  onAgentSelect,
}: AgentSelectorProps) {
  const { data: agents, isLoading } = useAgents();

  const handleSelect = (id: string) => {
    onAgentSelect(id);
  };

  if (isLoading) {
    return <Skeleton className="h-10 w-full" />;
  }

  if (!agents?.length) {
    return <div className="text-sm text-muted-foreground">No agents available</div>;
  }

  const options: OptionType[] = agents.map((agent) => ({
    id: agent.id,
    label: agent.id,
  }));

  const renderInfo = (selected: OptionType | undefined) => {
    if (!selected) return null;
    const agent = agents.find((a) => a.id === selected.id);
    if (!agent) return null;
    return (
      <>
        <p>
          <span className="font-medium">Enabled Actions:</span>{" "}
          {agent.actions.length}
        </p>
      </>
    );
  };

  return (
    <GenericSelector
      value={selectedAgentId}
      onValueChange={handleSelect}
      placeholder="Select an agent"
      options={options}
      renderAdditionalInfo={renderInfo}
    />
  );
}
