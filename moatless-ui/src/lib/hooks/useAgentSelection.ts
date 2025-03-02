import { useLastUsedStore } from "@/lib/stores/lastUsedStore";

export function useAgentSelection() {
  const { lastUsedAgent, setLastUsedAgent } = useLastUsedStore();

  const handleAgentSelect = (agentId: string) => {
    setLastUsedAgent(agentId);
    return agentId;
  };

  return {
    selectedAgentId: lastUsedAgent,
    onAgentSelect: handleAgentSelect,
  };
}
