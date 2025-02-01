import { useState } from 'react';
import { Button } from '@/lib/components/ui/button';
import { Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { useNavigate } from 'react-router-dom';
import { useStartValidation } from '@/lib/hooks/useSWEBench';
import { InstanceSelector } from '@/lib/components/selectors/InstanceSelector';
import { AgentSelector } from '@/lib/components/selectors/AgentSelector';
import { ModelSelector } from '@/lib/components/selectors/ModelSelector';
import { useValidationStore } from '@/stores/validationStore';
import { AlertCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';

export function ValidatePage() {
  const navigate = useNavigate();
  const startValidation = useStartValidation();
  const {
    lastUsedAgent,
    lastUsedModel,
    lastUsedInstance,
    lastSearchQuery,
    setLastUsedAgent,
    setLastUsedModel,
    setLastUsedInstance,
  } = useValidationStore();

  const [selectedAgentId, setSelectedAgentId] = useState<string>(lastUsedAgent);
  const [selectedModelId, setSelectedModelId] = useState<string>(lastUsedModel);
  const [selectedInstanceId, setSelectedInstanceId] = useState<string>(lastUsedInstance);

  // Add error state
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null); // Clear any previous errors

    try {
      const response = await startValidation.mutateAsync({
        instance_id: selectedInstanceId,
        model_id: selectedModelId,
        agent_id: selectedAgentId,
      });

      // Save the selections
      setLastUsedAgent(selectedAgentId);
      setLastUsedModel(selectedModelId);
      setLastUsedInstance(selectedInstanceId);

      toast.success('Validation started successfully');
      navigate(`/runs/${response.run_id}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start validation';
      setError(errorMessage);
      toast.error(errorMessage);
    }
  };

  if (startValidation.isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin" />
      </div>
    );
  }

  return (
    <div className="container mx-auto py-6">
      <h1 className="mb-6 text-2xl font-bold">Validate Agent</h1>
      
      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      <form onSubmit={handleSubmit} className="space-y-6">
        <InstanceSelector
          selectedInstanceId={selectedInstanceId}
          onInstanceSelect={setSelectedInstanceId}
          defaultSearchQuery={lastSearchQuery}
        />

        <div className="grid gap-6 md:grid-cols-2">
          <AgentSelector
            selectedAgentId={selectedAgentId}
            onAgentSelect={setSelectedAgentId}
          />
          <ModelSelector
            selectedModelId={selectedModelId}
            onModelSelect={setSelectedModelId}
          />
        </div>

        <div className="flex justify-end">
          <Button
            type="submit"
            disabled={!selectedAgentId || !selectedModelId || !selectedInstanceId}
          >
            Run Validation
          </Button>
        </div>
      </form>
    </div>
  );
}
