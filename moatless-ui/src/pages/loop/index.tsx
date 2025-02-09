import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/lib/components/ui/button";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { AgentSelector } from "@/lib/components/selectors/AgentSelector";
import { ModelSelector } from "@/lib/components/selectors/ModelSelector";
import { Input } from "@/lib/components/ui/input";
import { Textarea } from "@/lib/components/ui/textarea";
import { useStartLoop } from "@/lib/hooks/useSWEBench";
import { useLoopStore } from "./stores/loopStore";

export function LoopPage() {
  const navigate = useNavigate();
  const {
    lastUsedAgent,
    lastUsedModel,
    lastMessage,
    setLastUsedAgent,
    setLastUsedModel,
    setLastMessage,
  } = useLoopStore();
  const [selectedAgentId, setSelectedAgentId] = useState<string>(lastUsedAgent);
  const [selectedModelId, setSelectedModelId] = useState<string>(lastUsedModel);
  const [message, setMessage] = useState<string>(lastMessage);
  const [attachments, setAttachments] = useState<FileList | null>(null);

  const startLoop = useStartLoop();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setAttachments(e.target.files);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedAgentId || !selectedModelId || !message) {
      toast.error("Please fill all required fields.");
      return;
    }
    try {
      setLastUsedAgent(selectedAgentId);
      setLastUsedModel(selectedModelId);
      setLastMessage(message);

      const attachmentsData = attachments ? await Promise.all(Array.from(attachments).map(async (file) => ({
        name: file.name,
        data: await toBase64(file),
      }))) : undefined;

      const data = await startLoop.mutateAsync({
        agent_id: selectedAgentId,
        model_id: selectedModelId,
        message,
        attachments: attachmentsData,
      });
      toast.success("Loop started successfully");
      navigate(`/trajectories/${data.run_id}`);
    } catch (error: any) {
      toast.error(error.message || "Error starting loop");
    }
  };

  // Helper function to convert file to base64
  const toBase64 = (file: File): Promise<string> => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  return (
    <div className="container mx-auto py-6 h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto">
        <h1 className="mb-6 text-2xl font-bold">Start Loop</h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <label className="block mb-2 font-medium">Agent</label>
              <AgentSelector
                selectedAgentId={selectedAgentId}
                onAgentSelect={setSelectedAgentId}
              />
            </div>
            <div>
              <label className="block mb-2 font-medium">Model</label>
              <ModelSelector
                selectedModelId={selectedModelId}
                onModelSelect={setSelectedModelId}
              />
            </div>
          </div>
          <div>
            <label className="block mb-2 font-medium">Message</label>
            <Textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Enter your message..."
              className="w-full"
              rows={4}
            />
          </div>
          <div>
            <label className="block mb-2 font-medium">Attachments</label>
            <Input type="file" multiple onChange={handleFileChange} />
          </div>
          <div className="flex justify-end">
            <Button type="submit" disabled={startLoop.isPending}>
              {startLoop.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                "Run Loop"
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default LoopPage;
