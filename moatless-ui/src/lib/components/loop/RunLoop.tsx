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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/lib/components/ui/dialog";
import { useExpandNode } from "@/lib/hooks/useExpandNode";

interface RunLoopProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  defaultMessage?: string;
  defaultAgentId?: string;
  defaultModelId?: string;
  mode?: "new" | "expand";
  trajectoryId?: string;
  nodeId?: number;
}

export function RunLoop({
  open,
  onOpenChange,
  defaultMessage = "",
  defaultAgentId = "",
  defaultModelId = "",
  mode = "new",
  trajectoryId,
  nodeId,
}: RunLoopProps) {
  const navigate = useNavigate();
  const [selectedAgentId, setSelectedAgentId] =
    useState<string>(defaultAgentId);
  const [selectedModelId, setSelectedModelId] =
    useState<string>(defaultModelId);
  const [message, setMessage] = useState<string>(defaultMessage);
  const [attachments, setAttachments] = useState<FileList | null>(null);

  const startLoop = useStartLoop();
  const expandNode = useExpandNode();

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
      const attachmentsData = attachments
        ? await Promise.all(
            Array.from(attachments).map(async (file) => ({
              name: file.name,
              data: await toBase64(file),
            })),
          )
        : undefined;

      const params = {
        agent_id: selectedAgentId,
        model_id: selectedModelId,
        message,
        attachments: attachmentsData,
      };

      let data;
      if (mode === "expand" && trajectoryId && nodeId !== undefined) {
        data = await expandNode.mutateAsync({
          trajectoryId,
          nodeId,
          ...params,
        });
      } else {
        data = await startLoop.mutateAsync(params);
      }

      toast.success(
        mode === "expand"
          ? "Node expansion started"
          : "Loop started successfully",
      );
      onOpenChange(false);
      navigate(`/runs/${data.run_id}`);
    } catch (error: any) {
      toast.error(
        error.message ||
          `Error ${mode === "expand" ? "expanding node" : "starting loop"}`,
      );
    }
  };

  const toBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>
            {mode === "expand" && nodeId !== undefined
              ? `Expand Node ${nodeId}`
              : "Start New Loop"}
          </DialogTitle>
        </DialogHeader>
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
            <Button
              type="submit"
              disabled={startLoop.isPending || expandNode.isPending}
            >
              {startLoop.isPending || expandNode.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {mode === "expand" ? "Expanding..." : "Running..."}
                </>
              ) : mode === "expand" ? (
                "Expand Node"
              ) : (
                "Run Loop"
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
