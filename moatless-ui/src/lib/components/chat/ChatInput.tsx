import { Button } from "@/lib/components/ui/button";
import { Textarea } from "@/lib/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { useAgents } from "@/lib/hooks/useAgents";
import { useModels } from "@/lib/hooks/useModels";
import { forwardRef, useImperativeHandle, useState } from "react";
import { Send } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string, agentId: string, modelId: string) => void;
  disabled?: boolean;
  agentId?: string;
  modelId?: string;
}

export interface ChatInputRef {
  clear: () => void;
}

export const ChatInput = forwardRef<ChatInputRef, ChatInputProps>(
  function ChatInput({ onSend, disabled, agentId, modelId }, ref) {
    const [message, setMessage] = useState("");
    const [selectedAgent, setSelectedAgent] = useState<string>(agentId || "");
    const [selectedModel, setSelectedModel] = useState<string>(modelId || "");

    const { data: agents, isLoading: isLoadingAgents } = useAgents();
    const { data: models, isLoading: isLoadingModels } = useModels();

    useImperativeHandle(ref, () => ({
      clear: () => setMessage(""),
    }));

    const handleSend = () => {
      if (message.trim() && selectedAgent && selectedModel) {
        onSend(message.trim(), selectedAgent, selectedModel);
      }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    };

    return (
      <div className="flex flex-col gap-4 p-4 border-t">
        <Textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          className="min-h-[100px] resize-none"
        />

        <div className="flex gap-4">
          <Select
            value={selectedAgent}
            onValueChange={setSelectedAgent}
            disabled={isLoadingAgents}
          >
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select agent" />
            </SelectTrigger>
            <SelectContent>
              {agents?.map((agent) => (
                <SelectItem key={agent.id} value={agent.id}>
                  {agent.id}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select
            value={selectedModel}
            onValueChange={setSelectedModel}
            disabled={isLoadingModels}
          >
            <SelectTrigger className="w-[200px]">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {models?.models.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  {model.model}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            onClick={handleSend}
            disabled={
              disabled || !message.trim() || !selectedAgent || !selectedModel
            }
            className="ml-auto"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  },
);
