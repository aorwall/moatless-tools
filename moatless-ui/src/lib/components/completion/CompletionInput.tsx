import { JsonViewer } from "../ui/json-viewer";
import { ChevronDown } from "lucide-react";
import { useState } from "react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/lib/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { Badge } from "@/lib/components/ui/badge";
import { Card } from "@/lib/components/ui/card";

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

interface CompletionInputProps {
  input: any;
}

function isMessageArray(input: any): input is Message[] {
  if (!Array.isArray(input)) return false;
  return input.every(
    (msg) =>
      msg &&
      typeof msg === "object" &&
      "role" in msg &&
      "content" in msg &&
      ["system", "user", "assistant"].includes(msg.role),
  );
}

function MessageView({ message, index }: { message: Message; index: number }) {
  const [isOpen, setIsOpen] = useState(false);

  const badgeVariants = {
    system: "bg-secondary text-secondary-foreground",
    user: "bg-secondary text-secondary-foreground",
    assistant: "bg-secondary text-secondary-foreground",
  };

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className="border rounded-lg mb-2 last:mb-0"
    >
      <CollapsibleTrigger className="flex items-center w-full p-3 hover:bg-muted/50">
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 transition-transform duration-200",
            isOpen ? "rotate-180" : "rotate-0",
          )}
        />
        <Badge
          variant="secondary"
          className={cn("ml-2", badgeVariants[message.role])}
        >
          {message.role}
        </Badge>
        <span className="text-xs text-muted-foreground ml-2">#{index + 1}</span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="px-4 pb-3 pt-1">
          <div className="whitespace-pre-wrap text-sm">{message.content}</div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function CompletionInput({ input }: CompletionInputProps) {
  if (isMessageArray(input)) {
    return (
      <Card className="p-4">
        <div className="mb-3 text-sm font-medium">Messages</div>
        <div>
          {input.map((message, index) => (
            <MessageView key={index} message={message} index={index} />
          ))}
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-4">
      <div className="mb-3 text-sm font-medium">Input</div>
      <div className="overflow-hidden rounded-md bg-muted/50">
        <JsonViewer data={input} />
      </div>
    </Card>
  );
}
