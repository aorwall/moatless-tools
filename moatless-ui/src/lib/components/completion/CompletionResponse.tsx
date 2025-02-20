import { JsonViewer } from '../ui/json-viewer';
import { ChevronDown } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/lib/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { Badge } from "@/lib/components/ui/badge";
import { Card } from "@/lib/components/ui/card";

interface CompletionResponseProps {
  choices: Array<{
    message: {
      content?: string;
      role?: string;
      tool_calls?: Array<{
        function: {
          name: string;
          arguments: string;
        };
        id: string;
        type: string;
      }>;
    };
  }>;
}

export function CompletionResponse({ choices }: CompletionResponseProps) {
  const message = choices[0]?.message;
  if (!message) return null;

  return (
    <Card className="p-4">
      <div className="mb-3 text-sm font-medium">Response</div>
      <div className="space-y-2">
        {message.content && message.content !== "" && (
          <Collapsible defaultOpen>
            <CollapsibleTrigger className="flex items-center w-full p-3 hover:bg-muted/50 rounded-lg border">
              <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200 rotate-180" />
              <Badge variant="secondary" className="ml-2">content</Badge>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="px-4 pb-3 pt-1">
                <pre className="whitespace-pre-wrap text-sm">{message.content}</pre>
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {message.tool_calls?.map((tool, index) => (
          <Collapsible key={tool.id} defaultOpen>
            <CollapsibleTrigger className="flex items-center w-full p-3 hover:bg-muted/50 rounded-lg border">
              <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200 rotate-180" />
              <Badge variant="secondary" className="ml-2">{tool.function.name}</Badge>
              <span className="text-xs text-muted-foreground ml-2">#{index + 1}</span>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="px-4 pb-3 pt-1">
                <JsonViewer
                  data={
                    typeof tool.function.arguments === 'string'
                      ? JSON.parse(tool.function.arguments)
                      : tool.function.arguments
                  }
                />
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </div>
    </Card>
  );
} 