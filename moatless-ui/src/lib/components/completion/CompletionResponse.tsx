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

interface ContentItem {
  type: string;
  text: string;
  cache_control?: any;
}

interface CompletionResponseProps {
  choices: Array<{
    message: {
      content?: string | ContentItem[];
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

function isContentItemArray(content: any): content is ContentItem[] {
  if (!Array.isArray(content)) return false;
  return content.every(item => 
    item && 
    typeof item === 'object' && 
    'type' in item && 
    'text' in item
  );
}

function ContentItemView({ item, index }: { item: ContentItem; index: number }) {
  return (
    <Collapsible defaultOpen className="border rounded-lg mb-2 last:mb-0">
      <CollapsibleTrigger className="flex items-center w-full p-3 hover:bg-muted/50 rounded-lg border">
        <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200 rotate-180" />
        <Badge variant="secondary" className="ml-2">{item.type}</Badge>
        <span className="text-xs text-muted-foreground ml-2">#{index + 1}</span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="px-4 pb-3 pt-1">
          <pre className="whitespace-pre-wrap text-sm">{item.text}</pre>
          {item.cache_control && (
            <div className="mt-2 text-xs text-muted-foreground">
              Cache Control: {JSON.stringify(item.cache_control)}
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function CompletionResponse({ choices }: CompletionResponseProps) {
  const message = choices[0]?.message;
  if (!message) return null;

  return (
    <Card className="p-4">
      <div className="mb-3 text-sm font-medium">Response</div>
      <div className="space-y-2">
        {message.content && (
          <>
            {typeof message.content === 'string' ? (
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
            ) : isContentItemArray(message.content) ? (
              <div className="space-y-2">
                {message.content.map((item, index) => (
                  <ContentItemView key={index} item={item} index={index} />
                ))}
              </div>
            ) : (
              <div className="px-4 pb-3 pt-1">
                <JsonViewer data={message.content} />
              </div>
            )}
          </>
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