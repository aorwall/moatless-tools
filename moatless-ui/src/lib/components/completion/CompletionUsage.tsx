import { JsonViewer } from '../ui/json-viewer';
import { ChevronDown } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/lib/components/ui/collapsible";
import { Card } from "@/lib/components/ui/card";

interface CompletionUsageProps {
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    cached_tokens: number;
  };
}

export function CompletionUsage({ usage }: CompletionUsageProps) {
  return (
    <Card className="p-4">
      <div className="mb-3 text-sm font-medium">Usage</div>
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Prompt</span>
            <span className="font-mono text-gray-900">{usage.prompt_tokens}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Completion</span>
            <span className="font-mono text-gray-900">{usage.completion_tokens}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Cached</span>
            <span className="font-mono text-gray-900">{usage.cached_tokens}</span>
          </div>
        </div>
        
        <Collapsible>
          <CollapsibleTrigger className="flex items-center w-full p-3 hover:bg-muted/50 rounded-lg border">
            <ChevronDown className="h-4 w-4 shrink-0 transition-transform duration-200" />
            <span className="ml-2 text-sm">Raw Usage Data</span>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="px-4 pb-3 pt-1">
              <JsonViewer data={usage} />
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    </Card>
  );
} 