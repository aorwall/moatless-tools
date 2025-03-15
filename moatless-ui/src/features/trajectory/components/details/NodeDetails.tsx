import { CompletionContent } from "@/lib/components/completion/CompletionContent.tsx";
import { Trajectory } from "@/lib/types/trajectory.ts";
import { JsonViewer } from "@/lib/components/ui/json-viewer.tsx";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/lib/components/ui/tabs.tsx";
import { FileText, Code, Sparkles } from "lucide-react";
import { useState } from "react";
import { NodeCompletionContent } from "./NodeCompletionContent";

interface NodeDetailsProps {
  nodeId: number;
  trajectory: Trajectory;
}

export const NodeDetails = ({ nodeId, trajectory }: NodeDetailsProps) => {
  const node = trajectory.nodes.find((node) => node.nodeId === nodeId);
  const [activeTab, setActiveTab] = useState<string>("nodeCompletion");

  if (!node) {
    return <div>Node not found</div>;
  }

  // Create a wrapper object that matches the expected structure for CompletionContent
  const completionWrapper = node.completion
    ? {
      input: node.completion.input,
      response: node.completion.response,
      usage: node.completion.usage
        ? {
          prompt_tokens: node.completion.usage.prompt_tokens ?? 0,
          completion_tokens: node.completion.usage.completion_tokens ?? 0,
          cached_tokens: node.completion.usage.cache_read_tokens ?? 0,
        }
        : undefined,
    }
    : { input: undefined, response: undefined, usage: undefined };

  return (
    <Tabs
      value={activeTab}
      onValueChange={setActiveTab}
      className="flex h-full flex-col"
    >
      <TabsList className="grid w-full h-12 items-stretch rounded-none border-b bg-background p-0 grid-cols-3">
        <TabsTrigger
          value="nodeCompletion"
          className="rounded-none border-b-2 border-transparent px-4 data-[state=active]:border-primary data-[state=active]:bg-background hover:bg-muted/50 [&:not([data-state=active])]:hover:border-muted flex items-center gap-2"
        >
          <Sparkles className="h-4 w-4" />
          Node Completion
        </TabsTrigger>
        <TabsTrigger
          value="completion"
          className="rounded-none border-b-2 border-transparent px-4 data-[state=active]:border-primary data-[state=active]:bg-background hover:bg-muted/50 [&:not([data-state=active])]:hover:border-muted flex items-center gap-2"
        >
          <FileText className="h-4 w-4" />
          Legacy Completion
        </TabsTrigger>
        <TabsTrigger
          value="json"
          className="rounded-none border-b-2 border-transparent px-4 data-[state=active]:border-primary data-[state=active]:bg-background hover:bg-muted/50 [&:not([data-state=active])]:hover:border-muted flex items-center gap-2"
        >
          <Code className="h-4 w-4" />
          JSON
        </TabsTrigger>
      </TabsList>

      <TabsContent
        value="nodeCompletion"
        className="flex-1 p-0 m-0 data-[state=active]:flex overflow-auto"
      >
        <NodeCompletionContent trajectory={trajectory} node={node} />
      </TabsContent>

      <TabsContent
        value="completion"
        className="flex-1 p-0 m-0 data-[state=active]:flex overflow-auto"
      >
        <CompletionContent content={completionWrapper} />
      </TabsContent>

      <TabsContent
        value="json"
        className="flex-1 p-4 m-0 data-[state=active]:flex overflow-auto"
      >
        <JsonViewer data={node} />
      </TabsContent>
    </Tabs>
  );
};
