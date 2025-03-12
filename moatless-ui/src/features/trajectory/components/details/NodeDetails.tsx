import { CompletionContent } from "@/lib/components/completion/CompletionContent.tsx";
import { Trajectory } from "@/lib/types/trajectory.ts";
import { JsonViewer } from "@/lib/components/ui/json-viewer.tsx";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/lib/components/ui/tabs.tsx";
import { FileText, Code } from "lucide-react";
import { useState } from "react";

interface NodeDetailsProps {
  nodeId: number;
  trajectory: Trajectory;
}

export const NodeDetails = ({ nodeId, trajectory }: NodeDetailsProps) => {
  const node = trajectory.nodes.find((node) => node.nodeId === nodeId);
  const [activeTab, setActiveTab] = useState<string>("completion");

  if (!node) {
    return <div>Node not found</div>;
  }

  return (
    <Tabs
      value={activeTab}
      onValueChange={setActiveTab}
      className="flex h-full flex-col"
    >
      <TabsList className="grid w-full h-12 items-stretch rounded-none border-b bg-background p-0 grid-cols-2">
        <TabsTrigger
          value="completion"
          className="rounded-none border-b-2 border-transparent px-4 data-[state=active]:border-primary data-[state=active]:bg-background hover:bg-muted/50 [&:not([data-state=active])]:hover:border-muted flex items-center gap-2"
        >
          <FileText className="h-4 w-4" />
          Completion
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
        value="completion"
        className="flex-1 p-0 m-0 data-[state=active]:flex overflow-auto"
      >
        <CompletionContent content={node.completion} />
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
