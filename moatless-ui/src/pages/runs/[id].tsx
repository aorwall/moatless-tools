import { useParams } from "react-router-dom";
import { useRun } from "@/lib/hooks/useRun";
import { Card, CardContent } from "@/lib/components/ui/card";
import { Loader2, AlertCircle } from "lucide-react";
import { Timeline } from "@/lib/components/trajectory";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/lib/components/ui/resizable";
import { RunStatus } from "./components/RunStatus";
import { RunEvents } from "./components/RunEvents";
import { useEffect } from "react";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { TimelineItemDetails } from "./components/TimelineItemDetails";
import { useQueryClient } from "@tanstack/react-query";
import { useWebSocketStore } from "@/lib/stores/websocketStore";

export function RunPage() {
  const { id } = useParams<{ id: string }>();
  const { data: runData, isError, error } = useRun(id!);
  const queryClient = useQueryClient();
  const { subscribe } = useWebSocketStore();

  useEffect(() => {
    if (!id) return;

    const unsubscribe = subscribe(`run.${id}`, () => {
      queryClient.invalidateQueries({ queryKey: ["run", id] });
    });

    return () => unsubscribe();
  }, [id, subscribe, queryClient]);

  if (isError) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load run data"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!runData) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardContent className="py-6">
            <div className="flex items-center justify-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Loading run data...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-56px)]">
      <ResizablePanelGroup
        direction="horizontal"
        className="h-full border rounded-lg"
      >
        <ResizablePanel defaultSize={25} minSize={20} className="border-r">
          <ResizablePanelGroup direction="vertical">
            {/* Status Panel */}
            <ResizablePanel defaultSize={40} className="border-b">
              <div className="flex h-full flex-col">
                <div className="border-b p-4">
                  <h2 className="font-semibold">Status</h2>
                </div>
                <ScrollArea className="flex-1">
                  <RunStatus
                    status={runData.system_status}
                    trajectory={runData.trajectory}
                  />
                </ScrollArea>
              </div>
            </ResizablePanel>

            <ResizableHandle className="bg-border hover:bg-ring" />

            {/* Events Panel */}
            <ResizablePanel defaultSize={60}>
              <div className="flex h-full flex-col">
                <div className="border-b p-4">
                  <h2 className="font-semibold">Events</h2>
                </div>
                <ScrollArea className="flex-1">
                  <RunEvents events={runData.events} />
                </ScrollArea>
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-ring" />

        {/* Timeline Panel */}
        <ResizablePanel defaultSize={50} className="border-x">
          <div className="flex h-full flex-col">
            <div className="border-b p-4">
              <h2 className="font-semibold">Timeline</h2>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-4">
                {runData.trajectory && (
                  <Timeline nodes={runData.trajectory.nodes} instanceId={id!} />
                )}
              </div>
            </ScrollArea>
          </div>
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-ring" />

        <ResizablePanel defaultSize={25}>
          <TimelineItemDetails />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
