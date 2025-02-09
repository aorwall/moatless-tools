import { useParams } from "react-router-dom";
import { useGetTrajectory } from "@/lib/hooks/useGetTrajectory";
import { Card, CardContent } from "@/lib/components/ui/card";
import { Loader2, AlertCircle, MessageSquare, Box, GitBranch, Clock, Package } from "lucide-react";
import { Timeline } from "@/lib/components/trajectory";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/lib/components/ui/resizable";
import { TrajectoryStatus } from "../components/TrajectoryStatus";
import { TrajectoryEvents } from "../components/TrajectoryEvents";
import { useEffect } from "react";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { TimelineItemDetails } from "../components/TimelineItemDetails";
import { useQueryClient } from "@tanstack/react-query";
import { useWebSocketStore } from "@/lib/stores/websocketStore";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/lib/components/ui/tabs";
import { Chat } from "@/lib/components/chat/Chat";
import { Artifacts } from "../components/Artifacts";
import { cn } from "@/lib/utils";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { TrajectoryError } from "../components/TrajectoryError";

export function TrajectoryPage() {
  const { trajectoryId } = useParams();
  const { setTrajectoryId } = useTrajectoryStore();
  const { data: trajectoryData, isError, error } = useGetTrajectory(trajectoryId!);
  const queryClient = useQueryClient();
  const { subscribe } = useWebSocketStore();

  useEffect(() => {
    if (!trajectoryId) return;
    setTrajectoryId(trajectoryId);
    
    const unsubscribe = subscribe(`trajectory.${trajectoryId}`, () => {
      queryClient.invalidateQueries({ queryKey: ["trajectory", trajectoryId] });
    });

    return () => unsubscribe();
  }, [trajectoryId, subscribe, queryClient]);

  if (!trajectoryId) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            No trajectory id found  
          </AlertDescription>
        </Alert>
      </div>
    );
  }

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

  if (!trajectoryData) {
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

  interface TabItem {
    id: string
    label: string
    icon: React.ReactNode
  }
  
  const tabs: TabItem[] = [
    ...(trajectoryData?.system_status.error ? [{
      id: "error",
      label: "Error",
      icon: <AlertCircle className="h-4 w-4" />,
    }] : []),
    {
      id: "timeline",
      label: "Timeline",
      icon: <Clock className="h-4 w-4" />,
    },
    {
      id: "chat",
      label: "Chat",
      icon: <MessageSquare className="h-4 w-4" />,
    },
    {
      id: "artifacts",
      label: "Artifacts",
      icon: <Package className="h-4 w-4" />,
    },
  ]
  return (
    <div className="h-[calc(100vh-56px)]">
      <ResizablePanelGroup
        direction="horizontal"
        className="h-full border rounded-lg"
      >
        <ResizablePanel defaultSize={25} minSize={20} className="border-r">
          <ResizablePanelGroup direction="vertical">
            {/* Status Panel */}
            <ResizablePanel 
              defaultSize={60} 
              minSize={35}  // Ensure minimum height for status
              className="border-b"
            >
              <div className="flex h-full flex-col">
                <div className="border-b h-12 flex items-center px-4">
                  <h2 className="font-semibold">Status</h2>
                </div>
                <ScrollArea className="flex-1">
                  <TrajectoryStatus
                    trajectory={trajectoryData}
                  />
                </ScrollArea>
              </div>
            </ResizablePanel>

            <ResizableHandle className="bg-border hover:bg-ring" />

            {/* Events Panel */}
            <ResizablePanel 
              defaultSize={40}
              minSize={20}  // Allow events to be minimized
            >
              <div className="flex h-full flex-col">
                <div className="border-b h-12 flex items-center px-4">
                  <h2 className="font-semibold">Events</h2>
                </div>
                <ScrollArea className="flex-1">
                  <TrajectoryEvents events={trajectoryData.events} />
                </ScrollArea>
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-ring" />

        {/* Middle Panel */}
        <ResizablePanel defaultSize={50} className="border-x">
          <Tabs 
            defaultValue={trajectoryData?.system_status.error ? "error" : "timeline"} 
            className="flex h-full flex-col"
          >
            <TabsList 
              className={cn(
                "grid w-full h-12 items-stretch rounded-none border-b bg-background p-0",
                trajectoryData?.system_status.error ? "grid-cols-4" : "grid-cols-3"
              )}
            >
              {tabs.map((tab) => (
                <TabsTrigger
                  key={tab.id}
                  value={tab.id}
                  className={cn(
                    "rounded-none border-b-2 border-transparent px-4",
                    "data-[state=active]:border-primary data-[state=active]:bg-background",
                    "hover:bg-muted/50 [&:not([data-state=active])]:hover:border-muted",
                    "flex items-center gap-2",
                    tab.id === "error" && "text-destructive",
                  )}
                >
                  {tab.icon}
                  {tab.label}
                </TabsTrigger>
              ))}
            </TabsList>

            {trajectoryData?.system_status.error && (
              <TabsContent 
                value="error" 
                className="flex-1 p-0 m-0 data-[state=active]:flex overflow-hidden"
              >
                <TrajectoryError trajectory={trajectoryData} />
              </TabsContent>
            )}

            <TabsContent 
              value="timeline" 
              className="flex-1 p-0 m-0 data-[state=active]:flex overflow-hidden"
            >
              <ScrollArea className="flex-1 w-full">
                <div className="p-6 min-w-[600px]">
                  {trajectoryData.nodes && (
                    <Timeline nodes={trajectoryData.nodes} isRunning={trajectoryData.status === "running"} />
                  )}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent 
              value="chat" 
              className="flex-1 p-0 m-0 data-[state=active]:flex overflow-hidden"
            >
              <Chat trajectoryId={trajectoryId!} />
            </TabsContent>

            <TabsContent 
              value="artifacts" 
              className="flex-1 p-0 m-0 data-[state=active]:flex overflow-hidden"
            >
              <Artifacts trajectoryId={trajectoryId!} />
            </TabsContent>
          </Tabs>
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-ring" />

        <ResizablePanel defaultSize={25}>
          <TimelineItemDetails trajectoryId={trajectoryId!} />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
