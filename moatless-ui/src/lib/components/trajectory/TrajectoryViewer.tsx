import { MessageSquare, Clock, Package, AlertCircle } from "lucide-react";
import { Timeline } from "@/lib/components/trajectory";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/lib/components/ui/resizable";
import { TrajectoryStatus } from "@/pages/trajectories/components/TrajectoryStatus";
import { TrajectoryEvents } from "@/pages/trajectories/components/TrajectoryEvents";
import { useEffect } from "react";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { TimelineItemDetails } from "@/pages/trajectories/components/TimelineItemDetails";
import { useQueryClient } from "@tanstack/react-query";
import { useWebSocketStore } from "@/lib/stores/websocketStore";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/lib/components/ui/tabs";
import { Chat } from "@/lib/components/chat/Chat";
import { Artifacts } from "@/pages/trajectories/components/Artifacts";
import { cn } from "@/lib/utils";
import { TrajectoryError } from "@/pages/trajectories/components/TrajectoryError";
import { Trajectory } from "@/lib/types/trajectory";
import { toast } from "sonner";
import { useStartTrajectory } from "@/features/trajectories/hooks/useStartTrajectory";

interface TrajectoryViewerProps {
  trajectory: Trajectory;
  startInstance?: () => void;
}

export function TrajectoryViewer({ trajectory, startInstance }: TrajectoryViewerProps) {
  const queryClient = useQueryClient();
  const { subscribe } = useWebSocketStore();
  const startTrajectory = useStartTrajectory();

  // Function to start the trajectory
  const handleStartTrajectory = async () => {
    startTrajectory.mutate({
      projectId: trajectory.project_id,
      trajectoryId: trajectory.trajectory_id
    });
  };

  // Handle websocket subscription
  useEffect(() => {
    const unsubscribe = subscribe(`trajectory.${trajectory.id}`, (message) => {
      if (message.type === 'event') {
        queryClient.invalidateQueries({ queryKey: ["trajectory", trajectory.id] });
      }
    });

    return () => unsubscribe();
  }, [trajectory.id, subscribe, queryClient]);

  interface TabItem {
    id: string
    label: string
    icon: React.ReactNode
  }
  
  const enableTabs = false;

  const tabs: TabItem[] = [
    ...(trajectory.system_status.error ? [{
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
  ];

  return (
    <div className="h-[calc(100vh-56px)]">
      <ResizablePanelGroup
        direction="horizontal"
      >
        <ResizablePanel defaultSize={25} minSize={0} className="border-r">
          <ResizablePanelGroup direction="vertical">
            {/* Status Panel */}
            <ResizablePanel 
              defaultSize={60} 
              minSize={35}
              className="border-b"
            >
              <div className="flex h-full flex-col">
                <div className="border-b h-12 flex items-center px-4">
                  <h2 className="font-semibold">Status</h2>
                </div>
                <ScrollArea className="flex-1">
                  <TrajectoryStatus 
                    trajectory={trajectory} 
                    startInstance={startInstance || (trajectory.status !== "running" ? handleStartTrajectory : undefined)}
                  />
                </ScrollArea>
              </div>
            </ResizablePanel>

            <ResizableHandle className="bg-border hover:bg-ring" />

            {/* Events Panel */}
            <ResizablePanel 
              defaultSize={40}
              minSize={20}
            >
              <div className="flex h-full flex-col">
                <div className="border-b h-12 flex items-center px-4">
                  <h2 className="font-semibold">Events</h2>
                </div>
                <ScrollArea className="flex-1">
                  <TrajectoryEvents events={trajectory.events} />
                </ScrollArea>
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-ring" />

        {/* Middle Panel */}
        <ResizablePanel defaultSize={50} className="border-x">
          {enableTabs ? (
            <Tabs 
            defaultValue={trajectory.system_status.error ? "error" : "timeline"} 
            className="flex h-full flex-col"
          >
            <TabsList 
              className={cn(
                "grid w-full h-12 items-stretch rounded-none border-b bg-background p-0",
                trajectory.system_status.error ? "grid-cols-4" : "grid-cols-3"
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

            {trajectory.system_status.error && (
              <TabsContent 
                value="error" 
                className="flex-1 p-0 m-0 data-[state=active]:flex overflow-hidden"
              >
                <TrajectoryError trajectory={trajectory} />
              </TabsContent>
            )}

            <TabsContent 
              value="timeline" 
              className="flex-1 p-0 m-0 data-[state=active]:flex overflow-hidden"
            >
              <ScrollArea className="flex-1 w-full">
                <div className="p-6 min-w-[600px]">
                  {trajectory.nodes && (
                    <Timeline trajectory={trajectory} isRunning={trajectory.status === "running"} />
                  )}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent 
              value="chat" 
              className="flex-1 p-0 m-0 data-[state=active]:flex overflow-hidden"
            >
              <Chat trajectory={trajectory} />
            </TabsContent>

            <TabsContent 
              value="artifacts" 
              className="flex-1 p-0 m-0 data-[state=active]:flex overflow-hidden"
            >
              <Artifacts trajectoryId={trajectory.id} />
            </TabsContent>
          </Tabs> 
          ) : (
            <div className="flex h-full flex-col overflow-hidden">
              <ScrollArea className="flex-1">
                <div className="p-10 min-w-[600px]">
                  <Timeline trajectory={trajectory} isRunning={trajectory.status === "running"} />
                </div>
              </ScrollArea>
            </div>
          )}
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-ring" />

        <ResizablePanel defaultSize={25}>
          <TimelineItemDetails trajectoryId={trajectory.id} trajectory={trajectory} />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
