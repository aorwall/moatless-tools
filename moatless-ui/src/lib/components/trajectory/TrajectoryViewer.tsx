import { useStartTrajectory } from "@/features/trajectories/hooks/useStartTrajectory";
import { Chat } from "@/lib/components/chat/Chat";
import { Timeline } from "@/lib/components/trajectory";
import { Button } from "@/lib/components/ui/button";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/lib/components/ui/resizable";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/lib/components/ui/tabs";
import { useWebSocketStore } from "@/lib/stores/websocketStore";
import { Trajectory } from "@/lib/types/trajectory";
import { cn } from "@/lib/utils";
import { Artifacts } from "@/pages/trajectories/components/Artifacts";
import { TimelineItemDetails } from "@/pages/trajectories/components/TimelineItemDetails";
import { TrajectoryError } from "@/pages/trajectories/components/TrajectoryError";
import { TrajectoryEvents } from "@/pages/trajectories/components/TrajectoryEvents";
import { TrajectoryLogs } from "@/pages/trajectories/components/TrajectoryLogs";
import { TrajectoryStatus } from "@/pages/trajectories/components/TrajectoryStatus";
import { useQueryClient } from "@tanstack/react-query";
import { AlertCircle, ChevronDown, ChevronUp, Clock, List, MessageSquare, Package, Terminal } from "lucide-react";
import { useEffect, useState } from "react";

interface TrajectoryViewerProps {
  trajectory: Trajectory;
  startInstance?: () => void;
}

export function TrajectoryViewer({ trajectory, startInstance }: TrajectoryViewerProps) {
  const queryClient = useQueryClient();
  const { subscribe } = useWebSocketStore();
  const startTrajectory = useStartTrajectory();
  const [showBottomPanel, setShowBottomPanel] = useState(() => {
    // Try to get from localStorage, default to true if not found
    const saved = localStorage.getItem('trajectoryViewerShowBottomPanel');
    return saved !== null ? saved === 'true' : true;
  });
  const [activeBottomTab, setActiveBottomTab] = useState<"events" | "logs">(() => {
    // Try to get from localStorage, default to events if not found
    const saved = localStorage.getItem('trajectoryViewerActiveTab');
    return (saved === 'logs' ? 'logs' : 'events') as "events" | "logs";
  });

  // Save preferences when they change
  useEffect(() => {
    localStorage.setItem('trajectoryViewerShowBottomPanel', String(showBottomPanel));
  }, [showBottomPanel]);

  useEffect(() => {
    localStorage.setItem('trajectoryViewerActiveTab', activeBottomTab);
  }, [activeBottomTab]);

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
        <ResizablePanel defaultSize={20} minSize={0} className="border-r">
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

        {/* Main Content Area */}
        <ResizablePanel defaultSize={60}>
          <ResizablePanelGroup direction="vertical">
            {/* Timeline Panel */}
            <ResizablePanel
              defaultSize={showBottomPanel ? 70 : 100}
              minSize={30}
              className="border-x"
            >
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

            {/* Bottom Panel for Events and Logs */}
            {showBottomPanel && (
              <ResizablePanel defaultSize={30} minSize={15}>
                <div className="flex h-full flex-col overflow-hidden border-t">
                  {/* Tabs for switching between Events and Logs */}
                  <Tabs
                    value={activeBottomTab}
                    onValueChange={(value) => setActiveBottomTab(value as "events" | "logs")}
                    className="flex flex-col h-full"
                  >
                    <div className="flex items-center justify-between border-b h-9 bg-muted/10 flex-shrink-0">
                      <TabsList className="h-full border-0 bg-transparent p-0">
                        <TabsTrigger
                          value="events"
                          className="h-full rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-4"
                        >
                          <List className="h-4 w-4 mr-2" />
                          Events
                        </TabsTrigger>
                        <TabsTrigger
                          value="logs"
                          className="h-full rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-4"
                        >
                          <Terminal className="h-4 w-4 mr-2" />
                          Logs
                        </TabsTrigger>
                      </TabsList>

                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowBottomPanel(false)}
                        className="h-8 mr-2 px-2 hover:bg-muted/20"
                      >
                        <ChevronDown className="h-4 w-4" />
                        <span className="text-xs font-medium ml-1">Hide</span>
                      </Button>
                    </div>

                    {/* Panel Content */}
                    <div className="flex-1 relative overflow-hidden">
                      <TabsContent value="events" className="h-full data-[state=active]:flex flex-col m-0 p-0 overflow-hidden">
                        <TrajectoryEvents events={trajectory.events} />
                      </TabsContent>

                      <TabsContent value="logs" className="h-full data-[state=active]:flex flex-col m-0 p-0 overflow-hidden">
                        <TrajectoryLogs
                          projectId={trajectory.project_id}
                          trajectoryId={trajectory.trajectory_id}
                        />
                      </TabsContent>
                    </div>
                  </Tabs>
                </div>
              </ResizablePanel>
            )}

            {/* Show panel button when panel is hidden */}
            {!showBottomPanel && (
              <div
                className="flex justify-center items-center h-7 border-t cursor-pointer bg-muted/20 hover:bg-muted/30 transition-colors"
                onClick={() => setShowBottomPanel(true)}
              >
                <Button variant="ghost" size="sm" className="h-7 py-0 px-2 hover:bg-transparent">
                  <ChevronUp className="h-4 w-4" />
                  <span className="text-xs font-medium ml-1">Show Events & Logs</span>
                </Button>
              </div>
            )}
          </ResizablePanelGroup>
        </ResizablePanel>

        <ResizableHandle className="bg-border hover:bg-ring" />

        <ResizablePanel defaultSize={20}>
          <TimelineItemDetails trajectoryId={trajectory.id} trajectory={trajectory} />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
