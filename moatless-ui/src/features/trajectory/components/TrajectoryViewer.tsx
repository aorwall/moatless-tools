import { useStartTrajectory } from "@/features/trajectory/hooks/useStartTrajectory.ts";
import { Chat } from "@/lib/components/chat/Chat.tsx";
import { Timeline } from "@/features/trajectory/components/index.ts";
import { Button } from "@/lib/components/ui/button.tsx";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/lib/components/ui/resizable.tsx";
import { ScrollArea } from "@/lib/components/ui/scroll-area.tsx";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/lib/components/ui/tabs.tsx";
import { useWebSocketStore } from "@/lib/stores/websocketStore.ts";
import { Trajectory } from "@/lib/types/trajectory.ts";
import { cn } from "@/lib/utils.ts";
import { Artifacts } from "@/features/trajectory/components/Artifacts.tsx";
import { TimelineItemDetails } from "@/features/trajectory/components/TimelineItemDetails.tsx";
import { TrajectoryError } from "@/features/trajectory/components/TrajectoryError.tsx";
import { TrajectoryEvents } from "@/features/trajectory/components/TrajectoryEvents.tsx";
import { TrajectoryLogs } from "@/features/trajectory/components/TrajectoryLogs.tsx";
import { TrajectoryStatus } from "@/features/trajectory/components/TrajectoryStatus.tsx";
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
    <div className="h-[calc(100vh-56px)] flex flex-col">
      {/* Status Bar */}
      <TrajectoryStatus
        trajectory={trajectory}
        startInstance={startInstance || (trajectory.status !== "running" ? handleStartTrajectory : undefined)}
      />

      {/* Main Content Area with two columns */}
      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Left column: Timeline */}
        <ResizablePanel defaultSize={60} minSize={30} className="flex flex-col">
          <ResizablePanelGroup direction="vertical">
            {/* Timeline Panel */}
            <ResizablePanel
              defaultSize={showBottomPanel ? 70 : 100}
              minSize={30}
              className="flex flex-col overflow-hidden"
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

            {/* Resizable handle that only shows when bottom panel is visible */}
            {showBottomPanel && <ResizableHandle className="bg-border hover:bg-ring" />}

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

        {/* Right column: Details */}
        <ResizablePanel defaultSize={40} minSize={20}>
          <TimelineItemDetails trajectoryId={trajectory.id} trajectory={trajectory} />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
