import { Artifacts } from "@/features/trajectory/components/Artifacts.tsx";
import { Timeline as Timeline2 } from "@/features/trajectory2/timeline.tsx";
import { Timeline } from "@/features/trajectory/components/Timeline.tsx";
import { TimelineItemDetails } from "@/features/trajectory/components/TimelineItemDetails.tsx";
import { TrajectoryError } from "@/features/trajectory/components/TrajectoryError.tsx";
import { TrajectoryEvents } from "@/features/trajectory/components/TrajectoryEvents.tsx";
import { TrajectoryLogs } from "@/features/trajectory/components/TrajectoryLogs.tsx";
import { TrajectoryStatus, TrajectoryView } from "@/features/trajectory/components/TrajectoryStatus.tsx";
import { Chat } from "@/lib/components/chat/Chat.tsx";
import { Button } from "@/lib/components/ui/button.tsx";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/lib/components/ui/resizable.tsx";
import { ScrollArea } from "@/lib/components/ui/scroll-area.tsx";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/lib/components/ui/tabs.tsx";
import { Trajectory } from "@/lib/types/trajectory.ts";
import {
  ChevronDown,
  ChevronUp,
  List,
  Terminal,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useTreeView } from "../hooks/useTreeView";
import TreeView from "./tree-view/TreeView";
interface TrajectoryViewerProps {
  trajectory: Trajectory;
  handleStart?: () => Promise<void>;
  handleRetry?: () => Promise<void>;
}

export function TrajectoryViewer({
  trajectory,
  handleStart,
  handleRetry
}: TrajectoryViewerProps) {
  const [showBottomPanel, setShowBottomPanel] = useState(() => {
    // Try to get from localStorage, default to true if not found
    const saved = localStorage.getItem("trajectoryViewerShowBottomPanel");
    return saved !== null ? saved === "true" : true;
  });
  const [showRightPanel, setShowRightPanel] = useState(() => {
    // Try to get from localStorage, default to true if not found
    const saved = localStorage.getItem("trajectoryViewerShowRightPanel");
    return saved !== null ? saved === "true" : true;
  });
  const [activeBottomTab, setActiveBottomTab] = useState<"events" | "logs">(
    () => {
      // Try to get from localStorage, default to events if not found
      const saved = localStorage.getItem("trajectoryViewerActiveTab");
      return (saved === "logs" ? "logs" : "events") as "events" | "logs";
    },
  );

  // New state for current view
  const [currentView, setCurrentView] = useState<TrajectoryView>(() => {
    const saved = localStorage.getItem("trajectoryViewerCurrentView");
    return (saved as TrajectoryView) || "tree";
  });

  const { treeData, loading: isTreeLoading, error: treeError } = useTreeView({
    projectId: trajectory.project_id,
    trajectoryId: trajectory.trajectory_id,
  });

  console.log(treeData);

  // Save preferences when they change
  useEffect(() => {
    localStorage.setItem(
      "trajectoryViewerShowBottomPanel",
      String(showBottomPanel),
    );
  }, [showBottomPanel]);

  useEffect(() => {
    localStorage.setItem(
      "trajectoryViewerShowRightPanel",
      String(showRightPanel),
    );
  }, [showRightPanel]);

  useEffect(() => {
    localStorage.setItem("trajectoryViewerActiveTab", activeBottomTab);
  }, [activeBottomTab]);

  useEffect(() => {
    localStorage.setItem("trajectoryViewerCurrentView", currentView);
  }, [currentView]);

  const toggleRightPanel = () => {
    setShowRightPanel(!showRightPanel);
  };

  // Render current view based on selection
  const renderCurrentView = () => {
    const hasError = trajectory.system_status.error;

    // Render the view content based on the current selection
    const viewContent = () => {
      switch (currentView) {
        case "timeline":
          return (
            <ScrollArea className="flex-1 w-full">
              <div className="p-6 min-w-[600px]">
                {trajectory.nodes && (
                  <Timeline
                    trajectory={trajectory}
                    isRunning={trajectory.status === "running"}
                  />
                )}
              </div>
            </ScrollArea>
          );
        case "timeline2":
          return (
            <ScrollArea className="flex-1">
              <div className="p-10 min-w-[600px]">
                <Timeline2 trajectory={trajectory} />
              </div>
            </ScrollArea>
          );
        case "chat":
          return <Chat trajectory={trajectory} />;
        case "artifacts":
          return <Artifacts trajectoryId={trajectory.trajectory_id} />;
        case "tree":
          return <TreeView trajectory={trajectory} treeData={treeData} loading={isTreeLoading} error={treeError} />
        default:
          return (
            <div className="flex items-center justify-center h-full p-4 text-muted-foreground">
              Select a view to display
            </div>
          );
      }
    };

    return (
      <div className="flex h-full flex-col overflow-hidden relative">
        {viewContent()}

        {/* Show error overlay when there's an error */}
        {hasError && (
          <div className="absolute inset-0 bg-background/90 z-10 backdrop-blur-sm flex flex-col">
            <TrajectoryError trajectory={trajectory} />
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="h-[calc(100vh-56px)] flex flex-col">
      {/* Status Bar with View Selector */}
      <TrajectoryStatus
        trajectory={trajectory}
        handleStart={handleStart}
        handleRetry={handleRetry}
        showRightPanel={showRightPanel}
        onToggleRightPanel={toggleRightPanel}
        currentView={currentView}
        onViewChange={setCurrentView}
      />

      {/* Main Content Area with two columns */}
      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Left column */}
        <ResizablePanel
          defaultSize={showRightPanel ? 60 : 100}
          minSize={30}
          className="flex flex-col"
        >
          <ResizablePanelGroup direction="vertical">
            {/* Main View Panel */}
            <ResizablePanel
              defaultSize={showBottomPanel ? 70 : 100}
              minSize={30}
              className="flex flex-col overflow-hidden"
            >
              {renderCurrentView()}
            </ResizablePanel>

            {/* Resizable handle that only shows when bottom panel is visible */}
            {showBottomPanel && (
              <ResizableHandle className="bg-border hover:bg-ring" />
            )}

            {/* Bottom Panel for Events and Logs */}
            {showBottomPanel && (
              <ResizablePanel defaultSize={30} minSize={15}>
                <div className="flex h-full flex-col overflow-hidden border-t">
                  {/* Tabs for switching between Events and Logs */}
                  <Tabs
                    value={activeBottomTab}
                    onValueChange={(value) =>
                      setActiveBottomTab(value as "events" | "logs")
                    }
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
                      <TabsContent
                        value="events"
                        className="h-full data-[state=active]:flex flex-col m-0 p-0 overflow-hidden"
                      >
                        <TrajectoryEvents
                          projectId={trajectory.project_id}
                          trajectoryId={trajectory.trajectory_id}
                          status={trajectory.status}
                        />
                      </TabsContent>

                      <TabsContent
                        value="logs"
                        className="h-full data-[state=active]:flex flex-col m-0 p-0 overflow-hidden"
                      >
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
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 py-0 px-2 hover:bg-transparent"
                >
                  <ChevronUp className="h-4 w-4" />
                  <span className="text-xs font-medium ml-1">
                    Show Events & Logs
                  </span>
                </Button>
              </div>
            )}
          </ResizablePanelGroup>
        </ResizablePanel>

        {/* Right panel and handle only shown when visible */}
        {showRightPanel && (
          <>
            <ResizableHandle className="bg-border hover:bg-ring" />
            <ResizablePanel defaultSize={40} minSize={20}>
              <TimelineItemDetails
                trajectoryId={trajectory.trajectory_id}
                trajectory={trajectory}
              />
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>
    </div>
  );
}
