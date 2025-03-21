import { Artifacts } from "@/features/trajectory/components/Artifacts.tsx";
import { Timeline as Timeline2 } from "@/features/trajectory2/timeline.tsx";
import { Timeline } from "@/features/trajectory/components/Timeline.tsx";
import { TimelineItemDetails } from "@/features/trajectory/components/TimelineItemDetails.tsx";
import { TrajectoryError } from "@/features/trajectory/components/TrajectoryError.tsx";
import { TrajectoryEvents } from "@/features/trajectory/components/TrajectoryEvents.tsx";
import { TrajectoryLogs } from "@/features/trajectory/components/TrajectoryLogs.tsx";
import { TrajectoryStatus, TrajectoryView } from "@/features/trajectory/components/TrajectoryStatus.tsx";
import { BottomPanel } from "@/features/trajectory/components/BottomPanel.tsx";
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
          return <TreeView
            trajectory={trajectory}
            treeData={treeData || {
              id: '',
              type: 'node',
              label: '',
              node_id: 0,
              timestamp: new Date().toISOString(),
              children: []
            }}
            loading={isTreeLoading}
            error={treeError}
          />
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
      <TrajectoryStatus
        trajectory={trajectory}
        handleStart={handleStart}
        handleRetry={handleRetry}
        showRightPanel={showRightPanel}
        onToggleRightPanel={toggleRightPanel}
        currentView={currentView}
        onViewChange={setCurrentView}
      />

      <ResizablePanelGroup direction="vertical" className="flex-1">
        <ResizablePanel defaultSize={70} minSize={30}>
          <ResizablePanelGroup direction="horizontal">
            <ResizablePanel
              defaultSize={showRightPanel ? 60 : 100}
              minSize={30}
              className="flex flex-col overflow-hidden"
            >
              {renderCurrentView()}
            </ResizablePanel>

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
        </ResizablePanel>

        {showBottomPanel && <ResizableHandle className="bg-border hover:bg-ring" />}

        {showBottomPanel ? (
          <ResizablePanel defaultSize={30} minSize={15}>
            <BottomPanel
              showBottomPanel={showBottomPanel}
              setShowBottomPanel={setShowBottomPanel}
              activeBottomTab={activeBottomTab}
              setActiveBottomTab={setActiveBottomTab}
              projectId={trajectory.project_id}
              trajectoryId={trajectory.trajectory_id}
              status={trajectory.status}
            />
          </ResizablePanel>
        ) : (
          <div className="flex-shrink-0">
            <BottomPanel
              showBottomPanel={showBottomPanel}
              setShowBottomPanel={setShowBottomPanel}
              activeBottomTab={activeBottomTab}
              setActiveBottomTab={setActiveBottomTab}
              projectId={trajectory.project_id}
              trajectoryId={trajectory.trajectory_id}
              status={trajectory.status}
            />
          </div>
        )}
      </ResizablePanelGroup>
    </div>
  );
}
