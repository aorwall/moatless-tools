import { Button } from "@/lib/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/lib/components/ui/tabs";
import { ChevronDown, ChevronUp, List, Terminal } from "lucide-react";
import { TrajectoryEvents } from "./TrajectoryEvents";
import { TrajectoryLogs } from "./TrajectoryLogs";

interface BottomPanelProps {
    showBottomPanel: boolean;
    setShowBottomPanel: (show: boolean) => void;
    activeBottomTab: "events" | "logs";
    setActiveBottomTab: (tab: "events" | "logs") => void;
    projectId: string;
    trajectoryId: string;
    status: string;
}

export function BottomPanel({
    showBottomPanel,
    setShowBottomPanel,
    activeBottomTab,
    setActiveBottomTab,
    projectId,
    trajectoryId,
    status,
}: BottomPanelProps) {
    if (!showBottomPanel) {
        return (
            <div
                className="flex justify-center items-center h-7 border-t cursor-pointer bg-muted/20 hover:bg-muted/30 transition-colors"
                onClick={() => setShowBottomPanel(true)}
            >
                <Button variant="ghost" size="sm" className="h-7 py-0 px-2 hover:bg-transparent">
                    <ChevronUp className="h-4 w-4" />
                    <span className="text-xs font-medium ml-1">Show Events & Logs</span>
                </Button>
            </div>
        );
    }

    return (
        <div className="flex h-full flex-col overflow-hidden border-t">
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

                <div className="flex-1 relative overflow-hidden">
                    <TabsContent
                        value="events"
                        className="h-full data-[state=active]:flex flex-col m-0 p-0 overflow-hidden"
                    >
                        <TrajectoryEvents
                            projectId={projectId}
                            trajectoryId={trajectoryId}
                            status={status}
                        />
                    </TabsContent>

                    <TabsContent
                        value="logs"
                        className="h-full data-[state=active]:flex flex-col m-0 p-0 overflow-hidden"
                    >
                        <TrajectoryLogs
                            projectId={projectId}
                            trajectoryId={trajectoryId}
                        />
                    </TabsContent>
                </div>
            </Tabs>
        </div>
    );
} 