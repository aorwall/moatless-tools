import { useGetTrajectoryLogs } from "@/features/trajectory/hooks/useGetTrajectoryLogs.ts";
import { ScrollArea } from "@/lib/components/ui/scroll-area.tsx";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/lib/components/ui/select.tsx";
import { Skeleton } from "@/lib/components/ui/skeleton.tsx";
import { Terminal } from "lucide-react";
import { useState } from "react";

interface TrajectoryLogsProps {
    projectId: string;
    trajectoryId: string;
}

export function TrajectoryLogs({ projectId, trajectoryId }: TrajectoryLogsProps) {
    const [selectedFile, setSelectedFile] = useState<string | undefined>(undefined);
    const { data, isLoading, error } = useGetTrajectoryLogs(projectId, trajectoryId, selectedFile);

    if (isLoading) {
        return (
            <div className="flex flex-col h-full p-4 space-y-4">
                <Skeleton className="h-8 w-48" />
                <Skeleton className="flex-1 w-full" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col h-full p-4 text-destructive">
                <p>Error loading logs: {error.toString()}</p>
            </div>
        );
    }

    if (!data || !data.files.length) {
        return (
            <div className="flex items-center justify-center h-full p-4 text-muted-foreground">
                <div className="flex flex-col items-center gap-2">
                    <Terminal className="h-10 w-10" />
                    <p>No log files available</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full">
            {data.files.length > 1 && (
                <div className="flex p-2 border-b items-center gap-2 flex-shrink-0">
                    <span className="text-sm text-muted-foreground">Log file:</span>
                    <Select
                        value={selectedFile || data.current_file || ""}
                        onValueChange={setSelectedFile}
                    >
                        <SelectTrigger className="w-48 h-8">
                            <SelectValue placeholder="Select log file" />
                        </SelectTrigger>
                        <SelectContent>
                            {data.files.map((file) => (
                                <SelectItem key={file.name} value={file.name}>
                                    {file.name}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            )}

            <ScrollArea className="flex-1 w-full">
                <pre className="p-4 text-xs font-mono whitespace-pre-wrap">{data.logs}</pre>
            </ScrollArea>
        </div>
    );
} 