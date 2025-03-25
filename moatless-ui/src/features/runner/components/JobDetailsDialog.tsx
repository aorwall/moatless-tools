import { useState, useEffect } from "react";
import { JobInfo, JobDetails as JobDetailsType, JobDetailSection } from "../types";
import { useJobDetails } from "../hooks/useJobDetails";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from "@/lib/components/ui/dialog";
import {
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
} from "@/lib/components/ui/tabs";
import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from "@/lib/components/ui/accordion";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { Badge } from "@/lib/components/ui/badge";
import { dateTimeFormat } from "@/lib/utils/date";
import { Skeleton } from "@/lib/components/ui/skeleton";

interface JobDetailsDialogProps {
    job: JobInfo | null;
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function JobDetailsDialog({
    job,
    open,
    onOpenChange,
}: JobDetailsDialogProps) {
    const [activeTab, setActiveTab] = useState("overview");

    // Fetch job details when a job is selected and dialog is open
    const { data: jobDetails, isLoading, error } = useJobDetails(
        job?.project_id || undefined,
        job?.trajectory_id || undefined,
        open && !!job
    );

    // Reset active tab when dialog opens with a new job
    useEffect(() => {
        if (open && job) {
            setActiveTab("overview");
        }
    }, [open, job]);

    if (!job) return null;

    const formatValue = (value: any): string => {
        if (value === null || value === undefined) return "-";
        if (typeof value === "object") return JSON.stringify(value, null, 2);
        return String(value);
    };

    // Get status badge based on job status
    const getStatusBadge = (status: string) => {
        switch (status) {
            case "running":
                return (
                    <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                        {status}
                    </Badge>
                );
            case "initializing":
                return (
                    <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">
                        {status}
                    </Badge>
                );
            case "completed":
                return (
                    <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                        {status}
                    </Badge>
                );
            case "failed":
                return (
                    <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
                        {status}
                    </Badge>
                );
            case "canceled":
                return (
                    <Badge variant="outline" className="bg-orange-50 text-orange-700 border-orange-200">
                        {status}
                    </Badge>
                );
            default:
                return (
                    <Badge variant="outline" className="bg-gray-50 text-gray-700 border-gray-200">
                        {status}
                    </Badge>
                );
        }
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
                <DialogHeader>
                    <DialogTitle>Job Details: {job.id}</DialogTitle>
                    <DialogDescription>
                        <div className="flex items-center gap-3 mt-1 text-sm">
                            <div>
                                <span className="font-medium">Project:</span>{" "}
                                {job.project_id || "-"}
                            </div>
                            <div>
                                <span className="font-medium">Trajectory:</span>{" "}
                                {job.trajectory_id || "-"}
                            </div>
                            <div>
                                {getStatusBadge(job.status)}
                            </div>
                        </div>
                    </DialogDescription>
                </DialogHeader>

                {isLoading ? (
                    <div className="p-4 space-y-4">
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-3/4" />
                    </div>
                ) : error ? (
                    <div className="p-4 text-center text-red-500">
                        Failed to load job details: {error instanceof Error ? error.message : "Unknown error"}
                    </div>
                ) : jobDetails ? (
                    <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
                        <TabsList className="mx-4">
                            {jobDetails.sections.map((section) => (
                                <TabsTrigger key={section.name} value={section.name}>
                                    {section.display_name}
                                </TabsTrigger>
                            ))}
                            {jobDetails.error && <TabsTrigger value="error">Error</TabsTrigger>}
                        </TabsList>

                        <ScrollArea className="flex-1 p-4">
                            {jobDetails.sections.map((section) => (
                                <TabsContent key={section.name} value={section.name} className="space-y-4">
                                    {renderSectionContent(section)}
                                </TabsContent>
                            ))}

                            {jobDetails.error && (
                                <TabsContent value="error" className="space-y-4">
                                    <div className="border rounded-md p-4 bg-red-50">
                                        <h3 className="font-medium text-red-700">Error</h3>
                                        <pre className="mt-2 text-xs whitespace-pre-wrap text-red-900 font-mono">
                                            {jobDetails.error}
                                        </pre>
                                    </div>
                                </TabsContent>
                            )}
                        </ScrollArea>
                    </Tabs>
                ) : (
                    // Fallback to basic job info if detailed info isn't available
                    <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
                        <TabsList>
                            <TabsTrigger value="overview">Overview</TabsTrigger>
                        </TabsList>

                        <ScrollArea className="flex-1 p-4">
                            <TabsContent value="overview" className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <h3 className="font-medium text-sm">Job ID</h3>
                                        <p className="font-mono text-xs mt-1">{job.id}</p>
                                    </div>
                                    <div>
                                        <h3 className="font-medium text-sm">Status</h3>
                                        <p className="mt-1">{job.status}</p>
                                    </div>
                                    <div>
                                        <h3 className="font-medium text-sm">Enqueued At</h3>
                                        <p className="mt-1">
                                            {job.enqueued_at
                                                ? dateTimeFormat.format(new Date(job.enqueued_at))
                                                : "-"}
                                        </p>
                                    </div>
                                    <div>
                                        <h3 className="font-medium text-sm">Started At</h3>
                                        <p className="mt-1">
                                            {job.started_at
                                                ? dateTimeFormat.format(new Date(job.started_at))
                                                : "-"}
                                        </p>
                                    </div>
                                    <div>
                                        <h3 className="font-medium text-sm">Ended At</h3>
                                        <p className="mt-1">
                                            {job.ended_at
                                                ? dateTimeFormat.format(new Date(job.ended_at))
                                                : "-"}
                                        </p>
                                    </div>
                                    <div>
                                        <h3 className="font-medium text-sm">Duration</h3>
                                        <p className="mt-1">
                                            {job.started_at && job.ended_at
                                                ? formatDuration(
                                                    new Date(job.ended_at).getTime() -
                                                    new Date(job.started_at).getTime()
                                                )
                                                : job.started_at && !job.ended_at
                                                    ? "Running"
                                                    : "-"}
                                        </p>
                                    </div>
                                </div>
                            </TabsContent>
                        </ScrollArea>
                    </Tabs>
                )}
            </DialogContent>
        </Dialog>
    );
}

// Helper function to render a section's content based on its data structure
function renderSectionContent(section: JobDetailSection) {
    // Helper function to format values for display
    const formatValue = (value: any): string => {
        if (value === null || value === undefined) return "-";
        if (typeof value === "object") return JSON.stringify(value, null, 2);
        return String(value);
    };

    // If the section has items (array of objects), display them as an accordion list
    if (section.items && section.items.length > 0) {
        return (
            <Accordion type="single" collapsible className="w-full">
                {section.items.map((item, index) => (
                    <AccordionItem key={index} value={`item-${index}`}>
                        <AccordionTrigger>
                            {item.name || item.type || `Item ${index + 1}`}
                        </AccordionTrigger>
                        <AccordionContent>
                            <div className="space-y-2">
                                {Object.entries(item).map(([key, value]) => (
                                    <div key={key} className="grid grid-cols-3 text-sm">
                                        <div className="font-medium">{formatKey(key)}</div>
                                        <div className="col-span-2 font-mono text-xs whitespace-pre-wrap">
                                            {formatValue(value)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </AccordionContent>
                    </AccordionItem>
                ))}
            </Accordion>
        );
    }

    // If section has data (key-value pairs), display them in a grid
    if (section.data && Object.keys(section.data).length > 0) {
        // Special handling for logs
        if (section.name === "logs" && section.data.logs) {
            return (
                <div className="border rounded-md p-3 bg-slate-50">
                    <pre className="text-xs whitespace-pre-wrap font-mono overflow-auto max-h-96">
                        {section.data.logs}
                    </pre>
                </div>
            );
        }

        return (
            <div className="space-y-3">
                {Object.entries(section.data).map(([key, value]) => (
                    <div key={key} className="grid grid-cols-3 text-sm">
                        <div className="font-medium">{formatKey(key)}</div>
                        <div className="col-span-2 font-mono text-xs whitespace-pre-wrap">
                            {typeof value === "object" ? JSON.stringify(value, null, 2) : String(value || "-")}
                        </div>
                    </div>
                ))}
            </div>
        );
    }

    // Fallback if no data is available
    return <p className="text-muted-foreground">No data available</p>;
}

// Helper function to format keys for display
function formatKey(key: string): string {
    return key
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
}

// Helper function to format duration
function formatDuration(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
        return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
} 