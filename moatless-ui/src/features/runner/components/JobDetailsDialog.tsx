import { useState } from "react";
import { JobInfo } from "../types";
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

    if (!job) return null;

    const formatValue = (value: any): string => {
        if (value === null || value === undefined) return "-";
        if (typeof value === "object") return JSON.stringify(value, null, 2);
        return String(value);
    };

    const podStatus = job.metadata?.pod_status?.status;
    const podEvents = job.metadata?.pod_status?.events || [];
    const podEnvVars = job.metadata?.pod_status?.env_vars || {};
    const jobMetadata = job.metadata?.pod_status?.job_metadata || {};
    const error = job.metadata?.error;

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
                                <Badge
                                    variant="outline"
                                    className={`
                    ${job.status === "running" ? "bg-blue-50 text-blue-700 border-blue-200" : ""}
                    ${job.status === "initializing" ? "bg-yellow-50 text-yellow-700 border-yellow-200" : ""}
                    ${job.status === "completed" ? "bg-green-50 text-green-700 border-green-200" : ""}
                    ${job.status === "failed" ? "bg-red-50 text-red-700 border-red-200" : ""}
                    ${job.status === "canceled" ? "bg-orange-50 text-orange-700 border-orange-200" : ""}
                    ${job.status === "pending" ? "bg-gray-50 text-gray-700 border-gray-200" : ""}
                  `}
                                >
                                    {job.status}
                                </Badge>
                            </div>
                        </div>
                    </DialogDescription>
                </DialogHeader>

                <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
                    <TabsList>
                        <TabsTrigger value="overview">Overview</TabsTrigger>
                        <TabsTrigger value="pod">Pod Status</TabsTrigger>
                        <TabsTrigger value="events">Events</TabsTrigger>
                        <TabsTrigger value="env">Environment</TabsTrigger>
                        {Object.keys(jobMetadata).length > 0 && (
                            <TabsTrigger value="metadata">Job Metadata</TabsTrigger>
                        )}
                        {error && <TabsTrigger value="error">Error</TabsTrigger>}
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
                                {job.metadata?.pod_status?.node && (
                                    <div>
                                        <h3 className="font-medium text-sm">Node</h3>
                                        <p className="mt-1 font-mono text-xs">
                                            {job.metadata.pod_status.node}
                                        </p>
                                    </div>
                                )}
                                {job.metadata?.pod_status?.ip && (
                                    <div>
                                        <h3 className="font-medium text-sm">Pod IP</h3>
                                        <p className="mt-1 font-mono text-xs">
                                            {job.metadata.pod_status.ip}
                                        </p>
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        <TabsContent value="pod" className="space-y-4">
                            {podStatus ? (
                                <div className="space-y-4">
                                    <div>
                                        <h3 className="font-medium text-sm">Pod Name</h3>
                                        <p className="font-mono text-xs mt-1">{podStatus.name}</p>
                                    </div>
                                    <div>
                                        <h3 className="font-medium text-sm">Phase</h3>
                                        <p className="mt-1">
                                            <Badge
                                                variant="outline"
                                                className={`
                          ${podStatus.phase === "Running" ? "bg-blue-50 text-blue-700 border-blue-200" : ""}
                          ${podStatus.phase === "Succeeded" ? "bg-green-50 text-green-700 border-green-200" : ""}
                          ${podStatus.phase === "Failed" ? "bg-red-50 text-red-700 border-red-200" : ""}
                          ${podStatus.phase === "Pending" ? "bg-yellow-50 text-yellow-700 border-yellow-200" : ""}
                        `}
                                            >
                                                {podStatus.phase}
                                            </Badge>
                                        </p>
                                    </div>

                                    <Accordion type="single" collapsible className="w-full">
                                        <AccordionItem value="conditions">
                                            <AccordionTrigger>Pod Conditions</AccordionTrigger>
                                            <AccordionContent>
                                                <div className="space-y-2">
                                                    {podStatus.conditions.map((condition: any, i: number) => (
                                                        <div
                                                            key={i}
                                                            className="border rounded-md p-3 text-sm"
                                                        >
                                                            <div className="flex justify-between">
                                                                <span className="font-medium">
                                                                    {condition.type}
                                                                </span>
                                                                <Badge
                                                                    variant="outline"
                                                                    className={
                                                                        condition.status === "True"
                                                                            ? "bg-green-50 text-green-700 border-green-200"
                                                                            : "bg-red-50 text-red-700 border-red-200"
                                                                    }
                                                                >
                                                                    {condition.status}
                                                                </Badge>
                                                            </div>
                                                            {condition.message && (
                                                                <p className="mt-2 text-xs">
                                                                    {condition.message}
                                                                </p>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            </AccordionContent>
                                        </AccordionItem>

                                        <AccordionItem value="containers">
                                            <AccordionTrigger>Container Status</AccordionTrigger>
                                            <AccordionContent>
                                                <div className="space-y-4">
                                                    {podStatus.container_statuses.map((container: any, i: number) => (
                                                        <div
                                                            key={i}
                                                            className="border rounded-md p-3 text-sm"
                                                        >
                                                            <div className="flex justify-between items-center">
                                                                <span className="font-medium">
                                                                    {container.name}
                                                                </span>
                                                                <Badge
                                                                    variant="outline"
                                                                    className={
                                                                        container.ready
                                                                            ? "bg-green-50 text-green-700 border-green-200"
                                                                            : "bg-yellow-50 text-yellow-700 border-yellow-200"
                                                                    }
                                                                >
                                                                    {container.ready ? "Ready" : "Not Ready"}
                                                                </Badge>
                                                            </div>

                                                            <div className="mt-2">
                                                                <p className="text-xs">
                                                                    <span className="font-medium">Restarts:</span>{" "}
                                                                    {container.restart_count}
                                                                </p>
                                                            </div>

                                                            {container.state && (
                                                                <div className="mt-3 space-y-2">
                                                                    <h4 className="font-medium text-xs">Current State</h4>
                                                                    <div className="border rounded p-2 bg-slate-50">
                                                                        <p className="font-mono text-xs whitespace-pre">
                                                                            {formatValue(container.state)}
                                                                        </p>
                                                                    </div>
                                                                </div>
                                                            )}

                                                            {container.last_state && (
                                                                <div className="mt-3 space-y-2">
                                                                    <h4 className="font-medium text-xs">Last State</h4>
                                                                    <div className="border rounded p-2 bg-slate-50">
                                                                        <p className="font-mono text-xs whitespace-pre">
                                                                            {formatValue(container.last_state)}
                                                                        </p>
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            </AccordionContent>
                                        </AccordionItem>
                                    </Accordion>
                                </div>
                            ) : (
                                <p className="text-muted-foreground">Pod status not available</p>
                            )}
                        </TabsContent>

                        <TabsContent value="events" className="space-y-4">
                            {podEvents.length > 0 ? (
                                <div className="space-y-2">
                                    {podEvents.map((event: any, i: number) => (
                                        <div key={i} className="border rounded-md p-3 text-sm">
                                            <div className="flex justify-between">
                                                <span className="font-medium">{event.reason}</span>
                                                <Badge
                                                    variant="outline"
                                                    className={
                                                        event.type === "Normal"
                                                            ? "bg-green-50 text-green-700 border-green-200"
                                                            : "bg-amber-50 text-amber-700 border-amber-200"
                                                    }
                                                >
                                                    {event.type}
                                                </Badge>
                                            </div>
                                            <p className="mt-1 text-xs">{event.message}</p>
                                            <div className="mt-2 flex items-center text-xs text-muted-foreground">
                                                <span>
                                                    {event.last_timestamp
                                                        ? dateTimeFormat.format(
                                                            new Date(event.last_timestamp)
                                                        )
                                                        : "-"}
                                                </span>
                                                {event.count > 1 && (
                                                    <span className="ml-2">
                                                        Occurred {event.count} times
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-muted-foreground">No events available</p>
                            )}
                        </TabsContent>

                        <TabsContent value="env" className="space-y-4">
                            {Object.keys(podEnvVars).length > 0 ? (
                                <div className="border rounded-md overflow-hidden">
                                    <table className="w-full">
                                        <thead className="bg-muted">
                                            <tr>
                                                <th className="px-4 py-2 text-sm font-medium text-left">
                                                    Name
                                                </th>
                                                <th className="px-4 py-2 text-sm font-medium text-left">
                                                    Value
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {Object.entries(podEnvVars).map(([key, value]) => (
                                                <tr key={key} className="border-t">
                                                    <td className="px-4 py-2 text-sm font-mono">
                                                        {key}
                                                    </td>
                                                    <td className="px-4 py-2 text-sm font-mono">
                                                        {String(value)}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            ) : (
                                <p className="text-muted-foreground">
                                    No environment variables available
                                </p>
                            )}
                        </TabsContent>

                        {Object.keys(jobMetadata).length > 0 && (
                            <TabsContent value="metadata" className="space-y-4">
                                <div className="border rounded-md overflow-hidden">
                                    <table className="w-full">
                                        <thead className="bg-muted">
                                            <tr>
                                                <th className="px-4 py-2 text-sm font-medium text-left">
                                                    Property
                                                </th>
                                                <th className="px-4 py-2 text-sm font-medium text-left">
                                                    Value
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {Object.entries(jobMetadata).map(([key, value]) => (
                                                <tr key={key} className="border-t">
                                                    <td className="px-4 py-2 text-sm font-medium">
                                                        {key}
                                                    </td>
                                                    <td className="px-4 py-2 text-sm font-mono">
                                                        {String(value)}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </TabsContent>
                        )}

                        {error && (
                            <TabsContent value="error" className="space-y-4">
                                <div className="border rounded-md p-4 bg-red-50">
                                    <h3 className="font-medium text-red-700">Error</h3>
                                    <pre className="mt-2 text-xs whitespace-pre-wrap text-red-900 font-mono">
                                        {error}
                                    </pre>
                                </div>
                            </TabsContent>
                        )}
                    </ScrollArea>
                </Tabs>
            </DialogContent>
        </Dialog>
    );
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