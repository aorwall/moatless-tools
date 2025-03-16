import { useState, useCallback } from "react";
import { Trajectory, Node } from "@/lib/types/trajectory";
import { JsonViewer } from "@/lib/components/ui/json-viewer";
import {
    CompletionUsage,
    CompletionSection,
    CompletionContentBlock,
    CompletionToolCall,
    CompletionMessage,
    CompletionViewToggle
} from "@/lib/components/completion";
import { Tag, WrenchIcon, Code2, MessageSquare, Info, RotateCcw, ChevronDown } from "lucide-react";
import { Skeleton } from "@/lib/components/ui/skeleton";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { useGetNodeCompletions } from "../../hooks/useGetNodeCompletions";
import { cn } from "@/lib/utils";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue
} from "@/lib/components/ui/select";
import { Badge } from "@/lib/components/ui/badge";

interface NodeCompletionContentProps {
    trajectory: Trajectory;
    nodeId: number;
    actionStep?: number;
}

interface CompletionMessage {
    role: string;
    content: string | Array<{ type: string; text?: string;[key: string]: any }>;
    tool_calls?: Array<{
        name: string;
        arguments: any;
    }>;
    [key: string]: any;
}

export function NodeCompletionContent({
    trajectory,
    nodeId,
    actionStep,
}: NodeCompletionContentProps) {
    console.log("actionStep", actionStep);
    const [expandedMessageIndices, setExpandedMessageIndices] = useState<number[]>([]);
    const [isResponseExpanded, setIsResponseExpanded] = useState(false);
    const [selectedCompletionIndex, setSelectedCompletionIndex] = useState<number | null>(null);

    const toggleMessageExpansion = useCallback((index: number) => {
        setExpandedMessageIndices(prev =>
            prev.includes(index)
                ? prev.filter(i => i !== index)
                : [...prev, index]
        );
    }, [expandedMessageIndices]);

    const isMessageExpanded = useCallback((index: number) => {
        return expandedMessageIndices.includes(index);
    }, [expandedMessageIndices]);

    const { data: completions = [], isLoading, error } = useGetNodeCompletions(
        trajectory.project_id,
        trajectory.trajectory_id,
        nodeId,
        actionStep
    );

    // Default to the latest completion (last in the array) if available
    const currentCompletionIndex = selectedCompletionIndex !== null
        ? selectedCompletionIndex
        : completions.length - 1;

    const currentCompletion = completions[currentCompletionIndex];
    const hasMultipleCompletions = completions.length > 1;

    if (isLoading) {
        return (
            <div className="space-y-4 p-4">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-5/6" />
            </div>
        );
    }

    if (error) {
        return (
            <Alert variant="destructive" className="m-4">
                <AlertDescription>
                    Failed to load completion data: {(error as Error).message}
                </AlertDescription>
            </Alert>
        );
    }

    if (!completions.length || !currentCompletion) {
        return (
            <div className="p-4 text-muted-foreground">
                No completion data available for this node.
            </div>
        );
    }

    return (
        <div className="space-y-6 p-4 max-w-full">
            {/* Completion selector dropdown if multiple completions exist */}
            {hasMultipleCompletions && (
                <div className="flex items-center gap-2">
                    <RotateCcw className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Completion attempts:</span>
                    <Select
                        value={currentCompletionIndex.toString()}
                        onValueChange={(value) => setSelectedCompletionIndex(parseInt(value))}
                    >
                        <SelectTrigger className="h-8 w-auto min-w-[150px]">
                            <SelectValue placeholder="Select attempt" />
                        </SelectTrigger>
                        <SelectContent>
                            {completions.map((_, index) => (
                                <SelectItem
                                    key={`completion-${index}`}
                                    value={index.toString()}
                                >
                                    {index === completions.length - 1
                                        ? "Latest attempt"
                                        : `Attempt ${index + 1}`}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                    <Badge variant="outline" className="ml-2">
                        {currentCompletionIndex + 1} of {completions.length}
                    </Badge>
                </div>
            )}

            {/* Response content */}
            <CompletionSection
                title="Response"
                icon={<MessageSquare className="h-4 w-4 text-muted-foreground" />}
                badgeVariant="secondary"
                showRawToggle={true}
                rawData={currentCompletion.original_output}
            >
                <div className={cn("space-y-3 max-h-[600px] overflow-auto")}>
                    {/* Message content if present */}
                    {currentCompletion.output?.content && (
                        <CompletionMessage
                            role="assistant"
                            content={currentCompletion.output.content}
                            isExpanded={isResponseExpanded}
                            onToggleExpand={() => setIsResponseExpanded(!isResponseExpanded)}
                            disableTruncation={true}
                        />
                    )}

                    {/* Tool calls if present */}
                    {currentCompletion.output?.tool_calls && (
                        <div className="space-y-1">
                            {currentCompletion.output.tool_calls.length > 1 && (
                                <div className="flex items-center gap-2 mb-2">
                                    <Code2 className="h-4 w-4 text-muted-foreground" />
                                    <span className="text-sm text-muted-foreground">Tool Calls:</span>
                                </div>
                            )}
                            {currentCompletion.output.tool_calls.map((toolCall: any, index: number) => (
                                <CompletionToolCall
                                    key={`tool-${index}`}
                                    name={toolCall.name}
                                    arguments={toolCall.arguments}
                                    initialExpanded={true}
                                />
                            ))}
                        </div>
                    )}

                    {/* If no content or tool calls */}
                    {!currentCompletion.output?.content && !currentCompletion.output?.tool_calls && currentCompletion.original_output ? (
                        <div className="bg-background border rounded-md p-4 max-h-[300px] overflow-auto">
                            <JsonViewer data={currentCompletion.original_output} />
                        </div>
                    ) : !currentCompletion.output?.content && !currentCompletion.output?.tool_calls ? (
                        <div className="text-muted-foreground italic">
                            No response content available
                        </div>
                    ) : null}
                </div>
            </CompletionSection>

            <CompletionSection
                title="Input"
                icon={<MessageSquare className="h-4 w-4 text-muted-foreground" />}
                showRawToggle={true}
                rawData={currentCompletion.original_input}
            >
                <div className="bg-background border rounded-md p-3 text-sm whitespace-pre-wrap">
                    {currentCompletion.input?.map((message: CompletionMessage, index: number) => (
                        <CompletionMessage
                            key={index}
                            role={message.role}
                            content={message.content}
                            tool_calls={message.tool_calls}
                            index={index}
                            isExpanded={isMessageExpanded(index)}
                            onToggleExpand={() => toggleMessageExpansion(index)}
                            disableTruncation={true}
                        />
                    ))}
                </div>
            </CompletionSection>

        </div>
    );
} 