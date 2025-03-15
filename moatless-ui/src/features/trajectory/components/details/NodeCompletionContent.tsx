import { useState, useCallback } from "react";
import { Trajectory, Node } from "@/lib/types/trajectory";
import { JsonViewer } from "@/lib/components/ui/json-viewer";
import { CompletionUsage } from "@/lib/components/completion/CompletionUsage";
import { Button } from "@/lib/components/ui/button";
import { ChevronDown, ChevronRight, MessageSquare, Tag, ChevronUp, MinusSquare, PlusSquare, WrenchIcon, Code2 } from "lucide-react";
import { Skeleton } from "@/lib/components/ui/skeleton";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { useGetNodeCompletions } from "../../hooks/useGetNodeCompletions";
import { Badge } from "@/lib/components/ui/badge";
import { cn } from "@/lib/utils";

interface NodeCompletionContentProps {
    trajectory: Trajectory;
    node: Node;
}

interface ToolCall {
    id: string;
    type: string;
    function: {
        name: string;
        arguments: string;
    };
}

const MAX_LINES = 5; // Maximum number of lines to show by default
const MAX_HEIGHT = 300; // Maximum height for content containers in pixels

// Helper function to truncate text to a certain number of lines
const truncateText = (text: string, maxLines: number): string => {
    if (!text) return "";
    const lines = text.split('\n');
    if (lines.length <= maxLines) return text;

    return lines.slice(0, maxLines).join('\n') + '\n...';
};

// Helper to check if text needs truncation
const needsTruncation = (text: string): boolean => {
    if (!text) return false;
    return text.split('\n').length > MAX_LINES;
};

// Helper to parse JSON arguments safely
const parseJsonSafely = (jsonString: string): any => {
    try {
        return JSON.parse(jsonString);
    } catch (error) {
        return jsonString;
    }
};

export function NodeCompletionContent({
    trajectory,
    node,
}: NodeCompletionContentProps) {
    const [showInputMessages, setShowInputMessages] = useState(false);
    const [expandedMessageIndices, setExpandedMessageIndices] = useState<number[]>([]);
    const [expandedToolCallIds, setExpandedToolCallIds] = useState<string[]>([]);
    const [isResponseExpanded, setIsResponseExpanded] = useState(false);
    const [isResponseVisible, setIsResponseVisible] = useState(true);
    const [useTextTruncation, setUseTextTruncation] = useState(true);

    const toggleMessageExpansion = useCallback((index: number) => {
        setExpandedMessageIndices(prev =>
            prev.includes(index)
                ? prev.filter(i => i !== index)
                : [...prev, index]
        );
    }, []);

    const toggleToolCallExpansion = useCallback((id: string) => {
        setExpandedToolCallIds(prev =>
            prev.includes(id)
                ? prev.filter(tid => tid !== id)
                : [...prev, id]
        );
    }, []);

    const isMessageExpanded = useCallback((index: number) => {
        return expandedMessageIndices.includes(index);
    }, [expandedMessageIndices]);

    const isToolCallExpanded = useCallback((id: string) => {
        return expandedToolCallIds.includes(id);
    }, [expandedToolCallIds]);

    const { data: completion, isLoading, error } = useGetNodeCompletions(
        trajectory.project_id,
        trajectory.trajectory_id,
        node.nodeId
    );

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

    if (!completion) {
        return (
            <div className="p-4 text-muted-foreground">
                No completion data available for this node.
            </div>
        );
    }

    // Format usage data for display if it exists
    const usageData = completion.original_response?.usage
        ? {
            prompt_tokens: completion.original_response.usage.prompt_tokens,
            completion_tokens: completion.original_response.usage.completion_tokens,
            cached_tokens: 0, // Assuming cached_tokens might not exist in the original response
        }
        : undefined;

    // Get the message content and tool calls from the first choice if available
    const responseChoice = completion.original_response?.choices?.[0];
    const messageContent = responseChoice?.message?.content;
    const toolCalls = responseChoice?.message?.tool_calls as ToolCall[] | undefined;
    const hasMessageContent = !!messageContent;
    const hasToolCalls = !!toolCalls && toolCalls.length > 0;

    const responseContentNeedsTruncation = needsTruncation(messageContent || "");
    const displayedResponseContent = (responseContentNeedsTruncation && !isResponseExpanded && useTextTruncation)
        ? truncateText(messageContent || "", MAX_LINES)
        : messageContent;

    // Get the messages array from the input if available
    const inputMessages = completion.original_input?.messages || [];
    const modelName = completion.original_input?.model;

    // Render a tool call
    const renderToolCall = (toolCall: ToolCall) => {
        const isExpanded = isToolCallExpanded(toolCall.id);
        const parsedArgs = parseJsonSafely(toolCall.function.arguments);

        return (
            <div key={toolCall.id} className="border rounded-md overflow-hidden mb-3">
                <div className="bg-muted px-3 py-1.5 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <WrenchIcon className="h-4 w-4" />
                        <span className="text-xs font-medium">Function: <span className="font-bold">{toolCall.function.name}</span></span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">Tool ID: {toolCall.id}</Badge>
                        <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 px-2 text-xs"
                            onClick={() => toggleToolCallExpansion(toolCall.id)}
                        >
                            {isExpanded ? "Collapse" : "Expand"}
                        </Button>
                    </div>
                </div>
                {isExpanded && (
                    <div className="p-3 bg-slate-50 dark:bg-slate-900">
                        <div className="text-xs font-medium mb-1">Arguments:</div>
                        <div className="bg-background border rounded-md p-2">
                            {typeof parsedArgs === 'object' ? (
                                <JsonViewer data={parsedArgs} />
                            ) : (
                                <div className="text-sm font-mono whitespace-pre-wrap">
                                    {toolCall.function.arguments}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="space-y-6 p-4 max-w-full">
            {/* Model and Usage information */}
            <div className="flex flex-col space-y-2">
                {modelName && (
                    <div className="flex items-center gap-1.5">
                        <Tag className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">Model: <span className="font-semibold">{modelName}</span></span>
                    </div>
                )}
                {usageData && <CompletionUsage usage={usageData} />}
            </div>

            {/* Response content */}
            <div className="space-y-2">
                <div className="flex items-center justify-between border-b pb-1">
                    <div className="flex items-center gap-2">
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={() => setIsResponseVisible(!isResponseVisible)}
                            aria-label={isResponseVisible ? "Collapse response" : "Expand response"}
                        >
                            {isResponseVisible ? (
                                <MinusSquare className="h-4 w-4" />
                            ) : (
                                <PlusSquare className="h-4 w-4" />
                            )}
                        </Button>
                        <h3 className="text-sm font-medium text-gray-700">Response</h3>
                        {hasToolCalls && (
                            <Badge variant="secondary" className="text-xs">
                                {toolCalls?.length} tool {toolCalls?.length === 1 ? 'call' : 'calls'}
                            </Badge>
                        )}
                    </div>
                    <div className="flex items-center gap-2">
                        {(responseContentNeedsTruncation || hasToolCalls) && (
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 px-2 text-xs"
                                onClick={() => setUseTextTruncation(!useTextTruncation)}
                            >
                                {useTextTruncation ? "Show Full" : "Truncate"}
                            </Button>
                        )}
                        {responseChoice?.finish_reason && (
                            <Badge variant="outline" className="text-xs">
                                {responseChoice.finish_reason}
                            </Badge>
                        )}
                    </div>
                </div>

                {isResponseVisible && (
                    <div className={cn("space-y-3", !useTextTruncation && "max-h-[600px] overflow-auto")}>
                        {/* Message content if present */}
                        {hasMessageContent && (
                            <div
                                className={cn(
                                    "relative bg-muted/50 rounded-md p-4 whitespace-pre-wrap",
                                    !useTextTruncation && messageContent && messageContent.length > 500 && "max-h-[300px] overflow-auto"
                                )}
                            >
                                {displayedResponseContent}

                                {responseContentNeedsTruncation && useTextTruncation && (
                                    <Button
                                        size="sm"
                                        variant="ghost"
                                        className="absolute bottom-2 right-2 h-7 px-2"
                                        onClick={() => setIsResponseExpanded(!isResponseExpanded)}
                                    >
                                        {isResponseExpanded ? (
                                            <>
                                                <ChevronUp className="h-3.5 w-3.5 mr-1" /> Show Less
                                            </>
                                        ) : (
                                            <>
                                                <ChevronDown className="h-3.5 w-3.5 mr-1" /> Show More
                                            </>
                                        )}
                                    </Button>
                                )}
                            </div>
                        )}

                        {/* Tool calls if present */}
                        {hasToolCalls && (
                            <div className="space-y-1">
                                {toolCalls?.length > 1 && (
                                    <div className="flex items-center gap-2 mb-2">
                                        <Code2 className="h-4 w-4 text-muted-foreground" />
                                        <span className="text-sm text-muted-foreground">Tool Calls:</span>
                                    </div>
                                )}
                                {toolCalls?.map(renderToolCall)}
                            </div>
                        )}

                        {/* If no content or tool calls */}
                        {!hasMessageContent && !hasToolCalls && responseChoice ? (
                            <div className="bg-background border rounded-md p-4 max-h-[300px] overflow-auto">
                                <JsonViewer data={responseChoice} />
                            </div>
                        ) : !hasMessageContent && !hasToolCalls ? (
                            <div className="text-muted-foreground italic">
                                No response content available
                            </div>
                        ) : null}
                    </div>
                )}
            </div>

            {/* Input messages (collapsible) */}
            {inputMessages.length > 0 && (
                <div className="space-y-2">
                    <div className="flex items-center justify-between border-b pb-1">
                        <Button
                            variant="ghost"
                            className="flex items-center gap-2 p-0 h-auto text-sm font-medium text-gray-700"
                            onClick={() => setShowInputMessages(!showInputMessages)}
                        >
                            {showInputMessages ? (
                                <ChevronDown className="h-4 w-4" />
                            ) : (
                                <ChevronRight className="h-4 w-4" />
                            )}
                            Input Messages ({inputMessages.length})
                        </Button>

                        {showInputMessages && (
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 px-2 text-xs"
                                onClick={() => setUseTextTruncation(!useTextTruncation)}
                            >
                                {useTextTruncation ? "Show Full" : "Truncate"}
                            </Button>
                        )}
                    </div>

                    {showInputMessages && (
                        <div className="space-y-2 max-h-[600px] overflow-auto">
                            {inputMessages.map((message, index) => {
                                const isExpanded = isMessageExpanded(index);
                                const content = typeof message.content === 'string' ? message.content : JSON.stringify(message.content, null, 2);
                                const contentNeedsTruncation = needsTruncation(content);
                                const displayedContent = (contentNeedsTruncation && !isExpanded && useTextTruncation)
                                    ? truncateText(content, MAX_LINES)
                                    : content;

                                return (
                                    <div key={index} className="border rounded-md overflow-hidden">
                                        <div className="bg-muted px-3 py-1.5 flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                <MessageSquare className="h-4 w-4" />
                                                <span className="text-xs font-medium capitalize">{message.role}</span>
                                            </div>

                                            {contentNeedsTruncation && typeof message.content === 'string' && useTextTruncation && (
                                                <Button
                                                    size="sm"
                                                    variant="ghost"
                                                    className="h-6 px-2 text-xs"
                                                    onClick={() => toggleMessageExpansion(index)}
                                                >
                                                    {isExpanded ? "Collapse" : "Expand"}
                                                </Button>
                                            )}
                                        </div>
                                        <div
                                            className={cn(
                                                "relative p-3 text-sm whitespace-pre-wrap",
                                                !useTextTruncation && "max-h-[200px] overflow-auto"
                                            )}
                                        >
                                            {typeof message.content === 'string' ? (
                                                displayedContent
                                            ) : (
                                                <JsonViewer data={message.content} />
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}

            {/* If there's no messages but other input data */}
            {(!completion.original_input?.messages || completion.original_input.messages.length === 0) &&
                completion.original_input && (
                    <div className="space-y-2">
                        <div className="flex items-center justify-between border-b pb-1">
                            <Button
                                variant="ghost"
                                className="flex items-center gap-2 p-0 h-auto text-sm font-medium text-gray-700"
                                onClick={() => setShowInputMessages(!showInputMessages)}
                            >
                                {showInputMessages ? (
                                    <ChevronDown className="h-4 w-4" />
                                ) : (
                                    <ChevronRight className="h-4 w-4" />
                                )}
                                Input Data
                            </Button>
                        </div>

                        {showInputMessages && (
                            <div className="bg-background border rounded-md p-4 max-h-[300px] overflow-auto">
                                <JsonViewer data={completion.original_input} />
                            </div>
                        )}
                    </div>
                )}
        </div>
    );
} 