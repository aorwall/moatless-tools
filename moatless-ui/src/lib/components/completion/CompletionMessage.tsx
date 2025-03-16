import { useState } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/lib/components/ui/button";
import { Badge } from "@/lib/components/ui/badge";
import { MessageSquare, InfoIcon } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/lib/components/ui/tooltip";
import { CompletionToolCall } from "./CompletionToolCall";

interface MessageContentBlock {
    type: string;
    text?: string;
    [key: string]: any;
}

interface ToolCallBlock {
    type: string;
    id?: string;
    name: string;
    arguments: any;
}

interface ToolCall {
    name: string;
    arguments: any;
}

interface CompletionMessageProps {
    role: string;
    content?: string | MessageContentBlock[] | ToolCallBlock;
    tool_calls?: ToolCall[];
    index?: number;
    isExpanded?: boolean;
    onToggleExpand?: () => void;
    disableTruncation?: boolean;
    maxLines?: number;
}

export function CompletionMessage({
    role,
    content,
    tool_calls,
    index,
    isExpanded = false,
    onToggleExpand,
    disableTruncation = false,
    maxLines = 5,
}: CompletionMessageProps) {
    // Helper to truncate text to a certain number of lines
    const truncateText = (text: string, lines: number): string => {
        if (!text) return "";
        const textLines = text.split('\n');
        if (textLines.length <= lines) return text;
        return textLines.slice(0, lines).join('\n') + '\n...';
    };

    // Helper to check if text needs truncation
    const needsTruncation = (text: string): boolean => {
        if (!text) return false;
        return text.split('\n').length > maxLines;
    };

    // Render tool call component if the role is 'tool_call'
    if (role === 'tool_call' && typeof content === 'object' && !Array.isArray(content)) {
        const toolCall = content as ToolCallBlock;
        return (
            <CompletionToolCall
                name={toolCall.name}
                id={toolCall.id}
                arguments={toolCall.arguments}
                initialExpanded={isExpanded}
            />
        );
    }

    // Process content based on type
    let displayContent = "";
    let hasBlocks = false;
    let blocks: MessageContentBlock[] = [];

    if (typeof content === 'string') {
        displayContent = content;
    } else if (Array.isArray(content)) {
        hasBlocks = true;
        blocks = content;
        // Handle Anthropic's array format
        const textBlocks = content
            .filter(block => block.type === 'text')
            .map(block => block.text);

        if (textBlocks.length > 0) {
            displayContent = textBlocks.join('\n');
        } else {
            // Show raw JSON for non-text content
            displayContent = JSON.stringify(content, null, 2);
        }
    } else {
        displayContent = JSON.stringify(content, null, 2);
    }

    const contentNeedsTruncation = needsTruncation(displayContent);
    const displayedContent = (contentNeedsTruncation && !isExpanded)
        ? truncateText(displayContent, maxLines)
        : displayContent;

    // Determine icon based on role
    const getIconForRole = () => {
        if (role === 'system') return <InfoIcon className="h-4 w-4" />;
        return <MessageSquare className="h-4 w-4" />;
    };

    // Count of tool calls to show in badge
    const toolCallCount = tool_calls?.length || 0;

    return (
        <div className="border rounded-md overflow-hidden mt-2">
            <div className={cn(
                "px-3 py-1.5 flex items-center justify-between",
                "bg-muted"
            )}>
                <div className="flex items-center gap-2">
                    {getIconForRole()}
                    <span className="text-xs font-medium capitalize">{role}</span>
                    {toolCallCount > 0 && (
                        <Badge variant="secondary" className="text-xs">
                            {toolCallCount} tool {toolCallCount === 1 ? 'call' : 'calls'}
                        </Badge>
                    )}
                </div>

                <div className="flex items-center gap-1">
                    {contentNeedsTruncation && onToggleExpand && (
                        <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 px-2 text-xs"
                            onClick={onToggleExpand}
                        >
                            {isExpanded ? "Collapse" : "Expand"}
                        </Button>
                    )}
                </div>
            </div>
            <div
                className={cn(
                    "relative p-3 text-sm whitespace-pre-wrap",
                    !disableTruncation && "max-h-[200px] overflow-auto"
                )}
            >
                {displayedContent}
            </div>

            {/* Render tool calls if present */}
            {tool_calls && tool_calls.length > 0 && (
                <div className="px-3 pb-3">
                    {tool_calls.map((toolCall, idx) => (
                        <CompletionToolCall
                            key={idx}
                            name={toolCall.name}
                            arguments={toolCall.arguments}
                            initialExpanded={false}
                        />
                    ))}
                </div>
            )}
        </div>
    );
} 