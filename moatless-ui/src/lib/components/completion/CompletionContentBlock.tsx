import { useState } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/lib/components/ui/button";
import { ChevronDown, ChevronUp } from "lucide-react";

interface CompletionContentBlockProps {
    content: string;
    maxLines?: number;
    className?: string;
    disableTruncation?: boolean;
    initialExpanded?: boolean;
}

export function CompletionContentBlock({
    content,
    maxLines = 5,
    className,
    disableTruncation = false,
    initialExpanded = false,
}: CompletionContentBlockProps) {
    const [isExpanded, setIsExpanded] = useState(initialExpanded);

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

    const contentNeedsTruncation = needsTruncation(content);
    const displayedContent = (contentNeedsTruncation && !isExpanded && !disableTruncation)
        ? truncateText(content, maxLines)
        : content;

    return (
        <div
            className={cn(
                "relative bg-muted/50 rounded-md p-4 whitespace-pre-wrap",
                (!disableTruncation && content && content.length > 500) && "max-h-[400px] overflow-auto",
                className
            )}
        >
            {displayedContent}

            {contentNeedsTruncation && !disableTruncation && (
                <Button
                    size="sm"
                    variant="ghost"
                    className="absolute bottom-2 right-2 h-7 px-2"
                    onClick={() => setIsExpanded(!isExpanded)}
                >
                    {isExpanded ? (
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
    );
} 