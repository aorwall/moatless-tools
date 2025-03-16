import { ReactNode, useState } from "react";
import { Button } from "@/lib/components/ui/button";
import { cn } from "@/lib/utils";
import { ChevronDown, ChevronRight, Code2 } from "lucide-react";
import { Badge } from "@/lib/components/ui/badge";
import { JsonViewer } from "@/lib/components/ui/json-viewer";

interface CompletionSectionProps {
    title: string;
    icon?: ReactNode;
    children: ReactNode;
    badge?: string;
    badgeVariant?: "default" | "secondary" | "destructive" | "outline";
    isCollapsible?: boolean;
    isExpanded?: boolean;
    onToggleExpand?: () => void;
    headerRightContent?: ReactNode;
    className?: string;
    contentClassName?: string;
    rawData?: any;
    showRawToggle?: boolean;
}

export function CompletionSection({
    title,
    icon,
    children,
    badge,
    badgeVariant = "secondary",
    isCollapsible = false,
    isExpanded = true,
    onToggleExpand,
    headerRightContent,
    className,
    contentClassName,
    rawData,
    showRawToggle = false,
}: CompletionSectionProps) {
    const [showRaw, setShowRaw] = useState(false);

    const toggleRawView = () => {
        setShowRaw(!showRaw);
    };

    const rawViewToggle = showRawToggle && rawData && (
        <Button
            variant="ghost"
            size="sm"
            className={cn(
                "px-2 py-1 h-6 text-xs",
                showRaw ? "bg-muted" : ""
            )}
            onClick={toggleRawView}
        >
            <Code2 className="h-3 w-3 mr-1" />
            {showRaw ? "Show Formatted" : "Show Raw"}
        </Button>
    );

    return (
        <div className={cn("space-y-2", className)}>
            <div className="flex items-center justify-between border-b pb-1">
                {isCollapsible ? (
                    <Button
                        variant="ghost"
                        className="flex items-center gap-2 p-0 h-auto text-sm font-medium text-gray-700"
                        onClick={onToggleExpand}
                    >
                        {isExpanded ? (
                            <ChevronDown className="h-4 w-4" />
                        ) : (
                            <ChevronRight className="h-4 w-4" />
                        )}
                        <div className="flex items-center gap-2">
                            {icon && icon}
                            <span>{title}</span>
                            {badge && (
                                <Badge variant={badgeVariant} className="text-xs">
                                    {badge}
                                </Badge>
                            )}
                        </div>
                    </Button>
                ) : (
                    <div className="flex items-center gap-2">
                        {icon && icon}
                        <h3 className="text-sm font-medium text-gray-700">{title}</h3>
                        {badge && (
                            <Badge variant={badgeVariant} className="text-xs">
                                {badge}
                            </Badge>
                        )}
                    </div>
                )}
                <div className="flex items-center gap-2">
                    {rawViewToggle}
                    {headerRightContent}
                </div>
            </div>

            {(!isCollapsible || isExpanded) && (
                <div className={cn(contentClassName)}>
                    {showRaw && rawData ? (
                        <div className="bg-background border rounded-md p-4 max-h-[400px] overflow-auto">
                            <JsonViewer data={rawData} />
                        </div>
                    ) : (
                        children
                    )}
                </div>
            )}
        </div>
    );
} 