import { useState } from "react";
import { Button } from "@/lib/components/ui/button";
import { Badge } from "@/lib/components/ui/badge";
import { JsonViewer } from "@/lib/components/ui/json-viewer";
import { WrenchIcon } from "lucide-react";

interface CompletionToolCallProps {
    name: string;
    id?: string;
    arguments: any;
    initialExpanded?: boolean;
}

export function CompletionToolCall({
    name,
    id,
    arguments: args,
    initialExpanded = false,
}: CompletionToolCallProps) {
    const [isExpanded, setIsExpanded] = useState(initialExpanded);

    // Helper to parse JSON arguments safely
    const parseJsonSafely = (jsonString: string | object): any => {
        if (typeof jsonString === 'object') return jsonString;

        try {
            return JSON.parse(jsonString);
        } catch (error) {
            return jsonString;
        }
    };

    const parsedArgs = parseJsonSafely(args);

    return (
        <div className="border rounded-md overflow-hidden mb-3">
            <div className="bg-muted px-3 py-1.5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <WrenchIcon className="h-4 w-4" />
                    <span className="text-xs font-medium">Function: <span className="font-bold">{name}</span></span>
                </div>
                <div className="flex items-center gap-2">
                    {id && (
                        <Badge variant="outline" className="text-xs">Tool ID: {id}</Badge>
                    )}
                    <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 px-2 text-xs"
                        onClick={() => setIsExpanded(!isExpanded)}
                    >
                        {isExpanded ? "Collapse" : "Expand"}
                    </Button>
                </div>
            </div>
            {isExpanded && (
                <div className="p-3 ">
                    <div className="text-xs font-medium mb-1">Arguments:</div>
                    <div className="bg-background border rounded-md p-2 bg-slate-50 dark:bg-slate-900">
                        {typeof parsedArgs === 'object' ? (
                            <JsonViewer data={parsedArgs} />
                        ) : (
                            <div className="text-sm font-mono whitespace-pre-wrap">
                                {String(args)}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
} 