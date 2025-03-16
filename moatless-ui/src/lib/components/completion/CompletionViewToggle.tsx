import { Button } from "@/lib/components/ui/button";
import { FileJson, FileText } from "lucide-react";

interface CompletionViewToggleProps {
    isJsonView: boolean;
    onToggle: () => void;
}

export function CompletionViewToggle({
    isJsonView,
    onToggle,
}: CompletionViewToggleProps) {
    return (
        <div className="flex justify-end">
            <Button
                variant="outline"
                size="sm"
                onClick={onToggle}
                className="flex items-center gap-2"
            >
                {isJsonView ? (
                    <>
                        <FileText className="h-4 w-4" />
                        <span>Switch to Standard View</span>
                    </>
                ) : (
                    <>
                        <FileJson className="h-4 w-4" />
                        <span>Switch to JSON View</span>
                    </>
                )}
            </Button>
        </div>
    );
} 