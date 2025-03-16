import { ReactNode } from "react";
import { Button } from "@/lib/components/ui/button";
import { Plus } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface SettingsHeaderProps {
    title: string;
    description?: string;
    addButtonPath?: string;
    addButtonLabel?: string;
    children?: ReactNode;
}

export function SettingsHeader({
    title,
    description,
    addButtonPath,
    addButtonLabel = "Add New",
    children,
}: SettingsHeaderProps) {
    const navigate = useNavigate();

    return (
        <div className="flex items-center justify-between mb-6">
            <div>
                <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
                {description && (
                    <p className="text-muted-foreground mt-1">{description}</p>
                )}
            </div>
            <div className="flex items-center gap-2">
                {children}
                {addButtonPath && (
                    <Button onClick={() => navigate(addButtonPath)}>
                        <Plus className="mr-2 h-4 w-4" />
                        {addButtonLabel}
                    </Button>
                )}
            </div>
        </div>
    );
} 