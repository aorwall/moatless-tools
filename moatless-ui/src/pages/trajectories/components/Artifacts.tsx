import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { ArtifactFilters, useListArtifacts } from "@/lib/hooks/useArtifact";
import { Card } from "@/lib/components/ui/card";
import { formatDistanceToNow } from "date-fns";
import { Loader2, ChevronDown, ChevronRight, File, CheckCircle, Database } from "lucide-react";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { useState, useMemo } from "react";
import { Button } from "@/lib/components/ui/button";
import { ArrowUpDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface ArtifactsProps {
  trajectoryId: string;
}

export function Artifacts({ trajectoryId }: ArtifactsProps) {
  const [filters, setFilters] = useState<ArtifactFilters>({
    sortOrder: 'desc',
    sortBy: 'created_at'
  });
  
  const { data, isLoading, error } = useListArtifacts(trajectoryId, filters);
  const { setSelectedItem } = useTrajectoryStore();

  const artifactTypes = data?.types ?? [];
  const artifacts = data?.artifacts ?? [];

  const handleSelect = (artifact: {
    id: string;
    type: string;
    name: string | null;
    created_at: number;
  }) => {
    const fakeContent = {
      artifact_id: artifact.id,
      artifact_type: artifact.type,
      change_type: "added" as const,
      actor: "user" as const,
    };

    setSelectedItem({
      instanceId: trajectoryId,
      nodeId: -1,
      itemId: `artifact-${artifact.id}`,
      type: "artifact",
      content: fakeContent,
    });
  };

  const getArtifactIcon = (type: string) => {
    switch (type) {
      case "file":
        return <File className="w-5 h-5 text-blue-500" />;
      case "verification":
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case "receipt":
        return <Database className="w-5 h-5 text-purple-500" />;
      default:
        return <File className="w-5 h-5 text-gray-500" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-6 h-6 animate-spin" />
      </div>
    );
  }

  if (error) {
    return <div className="p-4 text-destructive">Failed to load artifacts</div>;
  }

  if (!artifacts?.length) {
    return <div className="p-4 text-muted-foreground">No artifacts found</div>;
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-4">
        <div className="flex items-center justify-between">
          {/* Type filters */}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              className={cn(
                "h-8",
                !filters.type && "bg-primary text-primary-foreground hover:bg-primary/90"
              )}
              onClick={() => setFilters(prev => ({ ...prev, type: undefined }))}
            >
              All
            </Button>
            {artifactTypes.map(type => (
              <Button
                key={type}
                variant="outline"
                size="sm"
                className={cn(
                  "h-8 flex items-center gap-2",
                  filters.type === type && "bg-primary text-primary-foreground hover:bg-primary/90",
                  filters.type && filters.type !== type && "opacity-50"
                )}
                onClick={() => setFilters(prev => ({ 
                  ...prev, 
                  type: prev.type === type ? undefined : type 
                }))}
              >
                {getArtifactIcon(type)}
                {type}
              </Button>
            ))}
          </div>

          {/* Sort controls */}
          <div className="flex gap-2 ml-4">
            <Button
              variant="outline"
              size="sm"
              className={cn(
                "h-8",
                filters.sortBy === 'created_at' && "bg-muted"
              )}
              onClick={() => setFilters(prev => ({
                ...prev,
                sortBy: 'created_at',
                sortOrder: prev.sortOrder === 'asc' ? 'desc' : 'asc'
              }))}
            >
              Date
              <ArrowUpDown className="ml-2 h-4 w-4" />
            </Button>

            <Button
              variant="outline"
              size="sm"
              className={cn(
                "h-8",
                filters.sortBy === 'name' && "bg-muted"
              )}
              onClick={() => setFilters(prev => ({
                ...prev,
                sortBy: 'name',
                sortOrder: prev.sortOrder === 'asc' ? 'desc' : 'asc'
              }))}
            >
              Name
              <ArrowUpDown className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {artifacts.map((artifact) => (
            <ArtifactItem 
              key={artifact.id} 
              artifact={artifact} 
              allArtifacts={artifacts}
              onSelect={handleSelect}
            />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}

interface ArtifactItemProps {
  artifact: {
    id: string;
    type: string;
    name: string | null;
    created_at: number;
    references?: Array<{ id: string; type: string }>;
  };
  allArtifacts: Array<any>;
  onSelect: (artifact: any) => void;
}

function ArtifactItem({ artifact, allArtifacts, onSelect }: ArtifactItemProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getArtifactIcon = (type: string) => {
    switch (type) {
      case "file":
        return <File className="w-5 h-5 text-blue-500" />;
      case "verification":
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case "receipt":
        return <Database className="w-5 h-5 text-purple-500" />;
      default:
        return <File className="w-5 h-5 text-gray-500" />;
    }
  };

  return (
    <Card className="p-4 mb-3">
      <div 
        className="flex items-center cursor-pointer" 
        onClick={() => setIsExpanded(!isExpanded)}
      >
        {isExpanded ? 
          <ChevronDown className="w-5 h-5 mr-2" /> : 
          <ChevronRight className="w-5 h-5 mr-2" />
        }
        {getArtifactIcon(artifact.type)}
        <span 
          className="ml-2 font-semibold hover:text-primary"
          onClick={(e) => {
            e.stopPropagation();
            onSelect(artifact);
          }}
        >
          {artifact.name || `${artifact.type} #${artifact.id}`}
        </span>
        <span className="ml-2 text-sm text-muted-foreground">
          ({artifact.type})
        </span>
      </div>

      {isExpanded && (
        <div className="mt-4 ml-7 space-y-3">
          <p className="text-sm text-muted-foreground">ID: {artifact.id}</p>
          <p className="text-sm text-muted-foreground">
            Created: {formatDistanceToNow(artifact.created_at, { addSuffix: true })}
          </p>
          {artifact.references && artifact.references.length > 0 && (
            <div className="mt-4">
              <p className="text-sm font-semibold mb-2">References:</p>
              <ul className="list-disc list-inside ml-2 space-y-2">
                {artifact.references.map((ref) => {
                  const refArtifact = allArtifacts.find((a) => a.id === ref.id);
                  return (
                    <li 
                      key={ref.id} 
                      className="text-sm text-muted-foreground flex items-center gap-2 cursor-pointer hover:text-primary"
                      onClick={() => refArtifact && onSelect(refArtifact)}
                    >
                      {getArtifactIcon(ref.type)}
                      <span>{refArtifact ? 
                        (refArtifact.name || refArtifact.id) : 
                        `${ref.id} (${ref.type})`}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>
      )}
    </Card>
  );
} 