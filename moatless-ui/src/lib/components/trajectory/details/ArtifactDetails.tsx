import { FC } from 'react';
import { useArtifact, usePersistArtifact } from '@/lib/hooks/useArtifact';
import { JsonViewer } from '@/lib/components/ui/json-viewer';
import { Badge } from '@/lib/components/ui/badge';
import { Skeleton } from '@/lib/components/ui/skeleton';
import { formatDate } from '@/lib/utils/date';
import { Button } from '@/lib/components/ui/button';
import { FileText } from 'lucide-react';
import { FilePreview } from '@/lib/components/artifact/FilePreview';
import { ErrorDisplay } from '@/lib/components/artifact/ErrorDisplay';
import { ContentView } from '@/lib/components/content/ContentView';

interface ArtifactDetailsProps {
  content: {
    artifact_id: string;
    artifact_type: string;
    change_type: "added" | "updated" | "removed";
    diff_details?: string;
    actor: "user" | "assistant";
  };
  trajectoryId: string;
}

export const ArtifactDetails: FC<ArtifactDetailsProps> = ({ content, trajectoryId }) => {
  const { data: artifact, isLoading, error } = useArtifact(
    trajectoryId,
    content.artifact_type,
    content.artifact_id
  );

  const { mutate: persistArtifact, isPending, error: persistError } = usePersistArtifact(
    trajectoryId,
    content.artifact_type,
    content.artifact_id
  );

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (error) {
    return <ErrorDisplay error={error} />;
  }

  if (!artifact) return null;

  const isFileArtifact = content.artifact_type === 'file';

  const getChangeTypeBadge = (status: string) => {
    switch (status) {
      case "persisted":
        return <Badge variant="outline" className="bg-green-50 text-green-700">Persisted</Badge>;
      case "updated":
        return <Badge variant="outline" className="bg-blue-50 text-blue-700">Updated</Badge>;
      case "new":
        return <Badge variant="outline" className="bg-gray-50 text-gray-700">New</Badge>;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h3 className="text-sm font-medium flex items-center gap-2">
            {isFileArtifact && (
              <FileText className="h-4 w-4 text-gray-400" />
            )}
            {artifact.name || artifact.type}
          </h3>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span>Created {formatDate(artifact.created_at)}</span>
            <span>•</span>
            <span>by {content.actor}</span>
            {isFileArtifact && artifact.data.mime_type && (
              <>
                <span>•</span>
                <span>{artifact.data.mime_type}</span>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => persistArtifact()}
            disabled={isPending || !artifact.can_persist}
          >
            {isPending ? 'Persisting...' : 'Persist'}
          </Button>
          {getChangeTypeBadge(artifact.status)}
        </div>
      </div>

      {/* Persist Error Alert */}
      {persistError && <ErrorDisplay error={persistError} />}

      {/* Content View */}
      {artifact.content && (
        <div className="space-y-2">
          <ContentView content={artifact.content} />
        </div>
      )}

      {/* File Preview */}
      {isFileArtifact && artifact.data.mime_type && artifact.data.content && (
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-700">File Preview</div>
            <FilePreview 
              mimeType={artifact.data.mime_type} 
              content={artifact.data.content}
              fileName={artifact.name || undefined}
            />
          </div>
      )}

      {/* Parsed Content for file artifacts */}
      {isFileArtifact && artifact.data.parsed_content && (
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-700">Parsed Content</div>
          <pre className="overflow-x-auto whitespace-pre-wrap rounded-md bg-gray-50 p-3 text-xs">
              {artifact.data.parsed_content}
            </pre>
          </div>
      )}

      {/* Change Details */}
      {content.diff_details && (
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-700">Changes</div>
          <pre className="overflow-x-auto whitespace-pre-wrap rounded-md bg-gray-50 p-3 text-xs">
              {content.diff_details}
            </pre>
          </div>
      )}

      {/* Artifact Data - Only show if not a file or if explicitly needed */}
      {(!isFileArtifact || process.env.NODE_ENV === 'development') && (
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-700">
            {isFileArtifact ? 'Raw File Data' : 'Artifact Data'}
          </div>
            <div className="rounded-md bg-gray-50 p-3">
              <JsonViewer data={artifact.data} />
          </div>
        </div>
      )}

      {/* References */}
      {artifact.references.length > 0 && (
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-700">References</div>
          <div className="flex flex-wrap gap-2">
            {artifact.references.map((ref) => (
              <Badge key={ref.id} variant="outline">
                {ref.type}: {ref.id}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}; 