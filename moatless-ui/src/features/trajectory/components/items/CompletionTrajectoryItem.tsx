
export interface CompletionTimelineContent {
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    cache_read_tokens: number;
  };
  input?: string | any;
  response?: string | any;
}

export interface CompletionTrajectoryItemProps {
  content: CompletionTimelineContent;
  header?: string;
}

export const CompletionTrajectoryItem = ({
  content,
  header,
}: CompletionTrajectoryItemProps) => {
  return (
    <div className="space-y-4">
      {/* Token Usage Summary */}
      {content.usage && (
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Prompt</span>
            <span className="font-mono text-gray-900">
              {content.usage.prompt_tokens}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Completion</span>
            <span className="font-mono text-gray-900">
              {content.usage.completion_tokens}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Cached</span>
            <span className="font-mono text-gray-900">
              {content.usage.cache_read_tokens}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
