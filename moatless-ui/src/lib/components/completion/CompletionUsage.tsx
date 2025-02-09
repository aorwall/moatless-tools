interface CompletionUsageProps {
  usage: {
    promptTokens: number;
    completionTokens: number;
    cachedTokens: number;
  };
}

export function CompletionUsage({ usage }: CompletionUsageProps) {
  return (
    <div className="flex items-center gap-4 text-sm">
      <div className="flex items-center gap-2">
        <span className="font-medium text-gray-500">Prompt</span>
        <span className="font-mono text-gray-900">{usage.promptTokens}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="font-medium text-gray-500">Completion</span>
        <span className="font-mono text-gray-900">{usage.completionTokens}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="font-medium text-gray-500">Cached</span>
        <span className="font-mono text-gray-900">{usage.cachedTokens}</span>
      </div>
    </div>
  );
} 