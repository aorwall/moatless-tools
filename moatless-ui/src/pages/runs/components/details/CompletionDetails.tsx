import { JsonView } from '@/lib/components/ui/json-view';

interface CompletionDetailsProps {
  content: {
    input?: any;
    response?: any;
    usage?: {
      promptTokens: number;
      completionTokens: number;
      cachedTokens: number;
    };
  };
}

export const CompletionDetails = ({ content }: CompletionDetailsProps) => {
  return (
    <div className="space-y-4">
      {content.usage && (
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Prompt</span>
            <span className="font-mono text-gray-900">{content.usage.promptTokens}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Completion</span>
            <span className="font-mono text-gray-900">{content.usage.completionTokens}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Cached</span>
            <span className="font-mono text-gray-900">{content.usage.cachedTokens}</span>
          </div>
        </div>
      )}

      {content.input && (
        <div>
          <div className="mb-2 text-sm font-medium text-gray-700">Input</div>
          <div className="overflow-hidden rounded-md bg-gray-50">
            <JsonView data={content.input} />
          </div>
        </div>
      )}

      {content.response && (
        <div>
          <div className="mb-2 text-sm font-medium text-gray-700">Response</div>
          <div className="overflow-hidden rounded-md bg-gray-50">
            <JsonView data={content.response} />
          </div>
        </div>
      )}
    </div>
  );
}; 