import { CompletionResponse } from './CompletionResponse';
import { CompletionInput } from './CompletionInput';
import { CompletionUsage } from './CompletionUsage';

interface CompletionContentProps {
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

export function CompletionContent({ content }: CompletionContentProps) {
  return (
    <div className="space-y-4 max-w-full">
      {content.usage && <CompletionUsage usage={content.usage} />}
      
      {content.response && (
        <div>
          <div className="mb-2 text-sm font-medium text-gray-700">Response</div>
          <CompletionResponse choices={content.response.choices} />
        </div>
      )}

      {content.input && <CompletionInput input={content.input} />}
    </div>
  );
} 