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
  // Transform usage data to match the expected format in CompletionUsage
  const transformedUsage = content.usage ? {
    prompt_tokens: content.usage.promptTokens,
    completion_tokens: content.usage.completionTokens,
    cached_tokens: content.usage.cachedTokens
  } : undefined;

  return (
    <div className="space-y-4 max-w-full">
      {transformedUsage && <CompletionUsage usage={transformedUsage} />}
      
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