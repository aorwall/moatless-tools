import { CompletionContent } from "@/lib/components/completion/CompletionContent.tsx";

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
  return <CompletionContent content={content} />;
};
