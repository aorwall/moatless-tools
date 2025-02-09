import { FC } from "react";
import { JsonView } from "@/lib/components/ui/json-view";

export interface CompletionTimelineContent {
  usage?: {
    promptTokens: number;
    completionTokens: number;
    cachedTokens: number;
  };
  input?: string | any;
  response?: string | any;
}

export interface CompletionTrajectoryItemProps {
  content: CompletionTimelineContent;
  header?: string;
  expandedState: boolean;
  isExpandable?: boolean;
}

const parseJsonSafely = (str: string | any): any => {
  if (typeof str !== "string") return str;
  try {
    return JSON.parse(str);
  } catch (e) {
    console.error(`Error parsing JSON: ${str}, ${e}`);
    return str;
  }
};

export const CompletionTrajectoryItem = ({
  content,
  header,
  expandedState,
}: CompletionTrajectoryItemProps) => {
  const isExpandable = !!(content.input || content.response);

  return (
    <div className="space-y-4">
      {/* Token Usage Summary */}
      {content.usage && (
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Prompt</span>
            <span className="font-mono text-gray-900">
              {content.usage.promptTokens}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Completion</span>
            <span className="font-mono text-gray-900">
              {content.usage.completionTokens}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-500">Cached</span>
            <span className="font-mono text-gray-900">
              {content.usage.cachedTokens}
            </span>
          </div>
        </div>
      )}

      {expandedState && (
        <div className="space-y-3">
          {header && (
            <div className="text-sm font-medium text-gray-900">{header}</div>
          )}

          {content.input && (
            <div>
              <div className="mb-2 text-xs font-medium text-gray-700">
                Input
              </div>
              <div className="overflow-hidden rounded-md bg-gray-50">
                <JsonView
                  data={parseJsonSafely(content.input)}
                  defaultExpanded={false}
                  expanded={expandedState}
                />
              </div>
            </div>
          )}

          {content.response && (
            <div>
              <div className="mb-2 text-xs font-medium text-gray-700">
                Response
              </div>
              <div className="overflow-hidden rounded-md bg-gray-50">
                <JsonView
                  data={parseJsonSafely(content.response)}
                  defaultExpanded={false}
                  expanded={expandedState}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
