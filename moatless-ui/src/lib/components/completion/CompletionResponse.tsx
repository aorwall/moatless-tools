import { JsonViewer } from '../ui/json-viewer';

interface CompletionResponseProps {
  choices: Array<{
    message: {
      content?: string;
      role?: string;
      tool_calls?: Array<{
        function: {
          name: string;
          arguments: string;
        };
        id: string;
        type: string;
      }>;
    };
  }>;
}

export function CompletionResponse({ choices }: CompletionResponseProps) {
  const message = choices[0]?.message;
  if (!message) return null;

  return (
    <div className="space-y-4">
      {message.content && message.content !== "" && (
        <div>
          <div className="mb-2 text-sm font-medium text-gray-700">Content</div>
          <pre className="whitespace-pre-wrap rounded-md bg-gray-50 p-4 text-sm">
            {message.content}
          </pre>
        </div>
      )}

      {message.tool_calls?.map((tool, index) => (
        <div key={tool.id} className="space-y-2">
          <div className="text-sm font-medium text-gray-700">
            {tool.function.name} ({index + 1})
          </div>
          <div className="rounded-md bg-gray-50 p-4">
            <div>
              <JsonViewer
                data={
                  typeof tool.function.arguments === 'string'
                    ? JSON.parse(tool.function.arguments)
                    : tool.function.arguments
                }
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
} 