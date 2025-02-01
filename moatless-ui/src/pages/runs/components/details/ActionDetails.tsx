import { JsonView } from '@/lib/components/ui/json-view';

interface ActionDetailsProps {
  content: {
    properties: Record<string, any>;
    errors?: string[];
    warnings?: string[];
  };
}

export const ActionDetails = ({ content }: ActionDetailsProps) => {
  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-md bg-gray-50 p-4">
        <div className="min-w-[300px] space-y-2">
          {Object.entries(content.properties || {}).map(([key, value]) => (
            <div key={key} className="grid grid-cols-[150px,1fr] gap-4">
              <div className="text-sm font-medium text-gray-600">{key}:</div>
              <div className="font-mono text-sm">
                <JsonView data={value} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {(content.errors?.length || content.warnings?.length) && (
        <div className="space-y-2">
          {content.errors?.map((error, index) => (
            <div key={index} className="rounded-md bg-red-50 p-3 text-sm text-red-600">
              {error}
            </div>
          ))}
          {content.warnings?.map((warning, index) => (
            <div key={index} className="rounded-md bg-yellow-50 p-3 text-sm text-yellow-600">
              {warning}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}; 