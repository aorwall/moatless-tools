import { useParams } from "react-router-dom";
import { useFlows } from "@/lib/hooks/useFlows";

export function FlowsPage() {
  const { id } = useParams();
  const { data: flows } = useFlows();
  const selectedFlow = flows?.find(f => f.id === id);

  if (!selectedFlow) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center text-gray-500">
          Select a flow to view details
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-none border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">{selectedFlow.id}</h1>
            <div className="mt-1 text-sm text-gray-500">
              Flow Configuration
            </div>
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-6">
        <div className="prose max-w-none">
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h3>Basic Information</h3>
              <dl className="space-y-2">
                <div>
                  <dt className="font-medium">Flow ID</dt>
                  <dd className="font-mono text-sm text-gray-600">
                    {selectedFlow.id}
                  </dd>
                </div>
                <div>
                  <dt className="font-medium">Description</dt>
                  <dd className="text-gray-600">
                    {selectedFlow.description || "No description"}
                  </dd>
                </div>
              </dl>
            </div>
            <div>
              <h3>Configuration</h3>
              <pre className="rounded-lg bg-gray-50 p-4">
                <code className="text-sm">
                  {JSON.stringify(selectedFlow, null, 2)}
                </code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 