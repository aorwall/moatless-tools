import { useParams } from "react-router-dom";
import { useModel } from "@/lib/hooks/useModels";

export function ModelsPage() {
  const { id } = useParams();
  const { data: selectedModel } = useModel(id ?? "");

  if (!selectedModel) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center text-gray-500">
          Select a model to view details
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-none border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">{selectedModel.model}</h1>
            <div className="mt-1 text-sm text-gray-500">
              Model Configuration
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
                  <dt className="font-medium">Response Format</dt>
                  <dd className="text-gray-600">
                    {selectedModel.response_format}
                  </dd>
                </div>
                <div>
                  <dt className="font-medium">Model ID</dt>
                  <dd className="font-mono text-sm text-gray-600">
                    {selectedModel.id}
                  </dd>
                </div>
              </dl>
            </div>
            <div>
              <h3>Configuration</h3>
              <pre className="rounded-lg bg-gray-50 p-4">
                <code className="text-sm">
                  {JSON.stringify(selectedModel, null, 2)}
                </code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
