import { JsonViewer } from '../ui/json-viewer';

interface CompletionInputProps {
  input: any;
}

export function CompletionInput({ input }: CompletionInputProps) {
  return (
    <div>
      <div className="mb-2 text-sm font-medium text-gray-700">Input</div>
      <div className="overflow-hidden rounded-md bg-gray-50">
        <JsonViewer data={input} />
      </div>
    </div>
  );
} 