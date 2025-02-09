import { JsonViewer } from "@/lib/components/ui/json-viewer";

interface ActionDetailsProps {
  content: any;
}

export const ActionDetails = ({ content }: ActionDetailsProps) => {
  // Get all properties except errors and warnings
  const properties = Object.entries(content).filter(
    ([key]) => key !== "errors" && key !== "warnings"
  );

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-md bg-gray-50 p-4">
        <div className="min-w-[300px] space-y-4">
          <div className="font-mono text-sm">
            <JsonViewer data={content} />
          </div>
        </div>
      </div>
    </div>
  );
};
