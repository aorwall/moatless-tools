interface ObservationDetailsProps {
  content: {
    message?: string;
  };
}

export const ObservationDetails = ({ content }: ObservationDetailsProps) => {
  return (
    <div className="space-y-2">
      <div className="prose prose-sm max-w-none">
        <pre className="whitespace-pre-wrap rounded-lg bg-gray-50 p-4 text-sm text-gray-700">
          {content.message || 'No observation available'}
        </pre>
      </div>
    </div>
  );
}; 