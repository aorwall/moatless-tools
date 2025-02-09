interface ErrorDetailsProps {
  content: {
    error: string;
  };
}

export const ErrorDetails = ({ content }: ErrorDetailsProps) => {
  return (
    <div className="rounded-lg bg-red-50 border border-red-200 p-4">
      <pre className="text-sm text-red-700 whitespace-pre-wrap font-mono">
        {content.error}
      </pre>
    </div>
  );
};
