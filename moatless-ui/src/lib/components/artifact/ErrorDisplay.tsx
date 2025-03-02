import { FC } from "react";
import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { ApiError } from "@/lib/api/config";

interface ErrorDetails {
  errorType?: string;
  message?: string;
  artifactType?: string;
  artifactId?: string;
}

interface FormattedError {
  status: number;
  message: string;
  details: ErrorDetails | null;
}

interface ErrorDisplayProps {
  error: unknown;
}

const formatError = (error: unknown): FormattedError => {
  if (error instanceof ApiError) {
    const details = error.data?.detail || error.data;
    return {
      status: error.status,
      message:
        typeof details === "object" && details?.message
          ? details.message
          : error.message,
      details:
        details && typeof details === "object"
          ? {
              errorType: details.error_type,
              message: details.message,
              artifactType: details.type,
              artifactId: details.id,
            }
          : null,
    };
  }
  if (error instanceof Error) {
    return {
      status: 500,
      message: error.message,
      details: null,
    };
  }
  return {
    status: 500,
    message: "An unexpected error occurred",
    details: null,
  };
};

export const ErrorDisplay: FC<ErrorDisplayProps> = ({ error }) => {
  const formattedError = formatError(error);

  return (
    <Alert variant="destructive">
      <AlertCircle className="h-4 w-4" />
      <AlertDescription className="flex flex-col gap-1">
        <div className="font-medium">Error {formattedError.status}</div>
        <div className="text-sm">{formattedError.message}</div>
        {formattedError.details && (
          <div className="mt-2 space-y-1 text-xs">
            {formattedError.details.errorType && (
              <div>Type: {formattedError.details.errorType}</div>
            )}
            {formattedError.details.message &&
              formattedError.details.message !== formattedError.message && (
                <div>Details: {formattedError.details.message}</div>
              )}
            {formattedError.details.artifactType && (
              <div className="text-gray-500">
                Artifact: {formattedError.details.artifactType}/
                {formattedError.details.artifactId}
              </div>
            )}
          </div>
        )}
      </AlertDescription>
    </Alert>
  );
};
