import React from "react";
import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { ScrollArea } from "@/lib/components/ui/scroll-area";

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: React.ErrorInfo;
}

class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Error caught by ErrorBoundary:", error, errorInfo);
    this.setState({ error, errorInfo });

    // Log to console with source map support
    if (error.stack) {
      console.error("Error stack with source map support:", error.stack);
    }
  }

  render() {
    // Always show detailed error info regardless of environment
    // This helps with debugging in production
    if (this.state.hasError) {
      return (
        <div className="p-4 space-y-4">
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Something went wrong</AlertTitle>
            <AlertDescription>{this.state.error?.message}</AlertDescription>
          </Alert>

          <div className="space-y-2">
            <h3 className="text-sm font-medium">Error Stack</h3>
            <ScrollArea className="h-[200px] w-full rounded-md border p-4">
              <pre className="text-xs font-mono whitespace-pre-wrap">
                {this.state.error?.stack}
              </pre>
            </ScrollArea>
          </div>

          {this.state.errorInfo && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Component Stack</h3>
              <ScrollArea className="h-[200px] w-full rounded-md border p-4">
                <pre className="text-xs font-mono whitespace-pre-wrap">
                  {this.state.errorInfo.componentStack}
                </pre>
              </ScrollArea>
            </div>
          )}

          <div className="mt-4 text-sm">
            <p>
              Error: Objects are not valid as a React child (found: object with
              keys {"{type, text, cache_control}"}). If you meant to render a
              collection of children, use an array instead.
            </p>
            <p className="mt-2">
              This usually happens when you're trying to render an object
              directly in JSX. Check for places where you might be rendering an
              object instead of its string representation.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
