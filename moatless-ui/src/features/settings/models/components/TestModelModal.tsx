import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from '@/lib/components/ui/dialog';
import { Button } from '@/lib/components/ui/button';
import { Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { type ModelTestResult } from '../types';

interface TestModelModalProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    modelId: string;
    isLoading: boolean;
    testResult: ModelTestResult | undefined;
}

export function TestModelModal({
    open,
    onOpenChange,
    modelId,
    isLoading,
    testResult,
}: TestModelModalProps) {
    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        {isLoading ? (
                            <>
                                <Loader2 className="h-4 w-4 animate-spin" />
                                Testing Model
                            </>
                        ) : testResult ? (
                            <>
                                {testResult.success ? (
                                    <CheckCircle2 className="h-5 w-5 text-green-500" />
                                ) : (
                                    <XCircle className="h-5 w-5 text-red-500" />
                                )}
                                {testResult.success ? 'Test Passed' : 'Test Failed'}
                            </>
                        ) : (
                            'Test Model'
                        )}
                    </DialogTitle>
                    <DialogDescription>
                        {isLoading
                            ? `Testing model ${modelId}...`
                            : testResult
                                ? `Results for model ${modelId}`
                                : `Run a test for model ${modelId}`}
                    </DialogDescription>
                </DialogHeader>

                <div className="space-y-4">
                    {isLoading ? (
                        <div className="flex justify-center items-center py-8">
                            <Loader2 className="h-8 w-8 animate-spin text-primary" />
                        </div>
                    ) : testResult ? (
                        <div className="space-y-4">
                            <div className="p-4 rounded-md bg-muted">
                                <p className="font-medium">{testResult.message}</p>

                                {testResult.model_response && (
                                    <div className="mt-4">
                                        <h4 className="text-sm font-semibold mb-1">Model Response:</h4>
                                        <p className="text-sm whitespace-pre-wrap bg-background p-2 rounded border">
                                            {testResult.model_response}
                                        </p>
                                    </div>
                                )}

                                {testResult.response_time_ms && (
                                    <p className="text-sm text-muted-foreground mt-4">
                                        Response time: {(testResult.response_time_ms / 1000).toFixed(2)}s
                                    </p>
                                )}

                                {testResult.error_details && (
                                    <div className="mt-4 p-2 rounded border border-red-200 bg-red-50 dark:bg-red-950/20 dark:border-red-800">
                                        <h4 className="text-sm font-semibold text-red-600 dark:text-red-400 mb-1">Error Details:</h4>
                                        <p className="text-sm text-red-600 dark:text-red-400 whitespace-pre-wrap">
                                            {testResult.error_details}
                                        </p>
                                    </div>
                                )}

                                {testResult.error_type && (
                                    <p className="text-sm text-red-600 dark:text-red-400 mt-2">
                                        Error Type: {testResult.error_type}
                                    </p>
                                )}
                            </div>
                        </div>
                    ) : (
                        <p>Click the button below to test the model.</p>
                    )}
                </div>

                <DialogFooter>
                    <Button
                        type="button"
                        onClick={() => onOpenChange(false)}
                        variant="default"
                    >
                        Close
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
} 