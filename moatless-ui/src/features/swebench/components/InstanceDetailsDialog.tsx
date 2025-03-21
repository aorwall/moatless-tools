import React from "react";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "@/lib/components/ui/dialog";
import {
    Tabs,
    TabsContent,
    TabsList,
    TabsTrigger,
} from "@/lib/components/ui/tabs";
import { Button } from "@/lib/components/ui/button";
import { Badge } from "@/lib/components/ui/badge";
import { FileIcon, GitBranchIcon, CheckCircleIcon, XCircleIcon, CodeIcon, BeakerIcon, GitPullRequestIcon, BugIcon } from "lucide-react";
import { useInstance } from "../hooks/useInstance";
import { cn } from "@/lib/utils";
import { extractFilesFromPatch, getLanguageFromFilename } from "../utils/patch";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface InstanceDetailsDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    instanceId: string | null;
}

export function InstanceDetailsDialog({
    open,
    onOpenChange,
    instanceId,
}: InstanceDetailsDialogProps) {
    const {
        data: instance,
        isLoading,
        isError,
        error
    } = useInstance(instanceId);

    // Get file information from the patch
    const patchFiles = React.useMemo(() => {
        if (!instance?.golden_patch) return [];
        return extractFilesFromPatch(instance.golden_patch);
    }, [instance?.golden_patch]);

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[90vw] max-h-[90vh] overflow-hidden flex flex-col">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        {instanceId && (
                            <>
                                <span>Instance Details: </span>
                                <Badge variant="outline">{instanceId}</Badge>
                                {patchFiles.length > 0 && (
                                    <div className="ml-4 flex items-center text-xs text-muted-foreground">
                                        <CodeIcon className="w-3 h-3 mr-1" />
                                        <span>Files: {patchFiles.join(", ")}</span>
                                    </div>
                                )}
                            </>
                        )}
                    </DialogTitle>
                </DialogHeader>

                <div className="flex-1 overflow-auto">
                    {isLoading ? (
                        <div className="text-center py-8">Loading instance details...</div>
                    ) : isError ? (
                        <div className="text-center py-8 text-destructive">
                            Error loading instance: {error?.toString()}
                        </div>
                    ) : instance ? (
                        <Tabs defaultValue="overview" className="w-full">
                            <TabsList className="grid grid-cols-5 mb-4">
                                <TabsTrigger value="overview">Overview</TabsTrigger>
                                <TabsTrigger value="code">Code Changes</TabsTrigger>
                                <TabsTrigger value="tests">Tests</TabsTrigger>
                                <TabsTrigger value="solutions">Solutions</TabsTrigger>
                                <TabsTrigger value="files">Files</TabsTrigger>
                            </TabsList>

                            {/* Overview Tab */}
                            <TabsContent value="overview" className="space-y-6">
                                {/* Repository and Stats */}
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="space-y-2">
                                        <h3 className="text-lg font-semibold">Repository</h3>
                                        <div className="flex items-center gap-2">
                                            <GitBranchIcon className="w-4 h-4" />
                                            <span>{instance.repo}</span>
                                            {instance.base_commit && (
                                                <Badge variant="outline" className="text-xs">
                                                    {instance.base_commit.substring(0, 7)}
                                                </Badge>
                                            )}
                                        </div>
                                    </div>
                                    <div className="space-y-2">
                                        <h3 className="text-lg font-semibold">Stats</h3>
                                        <div className="flex items-center gap-4">
                                            <div className="flex items-center gap-1">
                                                <FileIcon className="w-4 h-4" />
                                                <span>{instance.file_count} files</span>
                                            </div>
                                            <div className="flex items-center gap-1">
                                                <CheckCircleIcon className="w-4 h-4 text-green-500" />
                                                <span>{instance.resolved_count} solutions</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Problem Statement */}
                                <div className="space-y-2">
                                    <h3 className="text-lg font-semibold">Problem Statement</h3>
                                    <div className="bg-muted p-4 rounded-md whitespace-pre-wrap">
                                        {instance.problem_statement}
                                    </div>
                                </div>

                                {/* Dataset Information */}
                                <div className="space-y-2">
                                    <h3 className="text-lg font-semibold">Dataset Information</h3>
                                    <div className="bg-muted p-4 rounded-md">
                                        <div className="flex items-center gap-2 mb-2">
                                            <BeakerIcon className="w-4 h-4" />
                                            <span className="font-medium">Dataset: </span>
                                            <Badge>{instance.dataset}</Badge>
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                                            <div>
                                                <h4 className="text-sm font-medium mb-2">Files to Modify</h4>
                                                <ul className="space-y-1 text-sm">
                                                    {instance.expected_spans && Object.keys(instance.expected_spans).map((file) => (
                                                        <li key={file} className="flex items-center gap-1">
                                                            <FileIcon className="w-3 h-3" />
                                                            <span className="font-mono text-xs">{file}</span>
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>

                                            <div>
                                                <h4 className="text-sm font-medium mb-2">Test Files</h4>
                                                <ul className="space-y-1 text-sm">
                                                    {instance.test_file_spans && Object.keys(instance.test_file_spans).map((file) => (
                                                        <li key={file} className="flex items-center gap-1">
                                                            <FileIcon className="w-3 h-3" />
                                                            <span className="font-mono text-xs">{file}</span>
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </TabsContent>

                            {/* Code Changes Tab */}
                            <TabsContent value="code" className="space-y-6">
                                <Tabs defaultValue="golden-patch" className="w-full">
                                    <TabsList className="grid grid-cols-2">
                                        <TabsTrigger value="golden-patch">
                                            <GitPullRequestIcon className="w-4 h-4 mr-2" />
                                            Golden Patch
                                        </TabsTrigger>
                                        <TabsTrigger value="test-patch">
                                            <BugIcon className="w-4 h-4 mr-2" />
                                            Test Patch
                                        </TabsTrigger>
                                    </TabsList>

                                    <TabsContent value="golden-patch" className="bg-muted rounded-md overflow-auto max-h-[600px] mt-4">
                                        {instance?.golden_patch && (
                                            <SyntaxHighlighter
                                                language="diff"
                                                style={oneDark}
                                                customStyle={{
                                                    margin: 0,
                                                    padding: '1rem',
                                                    borderRadius: '0.5rem',
                                                    fontSize: '0.8rem',
                                                }}
                                                showLineNumbers={true}
                                            >
                                                {instance.golden_patch}
                                            </SyntaxHighlighter>
                                        )}
                                    </TabsContent>

                                    <TabsContent value="test-patch" className="bg-muted rounded-md overflow-auto max-h-[600px] mt-4">
                                        {instance?.test_patch && (
                                            <SyntaxHighlighter
                                                language="diff"
                                                style={oneDark}
                                                customStyle={{
                                                    margin: 0,
                                                    padding: '1rem',
                                                    borderRadius: '0.5rem',
                                                    fontSize: '0.8rem',
                                                }}
                                                showLineNumbers={true}
                                            >
                                                {instance.test_patch}
                                            </SyntaxHighlighter>
                                        )}
                                    </TabsContent>
                                </Tabs>
                            </TabsContent>

                            {/* Tests Tab */}
                            <TabsContent value="tests" className="space-y-6">
                                {(instance.fail_to_pass || instance.pass_to_pass) ? (
                                    <>
                                        {instance.fail_to_pass && instance.fail_to_pass.length > 0 && (
                                            <div className="space-y-2">
                                                <h3 className="text-lg font-semibold flex items-center gap-2">
                                                    <XCircleIcon className="w-5 h-5 text-destructive" />
                                                    Tests that fail but should pass with the fix
                                                </h3>
                                                <div className="bg-muted p-4 rounded-md max-h-[300px] overflow-auto">
                                                    <ul className="space-y-2">
                                                        {instance.fail_to_pass.map((test, i) => (
                                                            <li key={i} className="text-sm font-mono border-l-2 border-destructive pl-2">
                                                                {test}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            </div>
                                        )}

                                        {instance.pass_to_pass && instance.pass_to_pass.length > 0 && (
                                            <div className="space-y-2">
                                                <h3 className="text-lg font-semibold flex items-center gap-2">
                                                    <CheckCircleIcon className="w-5 h-5 text-green-500" />
                                                    Tests that pass and should continue to pass
                                                </h3>
                                                <div className="bg-muted p-4 rounded-md max-h-[300px] overflow-auto">
                                                    <div className="mb-2 text-sm">
                                                        {instance.pass_to_pass.length} passing tests
                                                    </div>
                                                    <ul className="space-y-1">
                                                        {instance.pass_to_pass.map((test, i) => (
                                                            <li key={i} className="text-xs font-mono border-l-2 border-green-500 pl-2">
                                                                {test}
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <div className="text-center py-8 text-muted-foreground">
                                        No test information available for this instance.
                                    </div>
                                )}
                            </TabsContent>

                            {/* Solutions Tab */}
                            <TabsContent value="solutions" className="space-y-6">
                                {instance.resolved_by && instance.resolved_by.length > 0 ? (
                                    <div className="space-y-4">
                                        <h3 className="text-lg font-semibold">Solutions ({instance.resolved_by.length})</h3>
                                        <div className="space-y-4 max-h-[600px] overflow-auto pr-2">
                                            {instance.resolved_by.map((solution, index) => (
                                                <div key={index} className="bg-muted p-4 rounded-md border border-border">
                                                    <h4 className="font-medium mb-4 pb-2 border-b">
                                                        Solution by: <Badge variant="outline">{solution.name}</Badge>
                                                    </h4>
                                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                                        <div>
                                                            <h5 className="font-medium text-sm mb-2">Updated Spans:</h5>
                                                            <ul className="space-y-2">
                                                                {Object.entries(solution.updated_spans).map(([file, spans]) => (
                                                                    <li key={file} className="bg-background p-2 rounded">
                                                                        <div className="font-mono text-sm mb-1">{file}</div>
                                                                        <div className="text-xs text-muted-foreground pl-4">
                                                                            {spans.map((span, i) => (
                                                                                <div key={i} className="mb-1">
                                                                                    <span className="font-medium">{span}</span>
                                                                                </div>
                                                                            ))}
                                                                        </div>
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>

                                                        {solution.alternative_spans && Object.keys(solution.alternative_spans).length > 0 && (
                                                            <div>
                                                                <h5 className="font-medium text-sm mb-2">Alternative Spans:</h5>
                                                                <ul className="space-y-2">
                                                                    {Object.entries(solution.alternative_spans).map(([file, spans]) => (
                                                                        <li key={file} className="bg-background p-2 rounded">
                                                                            <div className="font-mono text-sm mb-1">{file}</div>
                                                                            <div className="text-xs text-muted-foreground pl-4">
                                                                                {spans.map((span, i) => (
                                                                                    <div key={i} className="mb-1">
                                                                                        <span className="font-medium">{span}</span>
                                                                                    </div>
                                                                                ))}
                                                                            </div>
                                                                        </li>
                                                                    ))}
                                                                </ul>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-center py-8 text-muted-foreground">
                                        No solutions available for this instance.
                                    </div>
                                )}
                            </TabsContent>

                            {/* Files Tab */}
                            <TabsContent value="files" className="space-y-6">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="space-y-4">
                                        <h3 className="text-lg font-semibold">Expected Spans</h3>
                                        <div className="bg-muted p-4 rounded-md max-h-[600px] overflow-auto">
                                            <ul className="space-y-4">
                                                {instance.expected_spans && Object.entries(instance.expected_spans).map(([file, spans]) => (
                                                    <li key={file} className="bg-background p-3 rounded">
                                                        <div className="flex items-start gap-2">
                                                            <FileIcon className="w-4 h-4 mt-1" />
                                                            <div>
                                                                <div className="font-mono text-sm mb-2">{file}</div>
                                                                <div className="text-xs text-muted-foreground pl-2">
                                                                    <h6 className="font-medium mb-1">Spans:</h6>
                                                                    <div className="pl-2">
                                                                        {Array.isArray(spans) ?
                                                                            spans.map((span, i) => (
                                                                                <div key={i} className="mb-1">
                                                                                    <span className="font-medium">{span}</span>
                                                                                </div>
                                                                            ))
                                                                            : JSON.stringify(spans)
                                                                        }
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>

                                    <div className="space-y-4">
                                        <h3 className="text-lg font-semibold">Test File Spans</h3>
                                        <div className="bg-muted p-4 rounded-md max-h-[600px] overflow-auto">
                                            <ul className="space-y-4">
                                                {instance.test_file_spans && Object.entries(instance.test_file_spans).map(([file, spans]) => (
                                                    <li key={file} className="bg-background p-3 rounded">
                                                        <div className="flex items-start gap-2">
                                                            <FileIcon className="w-4 h-4 mt-1" />
                                                            <div>
                                                                <div className="font-mono text-sm mb-2">{file}</div>
                                                                <div className="text-xs text-muted-foreground pl-2">
                                                                    <h6 className="font-medium mb-1">Spans:</h6>
                                                                    <div className="pl-2">
                                                                        {Array.isArray(spans) ?
                                                                            spans.map((span, i) => (
                                                                                <div key={i} className="mb-1">
                                                                                    <span className="font-medium">{span}</span>
                                                                                </div>
                                                                            ))
                                                                            : JSON.stringify(spans)
                                                                        }
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                            </TabsContent>
                        </Tabs>
                    ) : null}
                </div>

                <DialogFooter>
                    <Button variant="outline" onClick={() => onOpenChange(false)}>
                        Close
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
} 