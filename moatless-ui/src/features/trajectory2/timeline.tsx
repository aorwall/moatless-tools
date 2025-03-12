"use client"

import React from "react"
import { FileText, Code, CheckCircle, Info, Search, Replace, Terminal, Brain, User, AlertCircle, Cpu, Beaker } from "lucide-react"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/lib/components/ui/accordion"
import { Badge } from "@/lib/components/ui/badge"
import { Card } from "@/lib/components/ui/card"
import { cn } from "@/lib/utils"
import { Action, ActionStep, ArtifactChangeContent, Node, Trajectory } from "@/lib/types/trajectory"
import { NodeCircle } from "../trajectory/components/NodeCircle"
import { useTrajectoryStore } from "../trajectory/stores/trajectoryStore"
import { GitHubDiffView } from "./components/GitHubDiffView"


// Helper function to get the appropriate icon for an action
const getActionIcon = (actionType: string) => {
    switch (actionType) {
        case "StringReplace":
            return <Replace className="h-4 w-4" />
        case "SemanticSearch":
            return <Search className="h-4 w-4" />
        case "RunTests":
            return <Beaker className="h-4 w-4" />
        case "FindClass":
            return <Search className="h-4 w-4" />
        case "FindFunction":
            return <Search className="h-4 w-4" />
        case "Finish":
            return <CheckCircle className="h-4 w-4" />
        case "Execute":
            return <Terminal className="h-4 w-4" />
        default:
            return <Code className="h-4 w-4" />
    }
}

// Helper function to get the appropriate color for a status
const getStatusColor = (status: "success" | "error" | "warning" | "info") => {
    switch (status) {
        case "success":
            return "bg-green-500/10 text-green-500 border-green-500/20"
        case "error":
            return "bg-red-500/10 text-red-500 border-red-500/20"
        case "warning":
            return "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
        case "info":
            return "bg-blue-500/10 text-blue-500 border-blue-500/20"
        default:
            return "bg-blue-500/10 text-blue-500 border-blue-500/20"
    }
}

// Helper function to get the appropriate icon for an artifact type
const getArtifactIcon = (artifactType: string) => {
    switch (artifactType) {
        case "file":
            return <FileText className="h-4 w-4" />
        case "context":
            return <Info className="h-4 w-4" />
        case "test":
            return <Beaker className="h-4 w-4" />
        default:
            return <Code className="h-4 w-4" />
    }
}


// New component for compact artifact summary
const ArtifactSummary: React.FC<{ artifacts: ArtifactChangeContent[] }> = ({ artifacts }) => {
    const totalAdded = artifacts.reduce((sum, artifact) => sum + (artifact.properties?.added_lines || 0), 0)
    const totalRemoved = artifacts.reduce((sum, artifact) => sum + (artifact.properties?.removed_lines || 0), 0)
    const totalTokens = artifacts.reduce((sum, artifact) => sum + (artifact.properties?.token_count || 0), 0)

    return (
        <div className="flex items-center gap-3 text-xs">
            <span className="text-muted-foreground">Changes:</span>
            <span className="text-green-500">+{totalAdded}</span>
            <span className="text-red-500">-{totalRemoved}</span>
            {totalTokens > 0 && <span className="text-blue-500">({totalTokens} tokens)</span>}
        </div>
    )
}

// Helper function to truncate text with ellipsis
const truncateText = (text: string, maxLength = 100) => {
    if (!text || text.length <= maxLength) return text

    // Find the last line break before the maxLength
    const lastBreakIndex = text.substring(0, maxLength).lastIndexOf('\n')

    // If there are line breaks, preserve them up to maxLength
    if (lastBreakIndex > 0) {
        // Return up to the last complete line that fits
        return text.substring(0, lastBreakIndex) + "\n..."
    }

    // Otherwise truncate normally
    return text.slice(0, maxLength).trim() + "..."
}

// Replace the existing CompletionBadge component with this new TokenBadge component
const TokenBadge: React.FC<{ prompt: number; completion: number, cached?: number }> = ({ prompt, completion, cached }) => {
    return (
        <Badge variant="outline"
            className="text-xs bg-white/50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700">
            <Cpu className="h-3 w-3 text-gray-400" />
            <span className="ml-1 text-gray-600 dark:text-gray-400">{prompt}</span>
            {cached !== undefined && cached > 0 && (

                <span className="ml-1 text-gray-500 dark:text-gray-400"> ({cached})</span>


            )}

            <span className="mx-1 text-gray-400">•</span>
            <span className="text-gray-600 dark:text-gray-400">{completion}</span>
        </Badge>
    )
}

// Add a new component to display text with line breaks
const TextWithLineBreaks: React.FC<{ text: string, className?: string, preserveWhitespace?: boolean }> = ({
    text,
    className,
    preserveWhitespace = false
}) => {
    if (!text) return null;

    // Split text by new lines and create paragraph elements
    const lines = text.split('\n');

    return (
        <div className={cn(className, preserveWhitespace && "whitespace-pre")}>
            {lines.map((line, index) => (
                <React.Fragment key={index}>
                    {line}
                    {index < lines.length - 1 && <br />}
                </React.Fragment>
            ))}
        </div>
    );
};

// Add a component to render property values of different types
const PropertyValue: React.FC<{ value: any, className?: string, maxLines?: number }> = ({ value, className, maxLines = 5 }) => {
    const [isExpanded, setIsExpanded] = React.useState(false);

    // Handle null/undefined
    if (value === null || value === undefined) {
        return "";
    }

    // Function to truncate text by number of lines
    const truncateByLines = (text: string, maxLines: number) => {
        if (!text) return text;

        const lines = text.split('\n');
        if (lines.length <= maxLines) return text;

        return lines.slice(0, maxLines).join('\n') + '\n...';
    }

    // Handle string values
    if (typeof value === 'string') {
        const lines = value.split('\n');
        const isLong = lines.length > maxLines;
        const displayText = isExpanded ? value : truncateByLines(value, maxLines);

        return (
            <>
                <TextWithLineBreaks
                    text={displayText}
                    className={className}
                    preserveWhitespace={true}
                />
                {isLong && (
                    <button
                        onClick={() => setIsExpanded(!isExpanded)}
                        className="text-xs text-blue-500 hover:underline mt-1"
                    >
                        {isExpanded ? 'Show less' : 'Show more'}
                    </button>
                )}
            </>
        );
    }
    // Handle arrays and objects
    if (typeof value === 'object') {
        try {
            // Handle arrays specifically
            if (Array.isArray(value)) {
                if (value.length === 0) {
                    return <span className={className}>[]</span>;
                }

                return (
                    <ul className={cn("list-disc pl-5", className)}>
                        {value.map((item, index) => (
                            <li key={index}>
                                <PropertyValue value={item} maxLines={maxLines} />
                            </li>
                        ))}
                    </ul>
                );
            }

            // Handle regular objects
            const formattedValue = JSON.stringify(value, null, 2);
            const lines = formattedValue.split('\n');
            const isLong = lines.length > maxLines;
            const displayText = isExpanded ? formattedValue : truncateByLines(formattedValue, maxLines);

            return (
                <>
                    <TextWithLineBreaks
                        text={displayText}
                        className={cn("font-mono", className)}
                        preserveWhitespace={true}
                    />
                    {isLong && (
                        <button
                            onClick={() => setIsExpanded(!isExpanded)}
                            className="text-xs text-blue-500 hover:underline mt-1"
                        >
                            {isExpanded ? 'Show less' : 'Show more'}
                        </button>
                    )}
                </>
            );
        } catch (e) {
            return <span className={className}>{String(value)}</span>;
        }
    }

    // Handle other primitive types
    return <span className={className}>{String(value)}</span>;
};

// Add a new component to display properties based on property names
const PropertyDisplay: React.FC<{ properties: Record<string, any> }> = ({ properties }) => {
    if (!properties || Object.keys(properties).length === 0) {
        return null;
    }

    // Special property names that need specific styling
    const specialPropertyStyles: Record<string, string> = {
        'old_str': 'bg-red-500/10 px-1 rounded',
        'new_str': 'bg-green-500/10 px-1 rounded',
        'diff': 'bg-blue-500/10 px-1 rounded',
        'query': 'bg-yellow-500/10 px-1 rounded text-wrap break-words',
        'path': 'font-semibold',
        'file_path': 'font-semibold',
        'error': 'bg-red-500/10 px-1 rounded',
        'warning': 'bg-yellow-500/10 px-1 rounded',
        'code_snippet': 'font-mono bg-gray-500/10 px-1 rounded',
        'command': 'font-mono bg-gray-500/10 px-1 rounded',
    };

    // Property names that should use monospace font
    const codeProperties = [
        'old_str', 'new_str', 'code_snippet', 'command', 'snippet',
        'content', 'stderr', 'stdout', 'script'
    ];

    // Special handling for property order - we want to show some properties first
    const priorityOrder = [
        'path', 'file_path', 'query', 'old_str', 'new_str', 'command',
        'code_snippet', 'class_name', 'function_name', 'pattern'
    ];

    // Sort properties by priority order, then alphabetically
    const sortedEntries = Object.entries(properties)
        .filter(([_, value]) => value !== null && value !== undefined)
        .sort(([keyA], [keyB]) => {
            const indexA = priorityOrder.indexOf(keyA);
            const indexB = priorityOrder.indexOf(keyB);

            if (indexA !== -1 && indexB !== -1) return indexA - indexB;
            if (indexA !== -1) return -1;
            if (indexB !== -1) return 1;
            return keyA.localeCompare(keyB);
        });

    return React.createElement(
        'div',
        { className: "bg-muted/50 p-2 rounded-md text-sm" },
        React.createElement(
            'div',
            { className: "space-y-2" },
            sortedEntries.map(([key, value]) => {
                const valueClassName = specialPropertyStyles[key] || '';
                const isCodeProperty = codeProperties.includes(key);
                const containerClass = isCodeProperty ? 'font-mono' : '';

                return React.createElement(
                    'div',
                    { key, className: "flex flex-col mb-2 last:mb-0" },
                    React.createElement(
                        'span',
                        { className: "text-xs text-muted-foreground" },
                        `${key}:`
                    ),
                    React.createElement(
                        'span',
                        { className: cn("ml-4", valueClassName, containerClass) },
                        React.createElement(PropertyValue, {
                            value,
                            maxLines: 5,
                            className: isCodeProperty ? 'font-mono' : ''
                        })
                    )
                );
            })
        )
    );
};

// Updated ActionItem component
const ActionItem: React.FC<{
    action: ActionStep
    promptTokens?: number
    completionTokens?: number
    cachedTokens?: number
}> = ({ action, promptTokens, completionTokens, cachedTokens }) => {
    // Get display text based on action type
    const actionName = action.action.name
    const getActionDisplayText = () => {
        switch (actionName) {
            case "StringReplace":
            case "AppendString":
                return action.action.properties?.path
            case "FindClass":
                return action.action.properties?.class_name
            case "FindFunction":
                return action.action.properties?.function_name
            case "FindCodeSnippet":
                return action.action.properties?.code_snippet
            case "RunTests":
                return action.action.properties?.test_files[0]
            case "ViewCode":
                return action.action.properties?.files[0].file_path
            case "SemanticSearch":
                return `\"${action.action.properties?.query}\"`
            case "Finish":
                return action.action.properties?.finish_reason ? truncateText(action.action.properties.finish_reason, 60) : ""
            default:
                return ""
        }
    }

    // Calculate total changes
    const totalAdded = action.artifacts?.reduce((sum, a) => sum + (a.properties?.added_lines || 0), 0) || 0
    const totalRemoved = action.artifacts?.reduce((sum, a) => sum + (a.properties?.removed_lines || 0), 0) || 0
    const totalTokens = action.artifacts?.reduce((sum, a) => sum + (a.properties?.token_count || 0), 0) || 0

    // Check if there are warnings or errors
    const hasWarnings = action.warnings && action.warnings.length > 0
    const hasErrors = action.errors && action.errors.length > 0

    return (
        <div className="relative">
            {promptTokens !== undefined && completionTokens !== undefined && (
                <div className="absolute top-2 right-2">
                    <TokenBadge
                        prompt={promptTokens}
                        completion={completionTokens}
                        cached={cachedTokens}
                    />
                </div>
            )}
            <Accordion type="single" collapsible className="w-full border rounded-md">
                <AccordionItem value="action" className="border-none">
                    <AccordionTrigger className="px-3 py-2 hover:no-underline">
                        <div className="flex items-center justify-between w-full">
                            <div className="flex items-center gap-2">
                                {getActionIcon(actionName)}
                                <h4 className="font-medium">
                                    {actionName}
                                    {getActionDisplayText() && (
                                        <span className="text-sm text-muted-foreground ml-2">{getActionDisplayText()}</span>
                                    )}
                                </h4>
                            </div>

                            <div className="flex items-center gap-4">
                                {/* Show warnings and errors indicators */}
                                {(hasWarnings || hasErrors) && (
                                    <div className="flex items-center gap-2 max-w-[50%]">
                                        {hasWarnings && (
                                            <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500 border-yellow-500/20 flex items-center">
                                                <AlertCircle className="h-3 w-3 mr-1 shrink-0" />
                                                <span className="truncate">
                                                    {action.warnings.length > 1
                                                        ? `${action.warnings.length} warnings: ${truncateText(action.warnings[0], 30)}`
                                                        : truncateText(action.warnings[0], 40)
                                                    }
                                                </span>
                                            </Badge>
                                        )}
                                        {hasErrors && (
                                            <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20 flex items-center">
                                                <AlertCircle className="h-3 w-3 mr-1 shrink-0" />
                                                <span className="truncate">
                                                    {action.errors.length > 1
                                                        ? `${action.errors.length} errors: ${truncateText(action.errors[0], 30)}`
                                                        : truncateText(action.errors[0], 40)
                                                    }
                                                </span>
                                            </Badge>
                                        )}
                                    </div>
                                )}

                                {/* Show changes summary */}
                                {(totalAdded > 0 || totalRemoved > 0 || totalTokens != 0) && (
                                    <div className="flex items-center gap-3 text-xs">
                                        {(totalAdded > 0 || totalRemoved > 0) && (
                                            <span>
                                                Changes:
                                                {totalAdded > 0 && <span className="text-green-500 ml-1">+{totalAdded}</span>}
                                                {totalRemoved > 0 && <span className="text-red-500 ml-1">-{totalRemoved}</span>}
                                            </span>
                                        )}
                                        {totalTokens > 0 && <span className="text-blue-500">+{totalTokens} tokens</span>}
                                    </div>
                                )}
                            </div>
                        </div>
                    </AccordionTrigger>
                    <AccordionContent className="px-3 pb-3">
                        {/* Action properties section */}
                        {action.action.properties && <PropertyDisplay properties={action.action.properties} />}

                        {/* Artifacts section */}
                        {action.artifacts?.length && action.artifacts?.length > 0 && (
                            <div className="mt-3 space-y-1">
                                <h5 className="text-sm font-medium">Artifacts ({action.artifacts?.length || 0})</h5>
                                <div className="space-y-1">
                                    {action.artifacts?.map((artifact, index) => (
                                        <div
                                            key={index}
                                            className={cn(
                                                "flex items-start p-2 rounded-md text-sm bg-gray-500/5",
                                            )}
                                        >
                                            <div className="mr-2 mt-0.5">{getArtifactIcon(artifact.artifact_type)}</div>
                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-2 overflow-hidden">
                                                        <span className="font-medium truncate">{artifact.artifact_id}</span>
                                                        <Badge
                                                            variant="outline"
                                                            className={cn(
                                                                "shrink-0",
                                                                artifact.change_type === "added"
                                                                    ? "bg-green-500/10 text-green-500"
                                                                    : artifact.change_type === "updated"
                                                                        ? "bg-blue-500/10 text-blue-500"
                                                                        : "bg-gray-500/10 text-gray-500",
                                                            )}
                                                        >
                                                            {artifact.change_type}
                                                        </Badge>
                                                        <Badge variant="outline" className="shrink-0">
                                                            {artifact.artifact_type}
                                                        </Badge>
                                                    </div>

                                                    {/* Always show lines/tokens stats */}
                                                    <div className="flex items-center gap-3 text-xs shrink-0">
                                                        {(artifact.properties?.added_lines !== undefined && artifact.properties?.added_lines > 0)
                                                            || (artifact.properties?.removed_lines !== undefined && artifact.properties?.removed_lines > 0) && (
                                                                <div className="flex items-center gap-1">
                                                                    <span className="text-muted-foreground">Lines:</span>
                                                                    <span className="text-green-500">+{artifact.properties?.added_lines || 0}</span>
                                                                    <span className="text-red-500">-{artifact.properties?.removed_lines || 0}</span>
                                                                </div>
                                                            )}
                                                        {artifact.properties?.token_count !== undefined && artifact.properties?.token_count > 0 && (
                                                            <div className="flex items-center gap-1">
                                                                <span className="text-muted-foreground">Tokens:</span>
                                                                <span>+{artifact.properties?.token_count}</span>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>

                                                {/* Test results display */}
                                                {artifact.artifact_type === "test" && (
                                                    <div className="mt-2 flex flex-wrap gap-2">
                                                        <Badge variant="outline" className="bg-green-500/10 text-green-500">
                                                            {artifact.properties?.passed || 0} passed
                                                        </Badge>
                                                        {(artifact.properties?.failed !== undefined && artifact.properties?.failed > 0) && (
                                                            <Badge variant="outline" className="bg-red-500/10 text-red-500">
                                                                {artifact.properties?.failed} failed
                                                            </Badge>
                                                        )}
                                                        {(artifact.properties?.errors !== undefined && artifact.properties?.errors > 0) && (
                                                            <Badge variant="outline" className="bg-yellow-500/10 text-yellow-500">
                                                                {artifact.properties?.errors} errors
                                                            </Badge>
                                                        )}
                                                        {(artifact.properties?.skipped !== undefined && artifact.properties?.skipped > 0) && (
                                                            <Badge variant="outline" className="bg-gray-500/10 text-gray-500">
                                                                {artifact.properties?.skipped} skipped
                                                            </Badge>
                                                        )}
                                                        <Badge variant="outline" className="bg-blue-500/10 text-blue-500">
                                                            {artifact.properties?.total || 0} total
                                                        </Badge>
                                                    </div>
                                                )}

                                                {artifact.properties?.diff && (
                                                    <Accordion type="single" collapsible className="w-full mt-1">
                                                        <AccordionItem value="diff" className="border-none">
                                                            <AccordionTrigger className="py-1 text-xs hover:no-underline">
                                                                View Changes
                                                            </AccordionTrigger>
                                                            <AccordionContent>
                                                                <GitHubDiffView diff={artifact.properties?.diff} />
                                                            </AccordionContent>
                                                        </AccordionItem>
                                                    </Accordion>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}


                    </AccordionContent>
                </AccordionItem>
            </Accordion>
        </div>
    )
}

// Update the TimelineNode component
const TimelineNode: React.FC<{ node: Node, trajectory: Trajectory }> = ({ node, trajectory }) => {
    const [showFullReasoning, setShowFullReasoning] = React.useState(false)
    const [showFullUserMessage, setShowFullUserMessage] = React.useState(false)
    const [showFullError, setShowFullError] = React.useState(false)
    const isLastNode = node.nodeId === trajectory.nodes.length - 1;
    const hasChildren = node.children && node.children.length > 0;
    const { setSelectedNode } = useTrajectoryStore();

    const handleSelect = () => {
        setSelectedNode(trajectory.trajectory_id, node.nodeId);
    };

    return (
        <div className="relative pb-8">
            <div className="relative flex items-start space-x-3">
                <div className="relative">
                    <NodeCircle
                        trajectory={trajectory}
                        node={node}
                        isLastNode={isLastNode && !hasChildren}
                        isRunning={trajectory.status === "running"}
                        onClick={handleSelect}
                    />
                </div>

                <div className="min-w-0 flex-1">
                    <Card className="p-4 shadow-sm">
                        <div className="flex flex-col">
                            <div className="flex justify-between items-start mb-3">
                                <div className="flex-1">
                                    {node.userMessage && (
                                        <div className="flex items-start gap-2 mb-3">
                                            <User className="h-4 w-4 mt-1 shrink-0 text-blue-500" />
                                            <div className="text-sm">
                                                {showFullUserMessage ? (
                                                    <>
                                                        <TextWithLineBreaks text={node.userMessage} />{" "}
                                                        <button
                                                            onClick={() => setShowFullUserMessage(false)}
                                                            className="text-xs text-blue-500 hover:underline inline-flex items-center"
                                                        >
                                                            Show less
                                                        </button>
                                                    </>
                                                ) : (
                                                    <>
                                                        <TextWithLineBreaks text={truncateText(node.userMessage, 120)} />{" "}
                                                        <button
                                                            onClick={() => setShowFullUserMessage(true)}
                                                            className="text-xs text-blue-500 hover:underline inline-flex items-center"
                                                        >
                                                            Show more
                                                        </button>
                                                    </>
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    {node.thoughts && (
                                        <div className="flex items-start gap-2 mb-3">
                                            <Brain className="h-4 w-4 mt-1 shrink-0" />
                                            <div className="text-sm text-muted-foreground">
                                                {showFullReasoning ? (
                                                    <>
                                                        <TextWithLineBreaks text={node.thoughts} />{" "}
                                                        <button
                                                            onClick={() => setShowFullReasoning(false)}
                                                            className="text-xs text-blue-500 hover:underline inline-flex items-center"
                                                        >
                                                            Show less
                                                        </button>
                                                    </>
                                                ) : (
                                                    <>
                                                        <TextWithLineBreaks text={truncateText(node.thoughts, 120)} />{" "}
                                                        <button
                                                            onClick={() => setShowFullReasoning(true)}
                                                            className="text-xs text-blue-500 hover:underline inline-flex items-center"
                                                        >
                                                            Show more
                                                        </button>
                                                    </>
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    {node.error && (
                                        <div className="flex items-start gap-2">
                                            <AlertCircle className="h-4 w-4 mt-1 shrink-0 text-red-500" />
                                            <div className="text-sm text-red-500">
                                                {showFullError ? (
                                                    <>
                                                        <TextWithLineBreaks text={node.error} />{" "}
                                                        <button
                                                            onClick={() => setShowFullError(false)}
                                                            className="text-xs text-blue-500 hover:underline inline-flex items-center"
                                                        >
                                                            Show less
                                                        </button>
                                                    </>
                                                ) : (
                                                    <>
                                                        <TextWithLineBreaks text={truncateText(node.error, 120)} />{" "}
                                                        <button
                                                            onClick={() => setShowFullError(true)}
                                                            className="text-xs text-blue-500 hover:underline inline-flex items-center"
                                                        >
                                                            Show more
                                                        </button>
                                                    </>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                <div className="shrink-0 ml-4">
                                    {node.usage !== undefined && node.usage.prompt_tokens !== undefined && node.usage.prompt_tokens > 0 && (
                                        <TokenBadge
                                            prompt={node.usage.prompt_tokens || 0}
                                            completion={node.usage.completion_tokens || 0}
                                            cached={node.usage.cache_read_tokens || 0}
                                        />
                                    )}
                                </div>
                            </div>

                            <div className="space-y-3">
                                {node.actionSteps.map((action, index) => (
                                    <ActionItem
                                        key={index}
                                        action={action}
                                        promptTokens={index === 0 ? undefined : action.completion?.usage?.prompt_tokens}
                                        completionTokens={index === 0 ? undefined : action.completion?.usage?.completion_tokens}
                                        cachedTokens={index === 0 ? undefined : action.completion?.usage?.cache_read_tokens}
                                    />
                                ))}
                            </div>

                            <div className="mt-3 text-xs text-muted-foreground">{node.timestamp}</div>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    )
}

interface TimelineProps {
    trajectory: Trajectory
}

export function Timeline({ trajectory }: TimelineProps) {
    return (
        <div className="flow-root">
            <ul className="-mb-8">
                {trajectory.nodes.map((node) => (
                    <li key={node.nodeId}>
                        <TimelineNode node={node} trajectory={trajectory} />
                    </li>
                ))}
            </ul>
        </div>
    )
}

