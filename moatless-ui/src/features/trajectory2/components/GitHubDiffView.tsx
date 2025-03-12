import React from 'react';

interface GitHubDiffViewProps {
    diff: string;
    className?: string;
}

export const GitHubDiffView: React.FC<GitHubDiffViewProps> = ({ diff, className }) => {
    if (!diff) return null;

    // Parse and process the diff content
    const lines = diff.split('\n');

    // Track line numbers for added/removed sections
    let oldLineNumber = 0;
    let newLineNumber = 0;

    // Prepare lines with appropriate styling and line numbers
    const styledDiff = lines.map((line, index) => {
        let lineClass = '';
        let lineContent = line;
        let showOldLine = true;
        let showNewLine = true;

        // Style based on Git diff prefixes
        if (line.startsWith('+') && !line.startsWith('+++')) {
            lineClass = 'bg-green-50 text-green-900 dark:bg-green-950 dark:text-green-300';
            // Only increment new line number for additions
            newLineNumber++;
            showOldLine = false; // Don't show old line number for additions
        } else if (line.startsWith('-') && !line.startsWith('---')) {
            lineClass = 'bg-red-50 text-red-900 dark:bg-red-950 dark:text-red-300';
            // Only increment old line number for deletions
            oldLineNumber++;
            showNewLine = false; // Don't show new line number for deletions
        } else if (line.startsWith('@@ ')) {
            // Parse the @@ line to get new line numbers
            const match = line.match(/@@ -(\d+),(\d+) \+(\d+),(\d+) @@/);
            if (match) {
                oldLineNumber = parseInt(match[1], 10) - 1; // -1 because we increment before using
                newLineNumber = parseInt(match[3], 10) - 1; // -1 because we increment before using
            }

            showOldLine = false;
            showNewLine = false;
        } else if (!line.startsWith('---') && !line.startsWith('+++')) {
            // For context lines, increment both line numbers
            oldLineNumber++;
            newLineNumber++;
        } else {
            // For file headers (--- or +++ lines), don't show line numbers
            showOldLine = false;
            showNewLine = false;
        }

        return (
            <div key={index} className={`${lineClass} flex`}>
                {/* Line numbers */}
                <div className="flex text-gray-500 select-none border-r border-gray-200 dark:border-gray-700 pr-2 w-14">
                    <div className="w-6 text-right tabular-nums">
                        {showOldLine ? oldLineNumber : ' '}
                    </div>
                    <div className="w-6 text-right tabular-nums ml-2">
                        {showNewLine ? newLineNumber : ' '}
                    </div>
                </div>

                {/* Line content */}
                <div className="px-2 whitespace-pre overflow-x-auto">
                    {lineContent}
                </div>
            </div>
        );
    });

    return (
        <div className={`overflow-x-auto rounded ${className}`}>
            <div className="border border-border rounded-md text-xs font-mono">
                {styledDiff}
            </div>
        </div>
    );
};

// Helper function to extract filename from diff content
function extractFileNameFromDiff(diff: string): string | null {
    // Look for the +++ line which typically contains the file path
    const filePathMatch = diff.match(/^\+\+\+ [b]\/(.+)$/m);
    if (filePathMatch && filePathMatch[1]) {
        return filePathMatch[1];
    }

    return null;
}

// Helper function to extract file extension from diff content
function getFileExtensionFromDiff(diff: string): string {
    // Try to find file path in the diff header
    const filePathMatch = diff.match(/^(---|\+\+\+) [ab]\/(.+)$/m);
    if (filePathMatch && filePathMatch[2]) {
        const filePath = filePathMatch[2];
        const extMatch = filePath.match(/\.([^./\\]+)$/);
        return extMatch ? extMatch[1] : '';
    }
    return '';
} 