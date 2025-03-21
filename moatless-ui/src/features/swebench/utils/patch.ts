/**
 * Extracts file paths from a git diff patch
 * @param patch Git diff patch content
 * @returns Array of file paths in the patch
 */
export function extractFilesFromPatch(patch: string): string[] {
    const filePathRegex = /^diff --git a\/(.+?) b\/(.+?)$/gm;
    const files: string[] = [];
    let match;

    while ((match = filePathRegex.exec(patch)) !== null) {
        // match[1] is the file path from "a/" side
        files.push(match[1]);
    }

    return files;
}

/**
 * Get the appropriate language for syntax highlighting based on file extension
 * @param filename File name or path
 * @returns Language for syntax highlighting
 */
export function getLanguageFromFilename(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase();

    switch (ext) {
        case 'js':
            return 'javascript';
        case 'ts':
        case 'tsx':
            return 'typescript';
        case 'jsx':
            return 'jsx';
        case 'py':
            return 'python';
        case 'java':
            return 'java';
        case 'c':
            return 'c';
        case 'cpp':
        case 'cc':
        case 'cxx':
            return 'cpp';
        case 'cs':
            return 'csharp';
        case 'go':
            return 'go';
        case 'rb':
            return 'ruby';
        case 'php':
            return 'php';
        case 'rs':
            return 'rust';
        case 'kt':
        case 'kts':
            return 'kotlin';
        case 'swift':
            return 'swift';
        case 'html':
            return 'html';
        case 'css':
            return 'css';
        case 'json':
            return 'json';
        case 'md':
            return 'markdown';
        case 'yml':
        case 'yaml':
            return 'yaml';
        case 'sh':
        case 'bash':
            return 'bash';
        default:
            return 'plaintext';
    }
} 