/**
 * Shared utility hooks for file operations
 */

/**
 * Count additions and deletions in a git patch
 * @param patch The git patch string
 * @returns Object with additions and deletions counts
 */
export const countPatchChanges = (patch: string) => {
  if (!patch) return { additions: 0, deletions: 0 };

  const lines = patch.split("\n");
  const additionCount = lines.filter(
    (line) => line.startsWith("+") && !line.startsWith("+++"),
  ).length;
  const deletionCount = lines.filter(
    (line) => line.startsWith("-") && !line.startsWith("---"),
  ).length;

  return { additions: additionCount, deletions: deletionCount };
};

/**
 * Extract file name from a file path
 * @param filePath The full file path
 * @returns The file name (last part of the path)
 */
export const getFileName = (filePath: string): string => {
  const parts = filePath.split("/");
  return parts[parts.length - 1];
};

/**
 * Truncate a file path to fit within a maximum length
 * @param filePath The full file path
 * @param maxLength Maximum length for the truncated path
 * @returns Truncated file path
 */
export const truncateFilePath = (
  filePath: string,
  maxLength: number = 30,
): string => {
  if (filePath.length <= maxLength) return filePath;

  const parts = filePath.split("/");
  const fileName = parts.pop() || "";

  // If just the filename is too long, truncate it
  if (fileName.length >= maxLength - 3) {
    return "..." + fileName.substring(fileName.length - (maxLength - 3));
  }

  // Otherwise, keep the filename and add as many directories as possible
  let result = fileName;
  let remainingLength = maxLength - fileName.length - 3; // -3 for the "..."

  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    // +1 for the slash
    if (part.length + 1 <= remainingLength) {
      result = part + "/" + result;
      remainingLength -= part.length + 1;
    } else {
      break;
    }
  }

  return "..." + (result.startsWith("/") ? result : "/" + result);
};

/**
 * Calculate total changes across multiple files
 * @param files Array of files with patches
 * @returns Object with total additions and deletions
 */
export const calculateTotalChanges = (files?: Array<{ patch?: string }>) => {
  return (
    files?.reduce(
      (acc, file) => {
        if (file.patch) {
          const { additions, deletions } = countPatchChanges(file.patch);
          acc.additions += additions;
          acc.deletions += deletions;
        }
        return acc;
      },
      { additions: 0, deletions: 0 },
    ) || { additions: 0, deletions: 0 }
  );
};
