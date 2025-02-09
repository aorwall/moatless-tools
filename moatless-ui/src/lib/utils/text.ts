/**
 * Truncates a string to a specified length and adds an ellipsis if truncated
 * @param message The string to truncate
 * @param maxLength Maximum length before truncation (default: 200)
 * @returns Truncated string with ellipsis if needed
 */
export const truncateMessage = (
  message?: string,
  maxLength: number = 200,
): string => {
  if (!message) return "";
  return message.length > maxLength
    ? `${message.slice(0, maxLength)}...`
    : message;
};
