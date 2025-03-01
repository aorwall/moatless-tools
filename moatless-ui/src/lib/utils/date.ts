/**
 * Standard date/time formatter for the application
 */
export const dateTimeFormat = new Intl.DateTimeFormat('en-US', {
  dateStyle: 'short',
  timeStyle: 'medium',
});

/**
 * Format for dates only
 */
export const dateFormat = new Intl.DateTimeFormat('en-US', {
  dateStyle: 'medium',
});

/**
 * Format a date for display
 * @param date The date to format
 * @returns A formatted date string
 */
export function formatDate(date: Date | string | number): string {
  return dateFormat.format(new Date(date));
}

/**
 * Format for relative time (e.g., "5 minutes ago")
 * @param date The date to format
 * @returns A string with the relative time
 */
export function formatRelativeTime(date: Date | string | number): string {
  const now = new Date();
  const targetDate = new Date(date);
  const diffMs = now.getTime() - targetDate.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHr = Math.floor(diffMin / 60);
  const diffDays = Math.floor(diffHr / 24);

  if (diffSec < 60) {
    return 'just now';
  } else if (diffMin < 60) {
    return `${diffMin} minute${diffMin !== 1 ? 's' : ''} ago`;
  } else if (diffHr < 24) {
    return `${diffHr} hour${diffHr !== 1 ? 's' : ''} ago`;
  } else if (diffDays < 7) {
    return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
  } else {
    return dateFormat.format(targetDate);
  }
} 