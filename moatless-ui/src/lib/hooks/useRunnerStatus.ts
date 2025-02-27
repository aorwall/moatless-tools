import { useQuery } from '@tanstack/react-query';
import { runnerApi } from '@/lib/api/runner';

export const runnerKeys = {
  all: ['runner'] as const,
  status: () => [...runnerKeys.all, 'status'] as const,
};

export function useRunnerStatus(options = {}) {
  return useQuery({
    queryKey: runnerKeys.status(),
    queryFn: () => runnerApi.getStatus(),
    refetchInterval: 10000, // Refetch every 10 seconds
    ...options,
  });
} 