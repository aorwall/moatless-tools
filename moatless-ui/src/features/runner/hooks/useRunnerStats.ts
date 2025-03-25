import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '@/lib/api/config';
import { RunnerStats } from '../types';

// Define the query key
export const runnerStatsKey = {
    all: ['runnerStats'] as const,
};

// Fetch function for runner stats
async function fetchRunnerStats(): Promise<RunnerStats> {
    return apiRequest<RunnerStats>('/runner/stats');
}

// Hook to get runner stats
export function useRunnerStats(options = {}) {
    return useQuery({
        queryKey: runnerStatsKey.all,
        queryFn: fetchRunnerStats,
        // Refresh every 15 seconds by default
        refetchInterval: 15000,
        ...options,
    });
} 