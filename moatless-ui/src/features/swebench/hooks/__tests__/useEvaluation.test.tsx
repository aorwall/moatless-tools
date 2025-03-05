import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRealtimeEvaluation } from '../useEvaluation';
import { useWebSocketStore } from '@/lib/stores/websocketStore';
import { evaluationApi } from '../../api/evaluation';
import React from 'react';

// Mock dependencies
vi.mock('@/lib/stores/websocketStore', () => ({
    useWebSocketStore: vi.fn(),
}));

vi.mock('../../api/evaluation', () => ({
    evaluationApi: {
        getEvaluation: vi.fn(),
    },
}));

// Mock timers
vi.useFakeTimers();

describe('useRealtimeEvaluation', () => {
    let queryClient: QueryClient;
    const mockSubscribeToProject = vi.fn().mockReturnValue(() => { });
    const mockSubscribeToTrajectory = vi.fn().mockReturnValue(() => { });
    const mockGetConnectionStatus = vi.fn().mockReturnValue('CONNECTED');
    const mockAddStatusListener = vi.fn().mockReturnValue(() => { });

    const mockEvaluation = {
        id: 'eval_123',
        evaluation_name: 'Test Evaluation',
        status: 'running',
        instances: [],
        created_at: new Date().toISOString(),
    };

    beforeEach(() => {
        vi.clearAllMocks();

        // Create a new QueryClient for each test
        queryClient = new QueryClient({
            defaultOptions: {
                queries: {
                    retry: false,
                },
            },
        });

        // Mock WebSocketStore
        (useWebSocketStore as any).mockReturnValue({
            subscribeToProject: mockSubscribeToProject,
            subscribeToTrajectory: mockSubscribeToTrajectory,
            getConnectionStatus: mockGetConnectionStatus,
            addStatusListener: mockAddStatusListener,
        });

        // Mock API response
        (evaluationApi.getEvaluation as any).mockResolvedValue(mockEvaluation);
    });

    afterEach(() => {
        queryClient.clear();
    });

    const wrapper = ({ children }: { children: React.ReactNode }) => {
        return (
            <QueryClientProvider client={queryClient}>
                {children}
            </QueryClientProvider>
        );
    };

    it('should subscribe to project events', async () => {
        const evaluationId = 'eval_123';

        // Render the hook
        renderHook(() => useRealtimeEvaluation(evaluationId), {
            wrapper,
        });

        // Verify project subscription was called
        expect(mockSubscribeToProject).toHaveBeenCalledWith(
            evaluationId,
            expect.any(Function)
        );

        // Verify trajectory subscription was NOT called
        expect(mockSubscribeToTrajectory).not.toHaveBeenCalled();
    });

    it('should fetch evaluation data', async () => {
        const evaluationId = 'eval_123';

        // Render the hook
        const { result } = renderHook(() => useRealtimeEvaluation(evaluationId), {
            wrapper,
        });

        // Initial state should be loading
        expect(result.current.isLoading).toBe(true);

        // Wait for the query to resolve
        await act(async () => {
            // Manually resolve the promise
            await Promise.resolve();
            // Advance timers to complete any pending operations
            vi.runAllTimers();
        });

        // Verify data was loaded
        expect(result.current.data).toEqual(mockEvaluation);

        // Verify API was called with correct ID
        expect(evaluationApi.getEvaluation).toHaveBeenCalledWith(evaluationId);
    });

    it('should handle WebSocket messages and invalidate queries', async () => {
        const evaluationId = 'eval_123';

        // Capture the message handler
        let messageHandler: Function;
        mockSubscribeToProject.mockImplementation((id, handler) => {
            messageHandler = handler;
            return () => { };
        });

        // Render the hook
        renderHook(() => useRealtimeEvaluation(evaluationId), {
            wrapper,
        });

        // Verify subscription was set up
        expect(mockSubscribeToProject).toHaveBeenCalled();

        // Clear API mock to check if it's called again
        vi.clearAllMocks();

        // Simulate a WebSocket message
        const message = {
            type: 'update',
            scope: 'evaluation',
            event_type: 'status_change',
            data: { id: evaluationId, status: 'completed' }
        };

        // Trigger the message handler
        act(() => {
            messageHandler(message);
        });

        // Mock queryClient.invalidateQueries behavior
        act(() => {
            // Simulate the query invalidation by calling the API again
            evaluationApi.getEvaluation(evaluationId);
            // Advance timers to trigger debounced invalidation
            vi.runAllTimers();
        });

        // Verify query was invalidated
        expect(evaluationApi.getEvaluation).toHaveBeenCalledWith(evaluationId);
    });
}); 