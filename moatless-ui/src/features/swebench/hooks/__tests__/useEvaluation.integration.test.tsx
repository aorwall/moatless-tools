import { renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRealtimeEvaluation } from '../useEvaluation';
import { useWebSocketStore } from '@/lib/stores/websocketStore';
import { evaluationApi } from '../../api/evaluation';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
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

describe('useRealtimeEvaluation Integration', () => {
    let queryClient: QueryClient;
    const sentMessages: any[] = [];
    const mockSubscribeToProject = vi.fn().mockImplementation((projectId, callback) => {
        // Record the subscription message
        sentMessages.push({
            type: 'subscribe',
            subscription: 'project',
            project_id: projectId,
        });
        return () => { };
    });

    const mockEvaluation = {
        id: 'eval_123',
        evaluation_name: 'Test Evaluation',
        status: 'running',
        instances: [],
        created_at: new Date().toISOString(),
    };

    beforeEach(() => {
        vi.clearAllMocks();
        sentMessages.length = 0;

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
            subscribeToTrajectory: vi.fn(),
            getConnectionStatus: vi.fn().mockReturnValue('CONNECTED'),
            addStatusListener: vi.fn().mockReturnValue(() => { }),
        });

        // Mock API response
        (evaluationApi.getEvaluation as any).mockResolvedValue(mockEvaluation);
    });

    afterEach(() => {
        queryClient.clear();
    });

    const wrapper = ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>
            {children}
        </QueryClientProvider>
    );

    it('should send a project subscription message', async () => {
        const evaluationId = 'eval_123';

        // Render the hook
        renderHook(() => useRealtimeEvaluation(evaluationId), {
            wrapper,
        });

        // Verify subscribeToProject was called
        expect(mockSubscribeToProject).toHaveBeenCalledWith(
            evaluationId,
            expect.any(Function)
        );

        // Verify a subscription message was recorded
        expect(sentMessages.length).toBeGreaterThan(0);

        // Find the subscription message
        const subscriptionMessage = sentMessages.find(
            msg => msg.subscription === 'project' && msg.project_id === evaluationId
        );

        // Verify the subscription details
        expect(subscriptionMessage).toBeDefined();
        expect(subscriptionMessage).toEqual({
            type: 'subscribe',
            subscription: 'project',
            project_id: evaluationId,
        });

        // Verify there's no trajectory_id in the message
        expect(subscriptionMessage).not.toHaveProperty('trajectory_id');
    });
}); 