import { useWebSocketStore } from '@/lib/stores/websocketStore';
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useTrajectorySubscription } from '../useTrajectorySubscription';

// Mock the Zustand store
vi.mock('@/lib/stores/websocketStore', () => {
    const subscribeToTrajectory = vi.fn().mockReturnValue(vi.fn());
    const isConnected = vi.fn().mockReturnValue(true);

    return {
        WebSocketMessage: {},
        useWebSocketStore: vi.fn().mockReturnValue({
            subscribeToTrajectory,
            isConnected,
        }),
    };
});

// Mock the React Query useQueryClient hook
vi.mock('@tanstack/react-query', () => {
    return {
        QueryClient: vi.fn(),
        QueryClientProvider: ({ children }: { children: React.ReactNode }) => children,
        useQueryClient: vi.fn().mockReturnValue({
            invalidateQueries: vi.fn(),
        }),
    };
});

describe('useTrajectorySubscription', () => {
    const mockProjectId = 'project-123';
    const mockTrajectoryId = 'trajectory-456';
    const mockOnEvent = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.resetAllMocks();
    });

    it('should subscribe to trajectory when mounted', () => {
        // Arrange
        const mockSubscribeToTrajectory = vi.fn().mockReturnValue(vi.fn());
        (useWebSocketStore as any).mockReturnValue({
            subscribeToTrajectory: mockSubscribeToTrajectory,
            isConnected: vi.fn().mockReturnValue(true),
        });

        // Act
        renderHook(() => useTrajectorySubscription(mockTrajectoryId, mockProjectId));

        // Assert
        expect(mockSubscribeToTrajectory).toHaveBeenCalledTimes(1);
        expect(mockSubscribeToTrajectory).toHaveBeenCalledWith(
            mockProjectId,
            mockTrajectoryId,
            expect.any(Function)
        );
    });

    it('should not subscribe if trajectoryId or projectId is missing', () => {
        // Arrange
        const mockSubscribeToTrajectory = vi.fn().mockReturnValue(vi.fn());
        (useWebSocketStore as any).mockReturnValue({
            subscribeToTrajectory: mockSubscribeToTrajectory,
            isConnected: vi.fn().mockReturnValue(true),
        });

        // Act - missing trajectoryId
        renderHook(() => useTrajectorySubscription('', mockProjectId));

        // Assert
        expect(mockSubscribeToTrajectory).not.toHaveBeenCalled();

        // Act - missing projectId
        renderHook(() => useTrajectorySubscription(mockTrajectoryId, ''));

        // Assert
        expect(mockSubscribeToTrajectory).not.toHaveBeenCalled();
    });

    it('should call onEvent callback when message is received', () => {
        // Arrange
        let messageHandler: Function;
        const mockSubscribeToTrajectory = vi.fn().mockImplementation((projectId, trajectoryId, handler) => {
            messageHandler = handler;
            return vi.fn();
        });

        (useWebSocketStore as any).mockReturnValue({
            subscribeToTrajectory: mockSubscribeToTrajectory,
            isConnected: vi.fn().mockReturnValue(true),
        });

        // Act
        renderHook(() => useTrajectorySubscription(mockTrajectoryId, mockProjectId, {
            onEvent: mockOnEvent,
        }));

        // Simulate a message
        const mockMessage = { type: 'test', project_id: mockProjectId, trajectory_id: mockTrajectoryId };
        act(() => {
            messageHandler(mockMessage);
        });

        // Assert
        expect(mockOnEvent).toHaveBeenCalledWith(mockMessage);
    });

    it('should unsubscribe when unmounted', () => {
        // Arrange
        const mockUnsubscribe = vi.fn();
        const mockSubscribeToTrajectory = vi.fn().mockReturnValue(mockUnsubscribe);

        (useWebSocketStore as any).mockReturnValue({
            subscribeToTrajectory: mockSubscribeToTrajectory,
            isConnected: vi.fn().mockReturnValue(true),
        });

        // Act
        const { unmount } = renderHook(() => useTrajectorySubscription(mockTrajectoryId, mockProjectId));

        // Unmount the hook
        unmount();

        // Assert
        expect(mockUnsubscribe).toHaveBeenCalledTimes(1);
    });
}); 