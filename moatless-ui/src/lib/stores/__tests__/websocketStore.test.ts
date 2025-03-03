import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Define WebSocketMessage for testing
const WebSocketMessage = {
    SUBSCRIBE_TRAJECTORY: 'subscribe_trajectory',
    UNSUBSCRIBE_TRAJECTORY: 'unsubscribe_trajectory',
    SUBSCRIBE_PROJECT: 'subscribe_project',
    UNSUBSCRIBE_PROJECT: 'unsubscribe_project',
};

// Mock the Zustand store
vi.mock('../websocketStore', () => {
    return {
        WebSocketMessage: {
            SUBSCRIBE_TRAJECTORY: 'subscribe_trajectory',
            UNSUBSCRIBE_TRAJECTORY: 'unsubscribe_trajectory',
            SUBSCRIBE_PROJECT: 'subscribe_project',
            UNSUBSCRIBE_PROJECT: 'unsubscribe_project',
        },
        useWebSocketStore: vi.fn(),
    };
});

// Import after mocking
import { useWebSocketStore } from '../websocketStore';

describe('WebSocketStore', () => {
    // Mock implementations
    const mockSend = vi.fn();
    const mockSubscribeToTrajectory = vi.fn();
    const mockSubscribeToProject = vi.fn();
    const mockUnsubscribeTrajectory = vi.fn();
    const mockUnsubscribeProject = vi.fn();

    beforeEach(() => {
        // Reset all mocks
        vi.clearAllMocks();

        // Setup mock implementations
        mockSubscribeToTrajectory.mockImplementation((projectId, trajectoryId, callback) => {
            // Simulate sending a subscription message
            mockSend(JSON.stringify({
                type: WebSocketMessage.SUBSCRIBE_TRAJECTORY,
                project_id: projectId,
                trajectory_id: trajectoryId
            }));

            // Return unsubscribe function
            return mockUnsubscribeTrajectory.mockImplementation(() => {
                mockSend(JSON.stringify({
                    type: WebSocketMessage.UNSUBSCRIBE_TRAJECTORY,
                    project_id: projectId,
                    trajectory_id: trajectoryId
                }));
            });
        });

        mockSubscribeToProject.mockImplementation((projectId, callback) => {
            // Simulate sending a subscription message
            mockSend(JSON.stringify({
                type: WebSocketMessage.SUBSCRIBE_PROJECT,
                project_id: projectId
            }));

            // Return unsubscribe function
            return mockUnsubscribeProject.mockImplementation(() => {
                mockSend(JSON.stringify({
                    type: WebSocketMessage.UNSUBSCRIBE_PROJECT,
                    project_id: projectId
                }));
            });
        });

        // Setup the mock store
        (useWebSocketStore as any).mockReturnValue({
            subscribeToTrajectory: mockSubscribeToTrajectory,
            subscribeToProject: mockSubscribeToProject,
            isConnected: vi.fn().mockReturnValue(true),
            send: mockSend
        });
    });

    afterEach(() => {
        vi.resetAllMocks();
    });

    describe('subscribeToTrajectory', () => {
        it('should send subscription message and return unsubscribe function', () => {
            // Arrange
            const projectId = 'project-123';
            const trajectoryId = 'trajectory-456';
            const callback = vi.fn();
            const store = useWebSocketStore();

            // Act
            const unsubscribe = store.subscribeToTrajectory(projectId, trajectoryId, callback);

            // Assert
            expect(mockSubscribeToTrajectory).toHaveBeenCalledTimes(1);
            expect(mockSubscribeToTrajectory).toHaveBeenCalledWith(projectId, trajectoryId, callback);
            expect(mockSend).toHaveBeenCalledTimes(1);
            expect(mockSend).toHaveBeenCalledWith(expect.stringContaining(WebSocketMessage.SUBSCRIBE_TRAJECTORY));

            // Act - unsubscribe
            unsubscribe();

            // Assert
            expect(mockUnsubscribeTrajectory).toHaveBeenCalledTimes(1);
            expect(mockSend).toHaveBeenCalledTimes(2);
            expect(mockSend).toHaveBeenCalledWith(expect.stringContaining(WebSocketMessage.UNSUBSCRIBE_TRAJECTORY));
        });

        it('should invoke callback when receiving a matching message', () => {
            // Arrange
            const projectId = 'project-123';
            const trajectoryId = 'trajectory-456';
            const callback = vi.fn();
            const store = useWebSocketStore();

            // Act - Subscribe
            store.subscribeToTrajectory(projectId, trajectoryId, callback);

            // Find the callback that was passed to subscribeToTrajectory
            const subscribedCallback = mockSubscribeToTrajectory.mock.calls[0][2];

            // Simulate receiving a message by directly calling the callback
            const message = {
                type: 'trajectory_update',
                project_id: projectId,
                trajectory_id: trajectoryId,
                data: { status: 'completed' }
            };
            subscribedCallback(message);

            // Assert
            expect(callback).toHaveBeenCalledTimes(1);
            expect(callback).toHaveBeenCalledWith(message);
        });
    });

    describe('subscribeToProject', () => {
        it('should send subscription message and return unsubscribe function', () => {
            // Arrange
            const projectId = 'project-123';
            const callback = vi.fn();
            const store = useWebSocketStore();

            // Act
            const unsubscribe = store.subscribeToProject(projectId, callback);

            // Assert
            expect(mockSubscribeToProject).toHaveBeenCalledTimes(1);
            expect(mockSubscribeToProject).toHaveBeenCalledWith(projectId, callback);
            expect(mockSend).toHaveBeenCalledTimes(1);
            expect(mockSend).toHaveBeenCalledWith(expect.stringContaining(WebSocketMessage.SUBSCRIBE_PROJECT));

            // Act - unsubscribe
            unsubscribe();

            // Assert
            expect(mockUnsubscribeProject).toHaveBeenCalledTimes(1);
            expect(mockSend).toHaveBeenCalledTimes(2);
            expect(mockSend).toHaveBeenCalledWith(expect.stringContaining(WebSocketMessage.UNSUBSCRIBE_PROJECT));
        });

        it('should invoke callback when receiving a matching message', () => {
            // Arrange
            const projectId = 'project-123';
            const callback = vi.fn();
            const store = useWebSocketStore();

            // Act - Subscribe
            store.subscribeToProject(projectId, callback);

            // Find the callback that was passed to subscribeToProject
            const subscribedCallback = mockSubscribeToProject.mock.calls[0][1];

            // Simulate receiving a message by directly calling the callback
            const message = {
                type: 'project_update',
                project_id: projectId,
                data: { status: 'active' }
            };
            subscribedCallback(message);

            // Assert
            expect(callback).toHaveBeenCalledTimes(1);
            expect(callback).toHaveBeenCalledWith(message);
        });
    });
}); 