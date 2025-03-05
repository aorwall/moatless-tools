import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { getSubscriptionManager } from '../subscriptionManager';
import { WebSocketMessageType, SubscriptionType, MessageHandler } from '../types';

// Mock the message store and connection manager
vi.mock('../messageStore', () => ({
    getMessageStore: vi.fn(() => ({
        getState: () => ({
            subscribe: vi.fn(() => vi.fn()),
            subscribers: {}
        })
    }))
}));

vi.mock('../connectionManager', () => ({
    getConnectionManager: vi.fn(() => ({
        getState: () => ({
            sendMessage: vi.fn().mockResolvedValue(true)
        })
    }))
}));

// Mock the store state and methods
const mockState = {
    activeProjectSubscriptions: new Set<string>(),
    activeTrajectorySubscriptions: new Set<any>(),
    sendSubscriptionMessage: vi.fn().mockResolvedValue(true),
    subscribeToProject: vi.fn().mockImplementation((projectId: string, callback: MessageHandler) => {
        mockState.serverSubscribeToProject(projectId);
        return vi.fn();
    }),
    subscribeToTrajectory: vi.fn().mockImplementation((projectId: string, trajectoryId: string, callback: MessageHandler) => {
        mockState.serverSubscribeToTrajectory(projectId, trajectoryId);
        return vi.fn();
    }),
    serverSubscribeToProject: vi.fn().mockImplementation(async (projectId: string) => {
        mockState.activeProjectSubscriptions.add(projectId);
        return true;
    }),
    serverSubscribeToTrajectory: vi.fn().mockImplementation(async (projectId: string, trajectoryId: string) => {
        mockState.activeTrajectorySubscriptions.add({ projectId, trajectoryId });
        return true;
    }),
    serverUnsubscribeFromProject: vi.fn().mockImplementation(async (projectId: string) => {
        mockState.activeProjectSubscriptions.delete(projectId);
        await mockState.sendSubscriptionMessage({
            type: WebSocketMessageType.UNSUBSCRIBE,
            subscription: SubscriptionType.PROJECT,
            project_id: projectId
        });
        return true;
    }),
    serverUnsubscribeFromTrajectory: vi.fn().mockImplementation(async (projectId: string, trajectoryId: string) => {
        const subscription = Array.from(mockState.activeTrajectorySubscriptions).find(
            sub => sub.projectId === projectId && sub.trajectoryId === trajectoryId
        );
        if (subscription) {
            mockState.activeTrajectorySubscriptions.delete(subscription);
        }
        await mockState.sendSubscriptionMessage({
            type: WebSocketMessageType.UNSUBSCRIBE,
            subscription: SubscriptionType.TRAJECTORY,
            project_id: projectId,
            trajectory_id: trajectoryId
        });
        return true;
    }),
    setActiveProjectSubscriptions: vi.fn(),
    setActiveTrajectorySubscriptions: vi.fn()
};

// Mock the store creation
vi.mock('../subscriptionManager', () => ({
    getSubscriptionManager: vi.fn(() => ({
        getState: () => mockState,
        setState: (fn: (state: typeof mockState) => Partial<typeof mockState>) => {
            Object.assign(mockState, fn(mockState));
        }
    }))
}));

describe('SubscriptionManager', () => {
    let store: ReturnType<typeof getSubscriptionManager>;
    const mockCallback = vi.fn() as MessageHandler;

    beforeEach(() => {
        store = getSubscriptionManager();
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.clearAllMocks();
        mockState.activeProjectSubscriptions.clear();
        mockState.activeTrajectorySubscriptions.clear();
    });

    it('should subscribe to project events', async () => {
        const projectId = 'test-project';
        const state = store.getState();
        const unsubscribe = state.subscribeToProject(projectId, mockCallback);

        expect(state.serverSubscribeToProject).toHaveBeenCalledWith(projectId);
        expect(typeof unsubscribe).toBe('function');
    });

    it('should subscribe to trajectory events', async () => {
        const projectId = 'test-project';
        const trajectoryId = 'test-trajectory';
        const state = store.getState();
        const unsubscribe = state.subscribeToTrajectory(projectId, trajectoryId, mockCallback);

        expect(state.serverSubscribeToTrajectory).toHaveBeenCalledWith(projectId, trajectoryId);
        expect(typeof unsubscribe).toBe('function');
    });

    it('should unsubscribe from project events', async () => {
        const projectId = 'test-project';
        const state = store.getState();

        // First subscribe to have something to unsubscribe from
        await state.serverSubscribeToProject(projectId);
        expect(state.activeProjectSubscriptions.has(projectId)).toBe(true);

        // Then unsubscribe
        await state.serverUnsubscribeFromProject(projectId);

        expect(state.sendSubscriptionMessage).toHaveBeenCalledWith({
            type: WebSocketMessageType.UNSUBSCRIBE,
            subscription: SubscriptionType.PROJECT,
            project_id: projectId
        });
        expect(state.activeProjectSubscriptions.has(projectId)).toBe(false);
    });

    it('should unsubscribe from trajectory events', async () => {
        const projectId = 'test-project';
        const trajectoryId = 'test-trajectory';
        const state = store.getState();

        // First subscribe to have something to unsubscribe from
        await state.serverSubscribeToTrajectory(projectId, trajectoryId);
        expect(state.activeTrajectorySubscriptions.size).toBe(1);

        // Then unsubscribe
        await state.serverUnsubscribeFromTrajectory(projectId, trajectoryId);

        expect(state.sendSubscriptionMessage).toHaveBeenCalledWith({
            type: WebSocketMessageType.UNSUBSCRIBE,
            subscription: SubscriptionType.TRAJECTORY,
            project_id: projectId,
            trajectory_id: trajectoryId
        });
        expect(state.activeTrajectorySubscriptions.size).toBe(0);
    });

    it('should manage active subscriptions', () => {
        const projectId = 'test-project';
        const state = store.getState();
        const newProjectSubscriptions = new Set([projectId]);

        state.setActiveProjectSubscriptions(newProjectSubscriptions);
        expect(state.setActiveProjectSubscriptions).toHaveBeenCalledWith(newProjectSubscriptions);
    });
}); 