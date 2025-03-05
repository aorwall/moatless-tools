import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
    getWebSocketUrl,
    getDefaultWebSocketUrl,
    getTrajectorySubscriptionKey,
    getProjectChannel,
    getTrajectoryChannel,
    findTrajectorySubscription,
    WS_CONFIG,
    ConnectionState,
    WebSocketMessageType,
    SubscriptionType
} from '../types';

describe('WebSocket URL Configuration', () => {
    beforeEach(() => {
        // Reset window.location
        vi.stubGlobal('window', {
            location: {
                protocol: 'http:',
                host: 'localhost:5173'
            }
        });

        // Reset import.meta.env
        vi.stubGlobal('import.meta', {
            env: {
                VITE_WS_URL: undefined,
                VITE_API_HOST: undefined,
                VITE_WS_PATH: undefined
            }
        });
    });

    it('should generate correct WebSocket URL with default values', () => {
        const url = getWebSocketUrl();
        expect(url).toBe('ws://localhost:8000/api/ws');
    });

    it('should use secure WebSocket protocol when served over HTTPS', () => {
        window.location.protocol = 'https:';
        const url = getWebSocketUrl();
        expect(url).toMatch(/^wss:\/\//);
    });

    it('should use environment variables when provided', () => {
        import.meta.env.VITE_WS_URL = 'ws://test.example.com';
        import.meta.env.VITE_WS_PATH = '/custom/ws';
        const url = getWebSocketUrl();
        expect(url).toBe('ws://test.example.com/custom/ws');
    });

    it('should handle invalid URLs gracefully', () => {
        import.meta.env.VITE_WS_URL = 'invalid://url';
        const url = getWebSocketUrl();
        expect(url).toBe('invalid://url/api/ws');
    });
});

describe('Channel and Subscription Helpers', () => {
    it('should generate correct trajectory subscription key', () => {
        const sub = { projectId: 'proj1', trajectoryId: 'traj1' };
        expect(getTrajectorySubscriptionKey(sub)).toBe('proj1:traj1');
    });

    it('should generate correct project channel', () => {
        expect(getProjectChannel('proj1')).toBe('project.proj1');
    });

    it('should generate correct trajectory channel', () => {
        expect(getTrajectoryChannel('proj1', 'traj1')).toBe('project.proj1.trajectory.traj1');
    });

    it('should find trajectory subscription in set', () => {
        const sub1 = { projectId: 'proj1', trajectoryId: 'traj1' };
        const sub2 = { projectId: 'proj2', trajectoryId: 'traj2' };
        const subs = new Set([sub1, sub2]);

        const found = findTrajectorySubscription(subs, 'proj1', 'traj1');
        expect(found).toEqual(sub1);
    });

    it('should return undefined when subscription not found', () => {
        const sub1 = { projectId: 'proj1', trajectoryId: 'traj1' };
        const subs = new Set([sub1]);

        const found = findTrajectorySubscription(subs, 'proj2', 'traj2');
        expect(found).toBeUndefined();
    });
});

describe('Enums and Constants', () => {
    it('should have correct WebSocket message types', () => {
        expect(WebSocketMessageType.PING).toBe('ping');
        expect(WebSocketMessageType.PONG).toBe('pong');
        expect(WebSocketMessageType.SUBSCRIBE).toBe('subscribe');
        expect(WebSocketMessageType.UNSUBSCRIBE).toBe('unsubscribe');
    });

    it('should have correct subscription types', () => {
        expect(SubscriptionType.PROJECT).toBe('project');
        expect(SubscriptionType.TRAJECTORY).toBe('trajectory');
    });

    it('should have correct connection states', () => {
        expect(ConnectionState.DISCONNECTED).toBe('disconnected');
        expect(ConnectionState.CONNECTING).toBe('connecting');
        expect(ConnectionState.CONNECTED).toBe('connected');
        expect(ConnectionState.ERROR).toBe('error');
        expect(ConnectionState.RECONNECTING).toBe('reconnecting');
    });

    it('should have correct WS_CONFIG values', () => {
        expect(WS_CONFIG.RETRY_DELAY).toBe(1000);
        expect(WS_CONFIG.MAX_RETRIES).toBe(5);
        expect(WS_CONFIG.RECONNECT_BACKOFF_FACTOR).toBe(1.5);
        expect(WS_CONFIG.PING_INTERVAL).toBe(15000);
        expect(WS_CONFIG.PONG_TIMEOUT).toBe(5000);
    });
}); 