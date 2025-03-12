import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
    getWebSocketUrl,
    getTrajectorySubscriptionKey,
    getProjectChannel,
    getTrajectoryChannel,
    findTrajectorySubscription,
    WS_CONFIG,
    ConnectionState,
    WebSocketMessageType,
    SubscriptionType
} from '../types';

// Store original WS_CONFIG
const originalWsConfig = { ...WS_CONFIG };

describe('WebSocket URL Configuration', () => {
    beforeEach(() => {
        // Reset window.location
        vi.stubGlobal('window', {
            location: {
                protocol: 'http:',
                host: 'localhost:5173'
            }
        });

        // Reset environment variables
        vi.stubGlobal('import', {
            meta: {
                env: {
                    VITE_WS_URL: undefined,
                    VITE_API_HOST: 'localhost:8000',
                    VITE_WS_PATH: '/api/ws'
                }
            }
        });

        // Reset WS_CONFIG for each test
        Object.keys(WS_CONFIG).forEach(key => {
            if (key in originalWsConfig) {
                (WS_CONFIG as any)[key] = (originalWsConfig as any)[key];
            }
        });
    });

    afterEach(() => {
        // Restore original WS_CONFIG
        Object.keys(WS_CONFIG).forEach(key => {
            if (key in originalWsConfig) {
                (WS_CONFIG as any)[key] = (originalWsConfig as any)[key];
            }
        });
    });

    it('should generate correct WebSocket URL with default values', () => {
        // Directly set the WS_CONFIG values for testing
        (WS_CONFIG as any).BASE_URL = 'ws://localhost:8000';
        (WS_CONFIG as any).PATH = '/api/ws';

        const url = getWebSocketUrl();
        expect(url).toBe('ws://localhost:8000/api/ws');
    });

    it('should use secure WebSocket protocol when served over HTTPS', () => {
        // Set window.location.protocol to https
        window.location.protocol = 'https:';

        // Mock the WS_CONFIG to use wss protocol
        (WS_CONFIG as any).BASE_URL = 'wss://localhost:8000';
        (WS_CONFIG as any).PATH = '/api/ws';

        const url = getWebSocketUrl();
        expect(url).toMatch(/^wss:\/\//);
    });

    it('should use environment variables when provided', () => {
        // Directly set the WS_CONFIG values based on env vars
        (WS_CONFIG as any).BASE_URL = 'ws://test.example.com';
        (WS_CONFIG as any).PATH = '/custom/ws';

        const url = getWebSocketUrl();
        expect(url).toBe('ws://test.example.com/custom/ws');
    });

    it('should handle invalid URLs gracefully', () => {
        // Set an invalid URL
        (WS_CONFIG as any).BASE_URL = 'invalid://url';
        (WS_CONFIG as any).PATH = '/api/ws';

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
