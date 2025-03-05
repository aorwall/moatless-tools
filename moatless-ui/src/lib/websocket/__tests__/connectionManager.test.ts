import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { getConnectionManager } from '../connectionManager';
import { ConnectionState, WebSocketMessageType } from '../types';

// Mock WebSocket
class MockWebSocket {
    private listeners: Record<string, Function[]> = {};
    public readyState: number = WebSocket.CONNECTING;
    public url: string;

    constructor(url: string) {
        this.url = url;
    }

    addEventListener(event: string, callback: Function) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    removeEventListener(event: string, callback: Function) {
        if (this.listeners[event]) {
            this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
        }
    }

    send(data: string) {
        // Mock send
    }

    close() {
        this.readyState = WebSocket.CLOSED;
        this.triggerEvent('close');
    }

    triggerEvent(event: string, data?: any) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }

    simulateOpen() {
        this.readyState = WebSocket.OPEN;
        this.triggerEvent('open');
    }

    simulateError() {
        this.triggerEvent('error', new Error('WebSocket error'));
    }

    simulateMessage(data: any) {
        this.triggerEvent('message', { data: JSON.stringify(data) });
    }
}

describe('ConnectionManager', () => {
    let connectionManager: ReturnType<typeof getConnectionManager>;
    let mockWebSocket: MockWebSocket;

    beforeEach(() => {
        vi.useFakeTimers();
        // Mock WebSocket global
        vi.stubGlobal('WebSocket', MockWebSocket);
        connectionManager = getConnectionManager();
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
        connectionManager.disconnect();
    });

    it('should initialize in disconnected state', () => {
        expect(connectionManager.getStatus()).toBe(ConnectionState.DISCONNECTED);
    });

    it('should transition to connecting state when connect is called', () => {
        connectionManager.connect();
        expect(connectionManager.getStatus()).toBe(ConnectionState.CONNECTING);
    });

    it('should transition to connected state when WebSocket opens', () => {
        connectionManager.connect();
        mockWebSocket = (connectionManager as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateOpen();
        expect(connectionManager.getStatus()).toBe(ConnectionState.CONNECTED);
    });

    it('should handle connection errors', () => {
        connectionManager.connect();
        mockWebSocket = (connectionManager as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateError();
        expect(connectionManager.getStatus()).toBe(ConnectionState.ERROR);
    });

    it('should attempt reconnection after error', () => {
        connectionManager.connect();
        mockWebSocket = (connectionManager as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateError();
        vi.advanceTimersByTime(1000); // Initial retry delay
        expect(connectionManager.getStatus()).toBe(ConnectionState.RECONNECTING);
    });

    it('should handle ping/pong messages', () => {
        connectionManager.connect();
        mockWebSocket = (connectionManager as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateOpen();

        // Simulate ping message
        mockWebSocket.simulateMessage({ type: WebSocketMessageType.PING });

        // Check if pong was sent (this would require spying on mockWebSocket.send)
        const sendSpy = vi.spyOn(mockWebSocket, 'send');
        expect(sendSpy).toHaveBeenCalledWith(JSON.stringify({ type: WebSocketMessageType.PONG }));
    });

    it('should notify status listeners of state changes', () => {
        const statusListener = vi.fn();
        connectionManager.addStatusListener(statusListener);

        connectionManager.connect();
        mockWebSocket = (connectionManager as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateOpen();

        expect(statusListener).toHaveBeenCalledWith(ConnectionState.CONNECTING);
        expect(statusListener).toHaveBeenCalledWith(ConnectionState.CONNECTED);
    });

    it('should clean up resources on disconnect', () => {
        connectionManager.connect();
        mockWebSocket = (connectionManager as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateOpen();

        const closeSpy = vi.spyOn(mockWebSocket, 'close');
        connectionManager.disconnect();

        expect(closeSpy).toHaveBeenCalled();
        expect(connectionManager.getStatus()).toBe(ConnectionState.DISCONNECTED);
    });

    it('should handle exponential backoff for reconnection attempts', () => {
        connectionManager.connect();
        mockWebSocket = (connectionManager as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateError();

        // First retry
        vi.advanceTimersByTime(1000);
        expect(connectionManager.getStatus()).toBe(ConnectionState.RECONNECTING);

        // Second retry (should be delayed by 1.5x)
        mockWebSocket.simulateError();
        vi.advanceTimersByTime(1500);
        expect(connectionManager.getStatus()).toBe(ConnectionState.RECONNECTING);
    });
}); 