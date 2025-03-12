import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { getConnectionManager } from '../connectionManager';
import { ConnectionState, WebSocketMessageType } from '../types';

class MockWebSocket {
    private listeners: Record<string, Function[]> = {};
    public readyState: number = WebSocket.CONNECTING;
    public url: string;
    public onopen: Function | null = null;
    public onclose: Function | null = null;
    public onerror: Function | null = null;
    public onmessage: Function | null = null;

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
        this.triggerEvent('close', { code: 1000 });
    }

    triggerEvent(event: string, data?: any) {
        // Call the on* handler if it exists
        if (event === 'open' && this.onopen) {
            this.onopen(data);
        } else if (event === 'close' && this.onclose) {
            this.onclose(data);
        } else if (event === 'error' && this.onerror) {
            this.onerror(data);
        } else if (event === 'message' && this.onmessage) {
            this.onmessage(data);
        }

        // Call event listeners
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
    let connectionManagerStore: ReturnType<typeof getConnectionManager>;
    let mockWebSocket: MockWebSocket;

    beforeEach(() => {
        vi.useFakeTimers();
        // Mock WebSocket global
        vi.stubGlobal('WebSocket', MockWebSocket);
        connectionManagerStore = getConnectionManager();
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
        connectionManagerStore.getState().disconnect();
    });

    it('should initialize in disconnected state', () => {
        expect(connectionManagerStore.getState().getStatus()).toBe(ConnectionState.DISCONNECTED);
    });

    it('should transition to connecting state when connect is called', () => {
        connectionManagerStore.getState().connect();
        expect(connectionManagerStore.getState().getStatus()).toBe(ConnectionState.CONNECTING);
    });

    it('should transition to connected state when WebSocket opens', () => {
        connectionManagerStore.getState().connect();
        mockWebSocket = (connectionManagerStore.getState() as any).connection.ws as MockWebSocket;

        // Use vi.spyOn to verify the updateConnectionStatus is called
        const updateSpy = vi.spyOn(connectionManagerStore.getState(), 'updateConnectionStatus');

        mockWebSocket.simulateOpen();

        // Force state update to be processed
        vi.runAllTimers();

        expect(updateSpy).toHaveBeenCalledWith(ConnectionState.CONNECTED);
        expect(connectionManagerStore.getState().getStatus()).toBe(ConnectionState.CONNECTED);
    });

    it('should handle connection errors', () => {
        connectionManagerStore.getState().connect();
        mockWebSocket = (connectionManagerStore.getState() as any).connection.ws as MockWebSocket;

        // Use vi.spyOn to verify the updateConnectionStatus is called
        const updateSpy = vi.spyOn(connectionManagerStore.getState(), 'updateConnectionStatus');

        mockWebSocket.simulateError();

        // Force state update to be processed
        vi.runAllTimers();

        expect(updateSpy).toHaveBeenCalledWith(ConnectionState.ERROR, expect.any(String));
        expect(connectionManagerStore.getState().getStatus()).toBe(ConnectionState.ERROR);
    });

    it('should attempt reconnection after error', () => {
        connectionManagerStore.getState().connect();
        mockWebSocket = (connectionManagerStore.getState() as any).connection.ws as MockWebSocket;

        // Spy on reconnect method
        const reconnectSpy = vi.spyOn(connectionManagerStore.getState(), 'reconnect');

        mockWebSocket.simulateError();

        // Force error handlers to run
        vi.runAllTimers();

        // Verify reconnect was called
        expect(reconnectSpy).toHaveBeenCalled();

        // Advance time to trigger reconnection
        vi.advanceTimersByTime(1000);

        // Check if status is now RECONNECTING
        expect(connectionManagerStore.getState().getStatus()).toBe(ConnectionState.RECONNECTING);
    });

    it('should handle ping/pong messages', () => {
        connectionManagerStore.getState().connect();
        mockWebSocket = (connectionManagerStore.getState() as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateOpen();

        // Run timers to ensure connection is established
        vi.runAllTimers();

        // Spy on send method before simulating message
        const sendSpy = vi.spyOn(mockWebSocket, 'send');

        // Simulate ping message
        mockWebSocket.simulateMessage({ type: WebSocketMessageType.PING });

        // Run any pending promises/timers
        vi.runAllTimers();

        // Check if pong was sent
        expect(sendSpy).toHaveBeenCalledWith(JSON.stringify({ type: WebSocketMessageType.PONG }));
    });

    it('should notify status listeners of state changes', () => {
        const statusListener = vi.fn();
        connectionManagerStore.getState().addStatusListener(statusListener);

        // Clear initial call with DISCONNECTED state
        statusListener.mockClear();

        connectionManagerStore.getState().connect();

        // Verify CONNECTING state was reported
        expect(statusListener).toHaveBeenCalledWith(ConnectionState.CONNECTING, undefined);

        mockWebSocket = (connectionManagerStore.getState() as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateOpen();

        // Run timers to ensure all handlers execute
        vi.runAllTimers();

        // Verify CONNECTED state was reported
        expect(statusListener).toHaveBeenCalledWith(ConnectionState.CONNECTED, undefined);
    });

    it('should clean up resources on disconnect', () => {
        connectionManagerStore.getState().connect();
        mockWebSocket = (connectionManagerStore.getState() as any).connection.ws as MockWebSocket;
        mockWebSocket.simulateOpen();

        // Run timers to ensure connection is established
        vi.runAllTimers();

        const closeSpy = vi.spyOn(mockWebSocket, 'close');
        connectionManagerStore.getState().disconnect();

        expect(closeSpy).toHaveBeenCalled();
        expect(connectionManagerStore.getState().getStatus()).toBe(ConnectionState.DISCONNECTED);
    });

    it('should handle exponential backoff for reconnection attempts', () => {
        connectionManagerStore.getState().connect();
        mockWebSocket = (connectionManagerStore.getState() as any).connection.ws as MockWebSocket;

        // Spy on reconnect method
        const reconnectSpy = vi.spyOn(connectionManagerStore.getState(), 'reconnect');

        mockWebSocket.simulateError();

        // Force error handlers to run
        vi.runAllTimers();

        // Verify reconnect was called
        expect(reconnectSpy).toHaveBeenCalled();

        // Advance time to trigger first reconnection
        vi.advanceTimersByTime(1000);

        // Check if status is now RECONNECTING
        expect(connectionManagerStore.getState().getStatus()).toBe(ConnectionState.RECONNECTING);

        // Get the new WebSocket instance after reconnection
        mockWebSocket = (connectionManagerStore.getState() as any).connection.ws as MockWebSocket;

        // Simulate another error to trigger second reconnection attempt
        mockWebSocket.simulateError();

        // Run timers to process error
        vi.runAllTimers();

        // Advance time for second retry (with backoff)
        vi.advanceTimersByTime(1500);

        // Check if status is still RECONNECTING
        expect(connectionManagerStore.getState().getStatus()).toBe(ConnectionState.RECONNECTING);
    });
}); 