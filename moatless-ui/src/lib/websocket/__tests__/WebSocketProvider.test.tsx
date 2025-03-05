import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, act, waitFor } from '@testing-library/react';
import { WebSocketProvider, useWebSocket } from '../WebSocketProvider';
import { ConnectionState } from '../types';
import { getConnectionManager } from '../connectionManager';

// Mock the connection manager
vi.mock('../connectionManager', () => ({
    getConnectionManager: vi.fn(() => ({
        connect: vi.fn(),
        disconnect: vi.fn(),
        addStatusListener: vi.fn(),
        removeStatusListener: vi.fn(),
        getStatus: vi.fn(() => ConnectionState.DISCONNECTED),
        getError: vi.fn(() => null)
    }))
}));

// Test component that uses the WebSocket context
const TestComponent = () => {
    const { isConnected, connectionState, error, reconnect } = useWebSocket();
    return (
        <div>
            <div data-testid="connection-state">{connectionState}</div>
            <div data-testid="is-connected">{isConnected.toString()}</div>
            {error && <div data-testid="error">{error}</div>}
            <button data-testid="reconnect" onClick={reconnect}>
                Reconnect
            </button>
        </div>
    );
};

describe('WebSocketProvider', () => {
    let mockConnectionManager: ReturnType<typeof getConnectionManager>;
    let statusListener: (status: ConnectionState, error?: string) => void;

    beforeEach(() => {
        vi.clearAllMocks();
        mockConnectionManager = getConnectionManager();
        (mockConnectionManager.addStatusListener as jest.Mock).mockImplementation((listener) => {
            statusListener = listener;
            return () => { };
        });
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    it('should initialize with disconnected state', () => {
        const { getByTestId } = render(
            <WebSocketProvider>
                <TestComponent />
            </WebSocketProvider>
        );

        expect(getByTestId('connection-state').textContent).toBe(ConnectionState.DISCONNECTED);
        expect(getByTestId('is-connected').textContent).toBe('false');
    });

    it('should connect on mount', () => {
        render(
            <WebSocketProvider>
                <TestComponent />
            </WebSocketProvider>
        );

        expect(mockConnectionManager.connect).toHaveBeenCalled();
    });

    it('should update state when connection status changes', async () => {
        const { getByTestId } = render(
            <WebSocketProvider>
                <TestComponent />
            </WebSocketProvider>
        );

        act(() => {
            statusListener(ConnectionState.CONNECTING);
        });
        expect(getByTestId('connection-state').textContent).toBe(ConnectionState.CONNECTING);

        act(() => {
            statusListener(ConnectionState.CONNECTED);
        });
        expect(getByTestId('connection-state').textContent).toBe(ConnectionState.CONNECTED);
        expect(getByTestId('is-connected').textContent).toBe('true');
    });

    it('should handle connection errors', () => {
        const { getByTestId } = render(
            <WebSocketProvider>
                <TestComponent />
            </WebSocketProvider>
        );

        act(() => {
            statusListener(ConnectionState.ERROR, 'Connection failed');
        });

        expect(getByTestId('connection-state').textContent).toBe(ConnectionState.ERROR);
        expect(getByTestId('error').textContent).toBe('Connection failed');
    });

    it('should attempt reconnection when reconnect is called', () => {
        const { getByTestId } = render(
            <WebSocketProvider>
                <TestComponent />
            </WebSocketProvider>
        );

        act(() => {
            getByTestId('reconnect').click();
        });

        expect(mockConnectionManager.connect).toHaveBeenCalledTimes(2); // Once on mount, once on reconnect
    });

    it('should clean up on unmount', () => {
        const { unmount } = render(
            <WebSocketProvider>
                <TestComponent />
            </WebSocketProvider>
        );

        unmount();
        expect(mockConnectionManager.disconnect).toHaveBeenCalled();
    });

    it('should handle multiple status updates', () => {
        const { getByTestId } = render(
            <WebSocketProvider>
                <TestComponent />
            </WebSocketProvider>
        );

        act(() => {
            statusListener(ConnectionState.CONNECTING);
            statusListener(ConnectionState.CONNECTED);
            statusListener(ConnectionState.ERROR, 'Test error');
            statusListener(ConnectionState.RECONNECTING);
        });

        expect(getByTestId('connection-state').textContent).toBe(ConnectionState.RECONNECTING);
    });

    it('should provide connection state to nested components', () => {
        const NestedComponent = () => {
            const { connectionState } = useWebSocket();
            return <div data-testid="nested-state">{connectionState}</div>;
        };

        const { getByTestId } = render(
            <WebSocketProvider>
                <div>
                    <TestComponent />
                    <NestedComponent />
                </div>
            </WebSocketProvider>
        );

        act(() => {
            statusListener(ConnectionState.CONNECTED);
        });

        expect(getByTestId('nested-state').textContent).toBe(ConnectionState.CONNECTED);
    });
}); 