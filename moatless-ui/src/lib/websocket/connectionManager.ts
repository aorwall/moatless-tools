import { create } from 'zustand';
import {
    ConnectionState,
    WebSocketConnection,
    StatusChangeListener,
    WebSocketMessageType,
    getWebSocketUrl,
    WS_CONFIG
} from './types';

export interface ConnectionManager {
    connection: WebSocketConnection | null;
    connect: () => void;
    disconnect: () => void;
    reconnect: () => void;
    isConnected: () => boolean;
    isConnecting: () => boolean;
    hasError: () => boolean;
    getStatus: () => ConnectionState;
    getError: () => string | undefined;
    getConnection: () => WebSocketConnection | null;
    retryCount: number;
    statusListeners: Set<StatusChangeListener>;
    addStatusListener: (listener: StatusChangeListener) => () => void;
    updateConnectionStatus: (status: ConnectionState, error?: string) => void;
    sendMessage: (message: any) => Promise<boolean>;
}

interface ConnectionManagerStore extends ConnectionManager {
    setConnection: (connection: WebSocketConnection | null) => void;
    setRetryCount: (count: number) => void;
}

// Create the store without immediately initializing it
export const createConnectionManager = () => {
    const store = create<ConnectionManagerStore>((set, get) => ({
        connection: null,
        retryCount: 0,
        statusListeners: new Set<StatusChangeListener>(),

        setConnection: (connection) => set({ connection }),
        setRetryCount: (count) => set({ retryCount: count }),

        isConnected: () => get().getStatus() === ConnectionState.CONNECTED,
        isConnecting: () => {
            const status = get().getStatus();
            return status === ConnectionState.CONNECTING || status === ConnectionState.RECONNECTING;
        },
        hasError: () => get().getStatus() === ConnectionState.ERROR,

        getStatus: () => get().connection?.status || ConnectionState.DISCONNECTED,
        getError: () => get().connection?.error,
        getConnection: () => get().connection,

        addStatusListener: (listener: StatusChangeListener) => {
            get().statusListeners.add(listener);

            // Immediately notify with current status
            const currentStatus = get().getStatus();
            const currentError = get().getError();
            listener(currentStatus, currentError);

            return () => {
                get().statusListeners.delete(listener);
            };
        },

        updateConnectionStatus: (status: ConnectionState, error?: string) => {
            const { connection, statusListeners } = get();

            if (connection) {
                set({
                    connection: {
                        ...connection,
                        status,
                        error
                    }
                });
            }

            // Notify all listeners
            statusListeners.forEach(listener => {
                listener(status, error);
            });
        },

        connect: () => {
            const { connection, disconnect } = get();

            // If already connected or connecting, do nothing
            if (connection && (
                connection.status === ConnectionState.CONNECTED ||
                connection.status === ConnectionState.CONNECTING
            )) {
                return;
            }

            // Clean up any existing connection before creating a new one
            if (connection) {
                disconnect();
            }

            get().updateConnectionStatus(ConnectionState.CONNECTING);

            try {
                const ws = new WebSocket(getWebSocketUrl());

                const newConnection: WebSocketConnection = {
                    ws,
                    status: ConnectionState.CONNECTING,
                };

                // Set up connection timeout
                const connectionTimeout = setTimeout(() => {
                    if (get().connection?.status === ConnectionState.CONNECTING) {
                        get().updateConnectionStatus(
                            ConnectionState.ERROR,
                            "Connection timed out. Please check your internet connection and try again."
                        );
                        disconnect();
                    }
                }, WS_CONFIG.CONNECTION_TIMEOUT);

                newConnection.connectionTimeout = connectionTimeout;
                get().setConnection(newConnection);

                ws.onopen = () => {
                    const { connection } = get();
                    if (!connection) return;

                    // Clear connection timeout
                    if (connection.connectionTimeout) {
                        clearTimeout(connection.connectionTimeout);
                    }

                    get().updateConnectionStatus(ConnectionState.CONNECTED);
                    get().setRetryCount(0);

                    // Set up ping interval for keeping connection alive
                    const pingInterval = setInterval(() => {
                        const heartbeat = () => {
                            const { connection } = get();
                            if (!connection || connection.status !== ConnectionState.CONNECTED) return;

                            try {
                                const pingMessage = { type: WebSocketMessageType.PING };
                                connection.ws.send(JSON.stringify(pingMessage));
                                connection.lastPingSent = Date.now();

                                // Set pong timeout
                                const pongTimeout = setTimeout(() => {
                                    const { connection } = get();
                                    if (!connection) return;

                                    // If we haven't received a pong since our last ping
                                    if (
                                        connection.lastPingSent &&
                                        (!connection.lastPongReceived || connection.lastPongReceived < connection.lastPingSent)
                                    ) {
                                        get().updateConnectionStatus(
                                            ConnectionState.ERROR,
                                            "Server not responding. Attempting to reconnect..."
                                        );
                                        get().reconnect();
                                    }
                                }, WS_CONFIG.PONG_TIMEOUT);

                                // Store the timeout so we can clear it if needed
                                set({
                                    connection: {
                                        ...connection,
                                        pongTimeout
                                    }
                                });
                            } catch (error) {
                                console.error("Error sending ping:", error);
                                get().reconnect();
                            }
                        };

                        heartbeat();
                    }, WS_CONFIG.PING_INTERVAL);

                    // Store the interval so we can clear it on disconnect
                    set({
                        connection: {
                            ...get().connection!,
                            pingInterval
                        }
                    });
                };

                ws.onclose = (event) => {
                    const { connection } = get();
                    if (!connection) return;

                    const wasConnected = connection.status === ConnectionState.CONNECTED;

                    // Only show error if we were previously connected and it wasn't a normal closure
                    if (wasConnected && event.code !== 1000) {
                        get().updateConnectionStatus(
                            ConnectionState.ERROR,
                            "Connection closed unexpectedly. Attempting to reconnect..."
                        );
                    } else {
                        get().updateConnectionStatus(ConnectionState.DISCONNECTED);
                    }

                    // Try to reconnect if it wasn't an intentional close
                    if (event.code !== 1000) {
                        get().reconnect();
                    }

                    // Clean up
                    get().disconnect();
                };

                ws.onerror = (error) => {
                    console.error("WebSocket error:", error);
                    get().updateConnectionStatus(
                        ConnectionState.ERROR,
                        "Failed to connect to server. Please check your internet connection."
                    );

                    // Clean up
                    get().disconnect();

                    // Try to reconnect
                    get().reconnect();
                };

                ws.onmessage = (event) => {
                    const { connection } = get();
                    if (!connection) return;

                    try {
                        const data = JSON.parse(event.data);

                        // Handle pong messages to track connection health
                        if (data.type === WebSocketMessageType.PONG) {
                            connection.lastPongReceived = Date.now();

                            // Clear any pending pong timeout
                            if (connection.pongTimeout) {
                                clearTimeout(connection.pongTimeout);
                                connection.pongTimeout = undefined;
                            }

                            return;
                        }

                        // Other message types will be handled by the MessageStore
                    } catch (error) {
                        console.error("Error parsing WebSocket message:", error);
                    }
                };
            } catch (error) {
                console.error("Error creating WebSocket connection:", error);
                get().updateConnectionStatus(
                    ConnectionState.ERROR,
                    "Failed to establish connection. Please try again later."
                );
            }
        },

        disconnect: () => {
            const { connection } = get();
            if (!connection) return;

            // Clear all timeouts and intervals
            if (connection.pingInterval) {
                clearInterval(connection.pingInterval);
            }

            if (connection.pongTimeout) {
                clearTimeout(connection.pongTimeout);
            }

            if (connection.connectionTimeout) {
                clearTimeout(connection.connectionTimeout);
            }

            // Close the WebSocket connection if it's open
            if (connection.ws.readyState === WebSocket.OPEN ||
                connection.ws.readyState === WebSocket.CONNECTING) {
                try {
                    connection.ws.close(1000, "Normal closure");
                } catch (error) {
                    console.error("Error closing WebSocket:", error);
                }
            }

            get().setConnection(null);
            get().updateConnectionStatus(ConnectionState.DISCONNECTED);
        },

        reconnect: () => {
            const { retryCount, connect } = get();

            // Exponential backoff for reconnection attempts
            const delay = retryCount < WS_CONFIG.MAX_RETRIES
                ? WS_CONFIG.RETRY_DELAY * Math.pow(WS_CONFIG.RECONNECT_BACKOFF_FACTOR, retryCount)
                : WS_CONFIG.LONG_RETRY_DELAY;

            get().updateConnectionStatus(ConnectionState.RECONNECTING);

            setTimeout(() => {
                get().setRetryCount(retryCount + 1);
                connect();
            }, delay);
        },

        sendMessage: (message): Promise<boolean> => {
            return new Promise((resolve) => {
                const { connection } = get();

                if (!connection || connection.status !== ConnectionState.CONNECTED) {
                    resolve(false);
                    return;
                }

                try {
                    connection.ws.send(JSON.stringify(message));
                    resolve(true);
                } catch (error) {
                    console.error("Error sending message:", error);
                    resolve(false);
                }
            });
        },
    }));

    // Return a function that creates a new instance when called within a component
    return () => store;
};

// Export the store creator function instead of an instance
export const connectionManagerCreator = createConnectionManager();

// Export a function to get or create the store instance
let connectionManagerInstance: ReturnType<typeof connectionManagerCreator> | null = null;

export const getConnectionManager = () => {
    if (!connectionManagerInstance) {
        connectionManagerInstance = connectionManagerCreator();
    }
    return connectionManagerInstance;
}; 