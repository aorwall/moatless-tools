export enum WebSocketMessageType {
    PING = "ping",
    PONG = "pong",
    SUBSCRIBE = "subscribe",
    UNSUBSCRIBE = "unsubscribe",
}

export enum SubscriptionType {
    PROJECT = "project",
    TRAJECTORY = "trajectory",
}

export enum ConnectionState {
    DISCONNECTED = "disconnected",
    CONNECTING = "connecting",
    CONNECTED = "connected",
    ERROR = "error",
    RECONNECTING = "reconnecting"
}

export type WebSocketMessage = {
    trajectory_id?: string;
    project_id?: string;
    // Backend event fields
    scope?: string;
    event_type?: string;
    timestamp?: string;
    data?: Record<string, any>;
    // Legacy fields for backward compatibility
    type?: string;
    message?: string;
    iteration?: number;
    action?: string;
    cost?: number;
};

export type WebSocketConnection = {
    ws: WebSocket;
    status: ConnectionState;
    error?: string;
    lastPingSent?: number;
    lastPongReceived?: number;
    pingInterval?: NodeJS.Timeout;
    pongTimeout?: NodeJS.Timeout;
    connectionTimeout?: NodeJS.Timeout;
};

export type TrajectorySubscription = {
    projectId: string;
    trajectoryId: string;
};

export type StatusChangeListener = (status: ConnectionState, error?: string) => void;

export type MessageHandler = (data: WebSocketMessage) => void;

export type SubscriptionMessage = {
    type: WebSocketMessageType.SUBSCRIBE | WebSocketMessageType.UNSUBSCRIBE;
    subscription: SubscriptionType;
    project_id: string;
    trajectory_id?: string;
};

export type BatchedNotification = {
    channel: string;
    messages: WebSocketMessage[];
};

// Constants for WebSocket configuration
export const WS_CONFIG = {
    // Default to environment variables, fallback to computed values
    BASE_URL: import.meta.env.VITE_WS_URL || getDefaultWebSocketUrl(),
    PATH: import.meta.env.VITE_WS_PATH || '/ws',
    RETRY_DELAY: 1000, // 1 second initial retry delay
    MAX_RETRIES: 5,
    RECONNECT_BACKOFF_FACTOR: 1.5,
    PING_INTERVAL: 15000, // 15 seconds
    PONG_TIMEOUT: 5000, // 5 seconds
    EVENT_BATCH_INTERVAL: 500, // 500ms
    MAX_MESSAGE_QUEUE_SIZE: 100,
    CONNECTION_TIMEOUT: 10000, // 10 seconds
    LONG_RETRY_DELAY: 30000, // 30 seconds
} as const;

// Helper function to compute default WebSocket URL
function getDefaultWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = import.meta.env.VITE_API_HOST || window.location.host;
    return `${protocol}//${host}`;
}

// Get the complete WebSocket URL
export const getWebSocketUrl = (): string => {
    try {
        const url = new URL(WS_CONFIG.PATH, WS_CONFIG.BASE_URL);
        return url.toString();
    } catch (error) {
        console.error('Error constructing WebSocket URL:', error);
        // Fallback to basic URL construction
        return `${WS_CONFIG.BASE_URL}${WS_CONFIG.PATH}`;
    }
};

// Helper functions
export const getTrajectorySubscriptionKey = (sub: TrajectorySubscription): string =>
    `${sub.projectId}:${sub.trajectoryId}`;

export const getProjectChannel = (projectId: string) => `project.${projectId}`;
export const getTrajectoryChannel = (projectId: string, trajectoryId: string): string =>
    `project.${projectId}.trajectory.${trajectoryId}`;

export const findTrajectorySubscription = (
    set: Set<TrajectorySubscription>,
    projectId: string,
    trajectoryId: string
): TrajectorySubscription | undefined => {
    for (const subscription of set) {
        if (
            subscription.projectId === projectId &&
            subscription.trajectoryId === trajectoryId
        ) {
            return subscription;
        }
    }
    return undefined;
}; 