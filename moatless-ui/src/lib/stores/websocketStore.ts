import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  getConnectionManager,
  getMessageStore,
  getSubscriptionManager,
  ConnectionState,
  WebSocketMessage,
  TrajectorySubscription,
  StatusChangeListener,
  MessageHandler,
  SubscriptionMessage,
  getProjectChannel,
  getTrajectoryChannel,
  getTrajectorySubscriptionKey,
  startWebSocketService
} from '@/lib/websocket';

// Re-export types from the WebSocket module
export type { ConnectionState, WebSocketMessage, TrajectorySubscription, StatusChangeListener };

// Get store instances
const connectionManager = getConnectionManager().getState();
const messageStore = getMessageStore().getState();
const subscriptionManager = getSubscriptionManager().getState();

// Define the WebSocketStore interface
export interface WebSocketStore {
  // Connection management
  connection: ReturnType<typeof connectionManager.getConnection>;
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
  isConnected: () => boolean;
  isConnecting: () => boolean;
  hasError: () => boolean;
  getConnectionStatus: () => ConnectionState;
  getConnectionError: () => string | undefined;
  retryCount: number;
  statusListeners: Set<StatusChangeListener>;
  addStatusListener: (listener: StatusChangeListener) => () => void;
  updateConnectionStatus: (status: ConnectionState, error?: string) => void;

  // Message handling
  batchedMessages: Record<string, WebSocketMessage[]>;
  processingBatch: boolean;
  messages: Record<string, WebSocketMessage[]>;
  addMessage: (message: WebSocketMessage) => void;
  clearMessages: () => void;

  // Subscription management
  subscribers: Record<string, Set<MessageHandler>>;
  subscribe: (channel: string, callback: MessageHandler) => () => void;
  unsubscribe: (channel: string, callback: MessageHandler) => void;
  sendSubscriptionMessage: (message: SubscriptionMessage) => Promise<boolean>;
  serverSubscribeToProject: (projectId: string) => Promise<boolean>;
  serverSubscribeToTrajectory: (projectId: string, trajectoryId: string) => Promise<boolean>;
  serverUnsubscribeFromProject: (projectId: string) => Promise<boolean>;
  serverUnsubscribeFromTrajectory: (projectId: string, trajectoryId: string) => Promise<boolean>;
  activeProjectSubscriptions: Set<string>;
  activeTrajectorySubscriptions: Set<TrajectorySubscription>;

  // Convenience subscriptions
  subscribeToProject: (projectId: string, callback: MessageHandler) => () => void;
  subscribeToTrajectory: (projectId: string, trajectoryId: string, callback: MessageHandler) => () => void;
}

// Create the WebSocketStore as a facade for the underlying modules
export const useWebSocketStore = create<WebSocketStore>()(
  persist(
    () => ({
      // Connection management (delegated to connectionManager)
      connection: connectionManager.getConnection(),
      connect: connectionManager.connect,
      disconnect: connectionManager.disconnect,
      reconnect: connectionManager.reconnect,
      isConnected: connectionManager.isConnected,
      isConnecting: connectionManager.isConnecting,
      hasError: connectionManager.hasError,
      getConnectionStatus: connectionManager.getStatus,
      getConnectionError: connectionManager.getError,
      retryCount: connectionManager.retryCount,
      statusListeners: connectionManager.statusListeners,
      addStatusListener: connectionManager.addStatusListener,
      updateConnectionStatus: connectionManager.updateConnectionStatus,

      // Message handling (delegated to messageStore)
      batchedMessages: messageStore.batchedMessages,
      processingBatch: messageStore.processingBatch,
      messages: messageStore.messages,
      addMessage: messageStore.addMessage,
      clearMessages: messageStore.clearMessages,

      // Subscription management (mixed)
      subscribers: messageStore.subscribers,
      subscribe: messageStore.subscribe,
      unsubscribe: messageStore.unsubscribe,
      sendSubscriptionMessage: subscriptionManager.sendSubscriptionMessage,
      serverSubscribeToProject: subscriptionManager.serverSubscribeToProject,
      serverSubscribeToTrajectory: subscriptionManager.serverSubscribeToTrajectory,
      serverUnsubscribeFromProject: subscriptionManager.serverUnsubscribeFromProject,
      serverUnsubscribeFromTrajectory: subscriptionManager.serverUnsubscribeFromTrajectory,
      activeProjectSubscriptions: subscriptionManager.activeProjectSubscriptions,
      activeTrajectorySubscriptions: subscriptionManager.activeTrajectorySubscriptions,

      // Convenience subscriptions (delegated to subscriptionManager)
      subscribeToProject: subscriptionManager.subscribeToProject,
      subscribeToTrajectory: subscriptionManager.subscribeToTrajectory,
    }),
    { name: "websocket-store" },
  ),
);

// Selectors for easier state extraction
export const selectConnectionStatus = () => (state: WebSocketStore) =>
  state.getConnectionStatus();

export const selectConnectionError = () => (state: WebSocketStore) =>
  state.getConnectionError();

// Create a hook for easier subscription management
export function useWebSocket() {
  const store = useWebSocketStore();

  return {
    // Connection status
    isConnected: store.isConnected,
    connectionStatus: store.getConnectionStatus(),
    connectionError: store.getConnectionError(),
    reconnect: store.reconnect,

    // Type-safe subscription methods (these handle both client and server subscriptions)
    subscribeToProject: store.subscribeToProject,
    subscribeToTrajectory: store.subscribeToTrajectory
  };
}

// Export a function to initialize the WebSocket service that will be called from a React component
export function initializeWebSocketStore() {
  if (!connectionManager.isConnected()) {
    startWebSocketService();
  }
}
