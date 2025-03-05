import { createContext, useContext, useEffect, useState } from "react";
import { useWebSocketStore, initializeWebSocketStore } from "@/lib/stores/websocketStore";
import { ConnectionState } from "@/lib/websocket";
import { toast } from "sonner";

// Define a comprehensive context interface
interface WebSocketContextValue {
  isConnected: boolean;
  connectionState: ConnectionState;
  reconnect: () => void;
  isAttemptingConnection: boolean;
  error?: string;
}

const WebSocketContext = createContext<WebSocketContextValue>({
  isConnected: false,
  connectionState: ConnectionState.DISCONNECTED,
  reconnect: () => { },
  isAttemptingConnection: false
});

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    ConnectionState.DISCONNECTED
  );
  const [error, setError] = useState<string | undefined>();
  const websocketStore = useWebSocketStore();

  useEffect(() => {
    // Initialize WebSocket store and connection
    initializeWebSocketStore();

    // Track if we've had a successful connection
    let hasConnectedBefore = false;

    // Subscribe to connection status changes
    const unsubscribe = websocketStore.addStatusListener(
      (status: ConnectionState, errorMsg?: string) => {
        setConnectionState(status);
        setError(errorMsg);

        // Update connection tracking
        if (status === ConnectionState.CONNECTED) {
          hasConnectedBefore = true;
        }

        // Only show error toast if we've connected before or if we're not in the initial connection phase
        if (
          status === ConnectionState.ERROR &&
          errorMsg &&
          errorMsg.length > 0 &&
          (hasConnectedBefore || websocketStore.retryCount > 1)
        ) {
          toast.warning(errorMsg, {
            id: "websocket-error",
            duration: 10000,
            action: {
              label: "Retry",
              onClick: () => websocketStore.reconnect(),
            },
          });
        }
      }
    );

    // Cleanup on unmount
    return () => {
      unsubscribe();
      websocketStore.disconnect();
    };
  }, [websocketStore]);

  // Context value
  const contextValue: WebSocketContextValue = {
    isConnected: connectionState === ConnectionState.CONNECTED,
    connectionState,
    reconnect: websocketStore.reconnect,
    isAttemptingConnection:
      connectionState === ConnectionState.CONNECTING ||
      connectionState === ConnectionState.RECONNECTING,
    error
  };

  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
}

export const useWebSocketContext = () => useContext(WebSocketContext);
