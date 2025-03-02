import { useWebSocketStore } from "@/lib/stores/websocketStore";
import { useEffect, useRef } from "react";

/**
 * Hook for WebSocket connection management and server-side subscriptions.
 * Automatically connects to the WebSocket server on first mount.
 */
export function useWebSocket() {
  // Use selectors to access store state
  const connection = useWebSocketStore((state) => state.connection);
  const connect = useWebSocketStore((state) => state.connect);
  const disconnect = useWebSocketStore((state) => state.disconnect);

  // Get server subscription methods
  const serverSubscribeToProject = useWebSocketStore(
    (state) => state.serverSubscribeToProject,
  );
  const serverSubscribeToTrajectory = useWebSocketStore(
    (state) => state.serverSubscribeToTrajectory,
  );
  const serverUnsubscribeFromProject = useWebSocketStore(
    (state) => state.serverUnsubscribeFromProject,
  );
  const serverUnsubscribeFromTrajectory = useWebSocketStore(
    (state) => state.serverUnsubscribeFromTrajectory,
  );

  // Client-side subscription methods
  const subscribe = useWebSocketStore((state) => state.subscribe);
  const unsubscribe = useWebSocketStore((state) => state.unsubscribe);
  const isConnected = useWebSocketStore((state) => state.isConnected);

  // Use a ref to track if we've attempted to connect
  const hasAttemptedConnection = useRef(false);

  useEffect(() => {
    // Only try to connect once and only if we haven't connected yet
    if (!hasAttemptedConnection.current && !connection) {
      hasAttemptedConnection.current = true;
      connect();
    }

    // Cleanup
    return () => {
      // We don't disconnect on unmount anymore
      // to keep subscriptions active across page navigations
      // disconnect();
    };
  }, []); // Empty dependency array - we only want this to run once

  return {
    status: connection?.status || "closed",
    error: connection?.error,
    isConnected: isConnected(),

    // Client-side subscription methods (for local event handling)
    subscribe,
    unsubscribe,

    // Server-side subscription methods (for backend communication)
    subscribeToProject: serverSubscribeToProject,
    subscribeToTrajectory: serverSubscribeToTrajectory,
    unsubscribeFromProject: serverUnsubscribeFromProject,
    unsubscribeFromTrajectory: serverUnsubscribeFromTrajectory,
  };
}
