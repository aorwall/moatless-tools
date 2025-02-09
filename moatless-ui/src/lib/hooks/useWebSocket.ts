import { useEffect, useRef } from "react";
import { useWebSocketStore } from "@/lib/stores/websocketStore";

export function useWebSocket() {
  // Use selectors to access store state
  const connection = useWebSocketStore((state) => state.connection);
  const connect = useWebSocketStore((state) => state.connect);
  const disconnect = useWebSocketStore((state) => state.disconnect);

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
      disconnect();
    };
  }, []); // Empty dependency array - we only want this to run once

  return {
    status: connection?.status || "closed",
    error: connection?.error,
  };
}
