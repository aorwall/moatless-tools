import { createContext, useContext, useEffect, useRef } from "react";
import { useWebSocketStore } from "@/lib/stores/websocketStore";

const WebSocketContext = createContext<{ isConnected: boolean }>({
  isConnected: false,
});

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const connect = useWebSocketStore((state) => state.connect);
  const disconnect = useWebSocketStore((state) => state.disconnect);
  const isConnected = useWebSocketStore((state) => state.isConnected());
  const hasAttemptedConnection = useRef(false);

  useEffect(() => {
    if (!hasAttemptedConnection.current) {
      hasAttemptedConnection.current = true;
      connect();
    }

    return () => {
      disconnect();
    };
  }, []);

  return (
    <WebSocketContext.Provider value={{ isConnected }}>
      {children}
    </WebSocketContext.Provider>
  );
}

export const useWebSocketContext = () => useContext(WebSocketContext);
