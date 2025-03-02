import { create } from "zustand";
import { devtools } from "zustand/middleware";

export type WebSocketMessage = {
  trajectory_id: string;
  project_id: string;
  type: string;
  message?: string;
  iteration?: number;
  action?: string;
  cost?: number;
  timestamp?: string;
};

type WebSocketConnection = {
  ws: WebSocket;
  status: "connecting" | "connected" | "error" | "closed";
  error?: string;
};

const WS_RETRY_DELAY = 3000; // 3 seconds
const MAX_RETRIES = 3;
const PING_INTERVAL = 15000; // 15 seconds instead of 30
const PONG_TIMEOUT = 5000; // 5 seconds to wait for pong response
const INITIAL_CONNECTION_DELAY = 1000; // 1 second delay before first connection attempt

// Add a function to get the WebSocket URL
const getWebSocketUrl = () => {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  // Use port 8000 for the API server
  return `${protocol}//localhost:8000/api/ws`;
};

interface WebSocketStore {
  connection: WebSocketConnection | null;
  messages: Record<string, WebSocketMessage[]>;
  connect: () => void;
  disconnect: () => void;
  addMessage: (message: WebSocketMessage) => void;
  isConnected: () => boolean;
  retryCount: number;
  subscribers: Record<string, Set<(data: any) => void>>;
  subscribe: (channel: string, callback: (data: any) => void) => () => void;
  unsubscribe: (channel: string, callback: (data: any) => void) => void;
  serverSubscribeToProject: (projectId: string) => Promise<boolean>;
  serverSubscribeToTrajectory: (trajectoryId: string) => Promise<boolean>;
  serverUnsubscribeFromProject: (projectId: string) => Promise<boolean>;
  serverUnsubscribeFromTrajectory: (trajectoryId: string) => Promise<boolean>;
  activeProjectSubscriptions: Set<string>;
  activeTrajectorySubscriptions: Set<string>;
}

export const useWebSocketStore = create<WebSocketStore>()(
  devtools(
    (set, get) => ({
      connection: null,
      messages: {},
      retryCount: 0,
      subscribers: {},
      activeProjectSubscriptions: new Set<string>(),
      activeTrajectorySubscriptions: new Set<string>(),

      isConnected: () => {
        return get().connection?.status === "connected";
      },

      connect: () => {
        const currentConnection = get().connection;
        const retryCount = get().retryCount;

        // Don't try to reconnect if we're already connected
        if (currentConnection?.status === "connected") {
          return;
        }

        // Add initial delay for first connection attempt
        const delay =
          retryCount === 0 ? INITIAL_CONNECTION_DELAY : WS_RETRY_DELAY;

        // Check retry limit
        if (retryCount >= MAX_RETRIES) {
          set({
            connection: {
              ws: null as any,
              status: "error",
              error: "Max retry attempts reached",
            },
            retryCount: 0,
          });
          return;
        }

        setTimeout(() => {
          try {
            console.log(
              `Connecting to WebSocket... (attempt ${retryCount + 1}/${MAX_RETRIES + 1})`,
            );
            const wsUrl = getWebSocketUrl();
            console.log("WebSocket URL:", wsUrl);

            const ws = new WebSocket(wsUrl);
            let pingTimeout: NodeJS.Timeout;
            let pongTimeout: NodeJS.Timeout;

            set({
              connection: {
                ws,
                status: "connecting",
              },
            });

            const heartbeat = () => {
              // Clear any existing timeouts
              if (pingTimeout) clearTimeout(pingTimeout);
              if (pongTimeout) clearTimeout(pongTimeout);

              // Send ping
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "ping" }));

                // Set timeout for pong response
                pongTimeout = setTimeout(() => {
                  console.log("Pong not received in time, closing connection");
                  ws.close();
                }, PONG_TIMEOUT);
              }

              // Schedule next ping
              pingTimeout = setTimeout(heartbeat, PING_INTERVAL);
            };

            ws.onopen = () => {
              console.log("WebSocket connected");

              set({
                connection: {
                  ws,
                  status: "connected",
                  error: undefined,
                },
                retryCount: 0,
              });

              const resubscribe = async () => {
                const store = get();

                for (const projectId of store.activeProjectSubscriptions) {
                  await store.serverSubscribeToProject(projectId);
                }

                for (const trajectoryId of store.activeTrajectorySubscriptions) {
                  await store.serverSubscribeToTrajectory(trajectoryId);
                }
              };

              resubscribe();

              heartbeat();
            };

            ws.onmessage = (event) => {
              const message = JSON.parse(event.data);
              console.log("Received message:", message);

              // Handle pong response
              if (message.type === "pong") {
                if (pongTimeout) clearTimeout(pongTimeout);
                return;
              }

              get().addMessage({
                ...message,
                timestamp: message.timestamp || new Date().toISOString(),
              });
            };

            ws.onclose = () => {
              console.log("WebSocket closed");
              // Clear heartbeat timeouts
              if (pingTimeout) clearTimeout(pingTimeout);
              if (pongTimeout) clearTimeout(pongTimeout);

              set((state) => ({
                connection: {
                  ws,
                  status: "closed",
                },
                retryCount: state.retryCount + 1,
              }));

              // Attempt to reconnect after delay
              const newRetryCount = get().retryCount + 1;
              if (newRetryCount < MAX_RETRIES) {
                setTimeout(() => get().connect(), WS_RETRY_DELAY);
              }
            };

            ws.onerror = (event) => {
              console.log("WebSocket error");
              console.log(event);
              // Clear heartbeat timeouts
              if (pingTimeout) clearTimeout(pingTimeout);
              if (pongTimeout) clearTimeout(pongTimeout);

              const newRetryCount = get().retryCount + 1;
              set({ retryCount: newRetryCount });

              if (newRetryCount < MAX_RETRIES) {
                setTimeout(() => get().connect(), WS_RETRY_DELAY);
              }
            };
          } catch (error) {
            console.error("WebSocket connection error:", error);
            const newRetryCount = get().retryCount + 1;
            set({
              connection: {
                ws: null as any,
                status: "error",
                error:
                  error instanceof Error ? error.message : "Failed to connect",
              },
              retryCount: newRetryCount,
            });

            if (newRetryCount < MAX_RETRIES) {
              setTimeout(() => get().connect(), WS_RETRY_DELAY);
            }
          }
        }, delay);
      },

      disconnect: () => {
        const { connection } = get();
        if (connection?.ws) {
          connection.ws.close();
        }
        set({ connection: null });
      },

      // Method to subscribe to local updates (client-side only)
      subscribe: (channel, callback) => {
        set((state) => ({
          subscribers: {
            ...state.subscribers,
            [channel]: (state.subscribers[channel] || new Set()).add(callback),
          },
        }));

        // Return unsubscribe function
        return () => get().unsubscribe(channel, callback);
      },

      // Method to unsubscribe from local updates
      unsubscribe: (channel, callback) => {
        set((state) => {
          const channelSubs = state.subscribers[channel];
          if (channelSubs) {
            channelSubs.delete(callback);
          }
          return { subscribers: state.subscribers };
        });
      },

      serverSubscribeToProject: async (projectId: string) => {
        const { connection } = get();
        if (!connection || connection.status !== "connected") {
          console.error("Cannot subscribe: WebSocket not connected");
          return false;
        }

        try {
          connection.ws.send(
            JSON.stringify({
              type: "subscribe",
              subscription: "project",
              id: projectId,
            }),
          );

          set((state) => {
            const newActiveProjectSubscriptions = new Set(
              state.activeProjectSubscriptions,
            );
            newActiveProjectSubscriptions.add(projectId);
            return {
              activeProjectSubscriptions: newActiveProjectSubscriptions,
            };
          });

          console.log(`Sent subscription request for project: ${projectId}`);
          return true;
        } catch (error) {
          console.error("Failed to send project subscription:", error);
          return false;
        }
      },

      serverSubscribeToTrajectory: async (trajectoryId: string) => {
        const { connection } = get();
        if (!connection || connection.status !== "connected") {
          console.error("Cannot subscribe: WebSocket not connected");
          return false;
        }

        try {
          connection.ws.send(
            JSON.stringify({
              type: "subscribe",
              subscription: "trajectory",
              id: trajectoryId,
            }),
          );

          set((state) => {
            const newActiveTrajectorySubscriptions = new Set(
              state.activeTrajectorySubscriptions,
            );
            newActiveTrajectorySubscriptions.add(trajectoryId);
            return {
              activeTrajectorySubscriptions: newActiveTrajectorySubscriptions,
            };
          });

          console.log(
            `Sent subscription request for trajectory: ${trajectoryId}`,
          );
          return true;
        } catch (error) {
          console.error("Failed to send trajectory subscription:", error);
          return false;
        }
      },

      serverUnsubscribeFromProject: async (projectId: string) => {
        const { connection } = get();
        if (!connection || connection.status !== "connected") {
          console.error("Cannot unsubscribe: WebSocket not connected");
          return false;
        }

        try {
          connection.ws.send(
            JSON.stringify({
              type: "unsubscribe",
              subscription: "project",
              id: projectId,
            }),
          );

          set((state) => {
            const newActiveProjectSubscriptions = new Set(
              state.activeProjectSubscriptions,
            );
            newActiveProjectSubscriptions.delete(projectId);
            return {
              activeProjectSubscriptions: newActiveProjectSubscriptions,
            };
          });

          console.log(`Sent unsubscription request for project: ${projectId}`);
          return true;
        } catch (error) {
          console.error("Failed to send project unsubscription:", error);
          return false;
        }
      },

      serverUnsubscribeFromTrajectory: async (trajectoryId: string) => {
        const { connection } = get();
        if (!connection || connection.status !== "connected") {
          console.error("Cannot unsubscribe: WebSocket not connected");
          return false;
        }

        try {
          connection.ws.send(
            JSON.stringify({
              type: "unsubscribe",
              subscription: "trajectory",
              id: trajectoryId,
            }),
          );

          set((state) => {
            const newActiveTrajectorySubscriptions = new Set(
              state.activeTrajectorySubscriptions,
            );
            newActiveTrajectorySubscriptions.delete(trajectoryId);
            return {
              activeTrajectorySubscriptions: newActiveTrajectorySubscriptions,
            };
          });

          console.log(
            `Sent unsubscription request for trajectory: ${trajectoryId}`,
          );
          return true;
        } catch (error) {
          console.error("Failed to send trajectory unsubscription:", error);
          return false;
        }
      },

      addMessage: (message: WebSocketMessage) => {
        const { trajectory_id, project_id } = message;

        // Notify trajectory subscribers
        if (trajectory_id) {
          const trajectoryChannel = `trajectory.${trajectory_id}`;
          const trajectorySubscribers = get().subscribers[trajectoryChannel];
          if (trajectorySubscribers) {
            trajectorySubscribers.forEach((callback) => callback(message));
          }
        }

        // Notify project subscribers
        if (project_id) {
          const projectChannel = `project.${project_id}`;
          const projectSubscribers = get().subscribers[projectChannel];
          if (projectSubscribers) {
            projectSubscribers.forEach((callback) => callback(message));
          }
        }

        // Store message in state
        if (trajectory_id) {
          set((state) => ({
            messages: {
              ...state.messages,
              [trajectory_id]: [
                ...(state.messages[trajectory_id] || []),
                message,
              ],
            },
          }));
        }
      },
    }),
    { name: "websocket-store" },
  ),
);

// Selector functions
export const selectMessages = (runId: string) => (state: WebSocketStore) =>
  state.messages[runId] || [];

export const selectConnectionStatus = () => (state: WebSocketStore) =>
  state.connection?.status || "closed";

export const selectConnectionError = () => (state: WebSocketStore) =>
  state.connection?.error;

// Create a hook for easier subscription management
export function useWebSocket() {
  const store = useWebSocketStore();

  return {
    subscribe: store.subscribe,
    unsubscribe: store.unsubscribe,
    isConnected: store.isConnected,
  };
}
