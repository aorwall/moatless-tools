import { create } from "zustand";
import { devtools } from "zustand/middleware";

// Constants
const WS_RETRY_DELAY = 3000; // 3 seconds
const MAX_RETRIES = 3;
const PING_INTERVAL = 15000; // 15 seconds instead of 30
const PONG_TIMEOUT = 5000; // 5 seconds to wait for pong response
const INITIAL_CONNECTION_DELAY = 1000; // 1 second delay before first connection attempt

// Message types
export enum WebSocketMessageType {
  PING = "ping",
  PONG = "pong",
  SUBSCRIBE = "subscribe",
  UNSUBSCRIBE = "unsubscribe",
}

// Subscription types
export enum SubscriptionType {
  PROJECT = "project",
  TRAJECTORY = "trajectory",
}

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

// Define a type for trajectory subscriptions
export type TrajectorySubscription = {
  projectId: string;
  trajectoryId: string;
};

// Helper to create a unique key for a trajectory subscription
export const getTrajectorySubscriptionKey = (sub: TrajectorySubscription): string =>
  `${sub.projectId}:${sub.trajectoryId}`;

// Helper to find a trajectory subscription in a set
const findTrajectorySubscription = (
  set: Set<TrajectorySubscription>,
  projectId: string,
  trajectoryId: string
): TrajectorySubscription | undefined => {
  for (const sub of set) {
    if (sub.projectId === projectId && sub.trajectoryId === trajectoryId) {
      return sub;
    }
  }
  return undefined;
};

// Channel helpers
const getProjectChannel = (projectId: string) => `project.${projectId}`;
const getTrajectoryChannel = (projectId: string, trajectoryId: string): string =>
  `project.${projectId}.trajectory.${trajectoryId}`;

// Add a function to get the WebSocket URL
const getWebSocketUrl = () => {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  // Use port 8000 for the API server
  return `${protocol}//localhost:8000/api/ws`;
};

// Type for subscription message
type SubscriptionMessage = {
  type: WebSocketMessageType.SUBSCRIBE | WebSocketMessageType.UNSUBSCRIBE;
  subscription: SubscriptionType;
  project_id: string;
  trajectory_id?: string;
};

interface WebSocketStore {
  connection: WebSocketConnection | null;
  connect: () => void;
  disconnect: () => void;
  addMessage: (message: WebSocketMessage) => void;
  isConnected: () => boolean;
  retryCount: number;
  subscribers: Record<string, Set<(data: WebSocketMessage) => void>>;
  subscribe: (channel: string, callback: (data: WebSocketMessage) => void) => () => void;
  unsubscribe: (channel: string, callback: (data: WebSocketMessage) => void) => void;
  sendSubscriptionMessage: (message: SubscriptionMessage) => Promise<boolean>;
  serverSubscribeToProject: (projectId: string) => Promise<boolean>;
  serverSubscribeToTrajectory: (projectId: string, trajectoryId: string) => Promise<boolean>;
  serverUnsubscribeFromProject: (projectId: string) => Promise<boolean>;
  serverUnsubscribeFromTrajectory: (projectId: string, trajectoryId: string) => Promise<boolean>;
  activeProjectSubscriptions: Set<string>;
  activeTrajectorySubscriptions: Set<TrajectorySubscription>;
  messages: Record<string, WebSocketMessage[]>;
  subscribeToProject: (projectId: string, callback: (data: WebSocketMessage) => void) => () => void;
  subscribeToTrajectory: (projectId: string, trajectoryId: string, callback: (data: WebSocketMessage) => void) => () => void;
}

export const useWebSocketStore = create<WebSocketStore>()(
  devtools(
    (set, get) => ({
      connection: null,
      messages: {},
      retryCount: 0,
      subscribers: {},
      activeProjectSubscriptions: new Set<string>(),
      activeTrajectorySubscriptions: new Set<TrajectorySubscription>(),

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
                ws.send(JSON.stringify({ type: WebSocketMessageType.PING }));

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

                // Resubscribe to all active project subscriptions
                for (const projectId of store.activeProjectSubscriptions) {
                  await store.serverSubscribeToProject(projectId);
                }

                // Resubscribe to all active trajectory subscriptions
                for (const channelId of store.activeTrajectorySubscriptions) {
                  const { projectId, trajectoryId } = channelId;
                  await store.serverSubscribeToTrajectory(projectId, trajectoryId);
                }
              };

              resubscribe();
              heartbeat();
            };

            ws.onmessage = (event) => {
              const message = JSON.parse(event.data);
              console.log("Received message:", message);

              // Handle pong response
              if (message.type === WebSocketMessageType.PONG) {
                if (pongTimeout) clearTimeout(pongTimeout);
                return;
              }

              get().addMessage({
                ...message,
                timestamp: message.timestamp || new Date().toISOString(),
              });
            };

            const handleDisconnect = () => {
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
              const newRetryCount = get().retryCount;
              if (newRetryCount < MAX_RETRIES) {
                setTimeout(() => get().connect(), WS_RETRY_DELAY);
              }
            };

            ws.onclose = () => {
              console.log("WebSocket closed");
              handleDisconnect();
            };

            ws.onerror = (event) => {
              console.log("WebSocket error", event);
              handleDisconnect();
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
          return { subscribers: { ...state.subscribers } };
        });
      },

      // Generic method to send subscription messages
      sendSubscriptionMessage: async (message: SubscriptionMessage) => {
        const { connection } = get();
        if (!connection || connection.status !== "connected") {
          console.error("Cannot send subscription message: WebSocket not connected");
          return false;
        }

        try {
          connection.ws.send(JSON.stringify(message));
          return true;
        } catch (error) {
          console.error(`Failed to send ${message.type} message:`, error);
          return false;
        }
      },

      // Type-safe method to subscribe to a project
      subscribeToProject: (projectId, callback) => {
        const channel = getProjectChannel(projectId);
        const clientUnsubscribe = get().subscribe(channel, callback);

        // Also subscribe on the server side
        get().serverSubscribeToProject(projectId).catch(error => {
          console.error(`Error subscribing to project ${projectId} on server:`, error);
        });

        // Return a cleanup function that handles both client and server unsubscription
        return () => {
          // Unsubscribe client-side
          clientUnsubscribe();

          // Unsubscribe server-side if connected
          if (get().isConnected()) {
            get().serverUnsubscribeFromProject(projectId).catch(error => {
              console.error(`Error unsubscribing from project ${projectId} on server:`, error);
            });
          }
        };
      },

      // Type-safe method to subscribe to a trajectory
      subscribeToTrajectory: (projectId, trajectoryId, callback) => {
        const channel = getTrajectoryChannel(projectId, trajectoryId);
        const clientUnsubscribe = get().subscribe(channel, callback);

        // Also subscribe on the server side
        get().serverSubscribeToTrajectory(projectId, trajectoryId).catch(error => {
          console.error(`Error subscribing to trajectory ${trajectoryId} on server:`, error);
        });

        // Return a cleanup function that handles both client and server unsubscription
        return () => {
          // Unsubscribe client-side
          clientUnsubscribe();

          // Unsubscribe server-side if connected
          if (get().isConnected()) {
            get().serverUnsubscribeFromTrajectory(projectId, trajectoryId).catch(error => {
              console.error(`Error unsubscribing from trajectory ${trajectoryId} on server:`, error);
            });
          }
        };
      },

      serverSubscribeToProject: async (projectId: string) => {
        const success = await get().sendSubscriptionMessage({
          type: WebSocketMessageType.SUBSCRIBE,
          subscription: SubscriptionType.PROJECT,
          project_id: projectId,
        });

        if (success) {
          set((state) => {
            const newActiveProjectSubscriptions = new Set(state.activeProjectSubscriptions);
            newActiveProjectSubscriptions.add(projectId);
            return { activeProjectSubscriptions: newActiveProjectSubscriptions };
          });
          console.log(`Sent subscription request for project: ${projectId}`);
        }

        return success;
      },

      serverSubscribeToTrajectory: async (projectId: string, trajectoryId: string) => {
        const success = await get().sendSubscriptionMessage({
          type: WebSocketMessageType.SUBSCRIBE,
          subscription: SubscriptionType.TRAJECTORY,
          project_id: projectId,
          trajectory_id: trajectoryId,
        });

        if (success) {
          set((state) => {
            // Only add if not already present
            if (!findTrajectorySubscription(state.activeTrajectorySubscriptions, projectId, trajectoryId)) {
              const newActiveTrajectorySubscriptions = new Set(state.activeTrajectorySubscriptions);
              newActiveTrajectorySubscriptions.add({ projectId, trajectoryId });
              return { activeTrajectorySubscriptions: newActiveTrajectorySubscriptions };
            }
            return {};
          });
          console.log(`Sent subscription request for trajectory: ${trajectoryId} (project: ${projectId})`);
        }

        return success;
      },

      serverUnsubscribeFromProject: async (projectId: string) => {
        const success = await get().sendSubscriptionMessage({
          type: WebSocketMessageType.UNSUBSCRIBE,
          subscription: SubscriptionType.PROJECT,
          project_id: projectId,
        });

        if (success) {
          set((state) => {
            const newActiveProjectSubscriptions = new Set(state.activeProjectSubscriptions);
            newActiveProjectSubscriptions.delete(projectId);
            return { activeProjectSubscriptions: newActiveProjectSubscriptions };
          });
          console.log(`Sent unsubscription request for project: ${projectId}`);
        }

        return success;
      },

      serverUnsubscribeFromTrajectory: async (projectId: string, trajectoryId: string) => {
        const success = await get().sendSubscriptionMessage({
          type: WebSocketMessageType.UNSUBSCRIBE,
          subscription: SubscriptionType.TRAJECTORY,
          project_id: projectId,
          trajectory_id: trajectoryId,
        });

        if (success) {
          set((state) => {
            const existingSub = findTrajectorySubscription(
              state.activeTrajectorySubscriptions,
              projectId,
              trajectoryId
            );

            if (existingSub) {
              const newActiveTrajectorySubscriptions = new Set(state.activeTrajectorySubscriptions);
              newActiveTrajectorySubscriptions.delete(existingSub);
              return { activeTrajectorySubscriptions: newActiveTrajectorySubscriptions };
            }
            return {};
          });
          console.log(`Sent unsubscription request for trajectory: ${trajectoryId} (project: ${projectId})`);
        }

        return success;
      },

      addMessage: (message: WebSocketMessage) => {
        const { trajectory_id, project_id } = message;

        // Notify subscribers based on message type
        const notifySubscribers = (channel: string) => {
          const subscribers = get().subscribers[channel];
          if (subscribers) {
            subscribers.forEach((callback) => callback(message));
          }
        };

        // Notify trajectory subscribers
        if (trajectory_id) {
          notifySubscribers(getTrajectoryChannel(project_id, trajectory_id));
        }

        // Notify project subscribers
        if (project_id) {
          notifySubscribers(getProjectChannel(project_id));
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

export const selectConnectionStatus = () => (state: WebSocketStore) =>
  state.connection?.status || "closed";

export const selectConnectionError = () => (state: WebSocketStore) =>
  state.connection?.error;

// Create a hook for easier subscription management
export function useWebSocket() {
  const store = useWebSocketStore();

  return {
    // Connection status
    isConnected: store.isConnected,

    // Type-safe subscription methods (these handle both client and server subscriptions)
    subscribeToProject: store.subscribeToProject,
    subscribeToTrajectory: store.subscribeToTrajectory
  };
}
