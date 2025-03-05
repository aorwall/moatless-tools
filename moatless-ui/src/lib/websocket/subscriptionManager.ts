import { create } from 'zustand';
import {
    WebSocketMessage,
    SubscriptionType,
    WebSocketMessageType,
    MessageHandler,
    TrajectorySubscription,
    SubscriptionMessage,
    getProjectChannel,
    getTrajectoryChannel,
    getTrajectorySubscriptionKey,
    findTrajectorySubscription
} from './types';
import { getConnectionManager } from './connectionManager';
import { getMessageStore } from './messageStore';

// Get store instances
const connectionManager = getConnectionManager().getState();
const messageStore = getMessageStore().getState();

export interface SubscriptionManager {
    activeProjectSubscriptions: Set<string>;
    activeTrajectorySubscriptions: Set<TrajectorySubscription>;
    subscribeToProject: (projectId: string, callback: MessageHandler) => () => void;
    subscribeToTrajectory: (projectId: string, trajectoryId: string, callback: MessageHandler) => () => void;
    serverSubscribeToProject: (projectId: string) => Promise<boolean>;
    serverSubscribeToTrajectory: (projectId: string, trajectoryId: string) => Promise<boolean>;
    serverUnsubscribeFromProject: (projectId: string) => Promise<boolean>;
    serverUnsubscribeFromTrajectory: (projectId: string, trajectoryId: string) => Promise<boolean>;
    sendSubscriptionMessage: (message: SubscriptionMessage) => Promise<boolean>;
}

interface SubscriptionManagerStore extends SubscriptionManager {
    setActiveProjectSubscriptions: (subscriptions: Set<string>) => void;
    setActiveTrajectorySubscriptions: (subscriptions: Set<TrajectorySubscription>) => void;
}

// Create the store without immediately initializing it
export const createSubscriptionManager = () => {
    const store = create<SubscriptionManagerStore>((set, get) => ({
        activeProjectSubscriptions: new Set<string>(),
        activeTrajectorySubscriptions: new Set<TrajectorySubscription>(),

        setActiveProjectSubscriptions: (subscriptions) => set({ activeProjectSubscriptions: subscriptions }),
        setActiveTrajectorySubscriptions: (subscriptions) => set({ activeTrajectorySubscriptions: subscriptions }),

        // Client-side subscription that also handles server-side subscription
        subscribeToProject: (projectId: string, callback: MessageHandler) => {
            // Get the project channel
            const channel = getProjectChannel(projectId);

            // Subscribe to the channel locally
            const unsubscribe = messageStore.subscribe(channel, callback);

            // Subscribe to the project on the server
            get().serverSubscribeToProject(projectId).catch((error) => {
                console.error("Error subscribing to project:", error);
            });

            // Return function to unsubscribe
            return () => {
                unsubscribe();

                // Check if we should unsubscribe from the server
                const { subscribers } = messageStore;
                const projectSubscribers = subscribers[channel];

                if (!projectSubscribers || projectSubscribers.size === 0) {
                    get().serverUnsubscribeFromProject(projectId).catch((error) => {
                        console.error("Error unsubscribing from project:", error);
                    });
                }
            };
        },

        // Client-side subscription that also handles server-side subscription
        subscribeToTrajectory: (projectId: string, trajectoryId: string, callback: MessageHandler) => {
            // Get the trajectory channel
            const channel = getTrajectoryChannel(projectId, trajectoryId);

            // Subscribe to the channel locally
            const unsubscribe = messageStore.subscribe(channel, callback);

            // Subscribe to the trajectory on the server
            get().serverSubscribeToTrajectory(projectId, trajectoryId).catch((error) => {
                console.error("Error subscribing to trajectory:", error);
            });

            // Return function to unsubscribe
            return () => {
                unsubscribe();

                // Check if we should unsubscribe from the server
                const { subscribers } = messageStore;
                const trajectorySubscribers = subscribers[channel];

                if (!trajectorySubscribers || trajectorySubscribers.size === 0) {
                    get().serverUnsubscribeFromTrajectory(projectId, trajectoryId).catch((error) => {
                        console.error("Error unsubscribing from trajectory:", error);
                    });
                }
            };
        },

        // Send a subscription message to the server
        sendSubscriptionMessage: async (message: SubscriptionMessage): Promise<boolean> => {
            return connectionManager.sendMessage(message);
        },

        // Server-side project subscription
        serverSubscribeToProject: async (projectId: string): Promise<boolean> => {
            const { activeProjectSubscriptions } = get();

            // Already subscribed
            if (activeProjectSubscriptions.has(projectId)) {
                return true;
            }

            // Send subscription message
            const subscriptionMessage: SubscriptionMessage = {
                type: WebSocketMessageType.SUBSCRIBE,
                subscription: SubscriptionType.PROJECT,
                project_id: projectId,
            };

            const success = await get().sendSubscriptionMessage(subscriptionMessage);

            if (success) {
                // Add to active subscriptions
                const newSubscriptions = new Set(activeProjectSubscriptions);
                newSubscriptions.add(projectId);
                get().setActiveProjectSubscriptions(newSubscriptions);
            }

            return success;
        },

        // Server-side trajectory subscription
        serverSubscribeToTrajectory: async (projectId: string, trajectoryId: string): Promise<boolean> => {
            const { activeTrajectorySubscriptions } = get();

            // Already subscribed
            const existingSubscription = findTrajectorySubscription(
                activeTrajectorySubscriptions,
                projectId,
                trajectoryId
            );

            if (existingSubscription) {
                return true;
            }

            // Send subscription message
            const subscriptionMessage: SubscriptionMessage = {
                type: WebSocketMessageType.SUBSCRIBE,
                subscription: SubscriptionType.TRAJECTORY,
                project_id: projectId,
                trajectory_id: trajectoryId,
            };

            const success = await get().sendSubscriptionMessage(subscriptionMessage);

            if (success) {
                // Add to active subscriptions
                const newSubscription: TrajectorySubscription = {
                    projectId,
                    trajectoryId,
                };

                const newSubscriptions = new Set(activeTrajectorySubscriptions);
                newSubscriptions.add(newSubscription);
                get().setActiveTrajectorySubscriptions(newSubscriptions);
            }

            return success;
        },

        // Server-side project unsubscription
        serverUnsubscribeFromProject: async (projectId: string): Promise<boolean> => {
            const { activeProjectSubscriptions } = get();

            // Not subscribed
            if (!activeProjectSubscriptions.has(projectId)) {
                return true;
            }

            // Send unsubscription message
            const unsubscriptionMessage: SubscriptionMessage = {
                type: WebSocketMessageType.UNSUBSCRIBE,
                subscription: SubscriptionType.PROJECT,
                project_id: projectId,
            };

            const success = await get().sendSubscriptionMessage(unsubscriptionMessage);

            if (success) {
                // Remove from active subscriptions
                const newSubscriptions = new Set(activeProjectSubscriptions);
                newSubscriptions.delete(projectId);
                get().setActiveProjectSubscriptions(newSubscriptions);
            }

            return success;
        },

        // Server-side trajectory unsubscription
        serverUnsubscribeFromTrajectory: async (projectId: string, trajectoryId: string): Promise<boolean> => {
            const { activeTrajectorySubscriptions } = get();

            // Not subscribed
            const existingSubscription = findTrajectorySubscription(
                activeTrajectorySubscriptions,
                projectId,
                trajectoryId
            );

            if (!existingSubscription) {
                return true;
            }

            // Send unsubscription message
            const unsubscriptionMessage: SubscriptionMessage = {
                type: WebSocketMessageType.UNSUBSCRIBE,
                subscription: SubscriptionType.TRAJECTORY,
                project_id: projectId,
                trajectory_id: trajectoryId,
            };

            const success = await get().sendSubscriptionMessage(unsubscriptionMessage);

            if (success) {
                // Remove from active subscriptions
                const newSubscriptions = new Set(activeTrajectorySubscriptions);
                newSubscriptions.delete(existingSubscription);
                get().setActiveTrajectorySubscriptions(newSubscriptions);
            }

            return success;
        },
    }));

    return () => store;
};

// Export the store creator function instead of an instance
export const subscriptionManagerCreator = createSubscriptionManager();

// Export a function to get or create the store instance
let subscriptionManagerInstance: ReturnType<typeof subscriptionManagerCreator> | null = null;

export const getSubscriptionManager = () => {
    if (!subscriptionManagerInstance) {
        subscriptionManagerInstance = subscriptionManagerCreator();
    }
    return subscriptionManagerInstance;
}; 