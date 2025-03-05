import { create } from 'zustand';
import { debounce } from 'lodash';
import {
    WebSocketMessage,
    BatchedNotification,
    MessageHandler,
    WS_CONFIG
} from './types';

export interface MessageStore {
    subscribers: Record<string, Set<MessageHandler>>;
    batchedMessages: Record<string, WebSocketMessage[]>;
    processingBatch: boolean;
    messages: Record<string, WebSocketMessage[]>;
    addMessage: (message: WebSocketMessage) => void;
    getMessages: (channel: string) => WebSocketMessage[];
    clearMessages: () => void;
    subscribe: (channel: string, callback: MessageHandler) => () => void;
    unsubscribe: (channel: string, callback: MessageHandler) => void;
}

interface MessageStoreState extends MessageStore {
    setBatchedMessages: (messages: Record<string, WebSocketMessage[]>) => void;
    setProcessingBatch: (processing: boolean) => void;
    setMessages: (messages: Record<string, WebSocketMessage[]>) => void;
    notifySubscribers: (channel: string) => void;
    updateMessageHistory: (channel: string, newMessages: WebSocketMessage[]) => void;
}

export const createMessageStore = () => {
    const store = create<MessageStoreState>((set, get) => {
        // Create a debounced function to process batched messages
        const processBatchedMessages = debounce(() => {
            const { batchedMessages } = get();

            if (Object.keys(batchedMessages).length === 0) {
                get().setProcessingBatch(false);
                return;
            }

            // Process each channel's batched messages
            Object.keys(batchedMessages).forEach((channel) => {
                const channelMessages = batchedMessages[channel];
                if (!channelMessages || channelMessages.length === 0) return;

                // Notify subscribers
                get().notifySubscribers(channel);

                // Add messages to the history
                get().updateMessageHistory(channel, channelMessages);
            });

            // Clear the batch
            get().setBatchedMessages({});
            get().setProcessingBatch(false);
        }, WS_CONFIG.EVENT_BATCH_INTERVAL);

        return {
            subscribers: {},
            batchedMessages: {},
            processingBatch: false,
            messages: {},

            setBatchedMessages: (messages) => set({ batchedMessages: messages }),
            setProcessingBatch: (processing) => set({ processingBatch: processing }),
            setMessages: (messages) => set({ messages }),

            addMessage: (message: WebSocketMessage) => {
                // Determine which channels this message belongs to
                const channels: string[] = [];

                // Project-level events
                if (message.project_id) {
                    channels.push(`project.${message.project_id}`);

                    // Trajectory-level events
                    if (message.trajectory_id) {
                        channels.push(`project.${message.project_id}.trajectory.${message.trajectory_id}`);
                    }
                }

                // No valid channels for this message
                if (channels.length === 0) return;

                // Add the message to each channel's batch
                channels.forEach((channel) => {
                    const { batchedMessages } = get();
                    const currentBatch = batchedMessages[channel] || [];

                    // Add the message to the batch, respecting max queue size
                    const newBatch = [...currentBatch, message].slice(-WS_CONFIG.MAX_MESSAGE_QUEUE_SIZE);

                    get().setBatchedMessages({
                        ...batchedMessages,
                        [channel]: newBatch
                    });
                });

                // Trigger batch processing if not already in progress
                if (!get().processingBatch) {
                    get().setProcessingBatch(true);
                    processBatchedMessages();
                }
            },

            notifySubscribers: (channel: string) => {
                const { subscribers, batchedMessages } = get();
                const channelSubscribers = subscribers[channel];
                const channelMessages = batchedMessages[channel];

                if (!channelSubscribers || !channelMessages || channelMessages.length === 0) return;

                // Notify each subscriber with each message
                channelMessages.forEach((message) => {
                    channelSubscribers.forEach((callback) => {
                        try {
                            callback(message);
                        } catch (error) {
                            console.error(`Error notifying subscriber for channel ${channel}:`, error);
                        }
                    });
                });
            },

            updateMessageHistory: (channel: string, newMessages: WebSocketMessage[]) => {
                const { messages } = get();
                const existingMessages = messages[channel] || [];

                // Add new messages to history, respecting max size
                const updatedMessages = [...existingMessages, ...newMessages].slice(-WS_CONFIG.MAX_MESSAGE_QUEUE_SIZE);

                get().setMessages({
                    ...messages,
                    [channel]: updatedMessages
                });
            },

            getMessages: (channel: string) => {
                return get().messages[channel] || [];
            },

            clearMessages: () => {
                get().setMessages({});
            },

            subscribe: (channel: string, callback: MessageHandler) => {
                const { subscribers } = get();

                // Create set if it doesn't exist
                if (!subscribers[channel]) {
                    subscribers[channel] = new Set();
                }

                // Add the subscriber
                subscribers[channel].add(callback);

                // Return unsubscribe function
                return () => {
                    get().unsubscribe(channel, callback);
                };
            },

            unsubscribe: (channel: string, callback: MessageHandler) => {
                const { subscribers } = get();
                const channelSubscribers = subscribers[channel];

                if (channelSubscribers) {
                    channelSubscribers.delete(callback);

                    // Clean up empty subscriber sets
                    if (channelSubscribers.size === 0) {
                        delete subscribers[channel];
                    }
                }
            },
        };
    });

    return () => store;
};

// Export the store creator function instead of an instance
export const messageStoreCreator = createMessageStore();

// Export a function to get or create the store instance
let messageStoreInstance: ReturnType<typeof messageStoreCreator> | null = null;

export const getMessageStore = () => {
    if (!messageStoreInstance) {
        messageStoreInstance = messageStoreCreator();
    }
    return messageStoreInstance;
}; 