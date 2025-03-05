import { WebSocketMessage } from './types';
import { getConnectionManager } from './connectionManager';
import { getMessageStore } from './messageStore';

const messageStore = getMessageStore().getState();

// Initialize the WebSocket service
export function initializeWebSocketService() {
    const store = getConnectionManager();
    const state = store.getState();
    // Make sure we have a connection with a valid WebSocket
    if (!state.connection?.ws) {
        return;
    }

    // Set up the message handler for the ConnectionManager
    const ws = state.connection.ws;
    const originalOnMessage = ws.onmessage;

    ws.onmessage = (event: MessageEvent) => {
        // Call the original handler first
        if (originalOnMessage) {
            originalOnMessage.call(ws, event);
        }

        try {
            const data = JSON.parse(event.data) as WebSocketMessage;

            // Forward the message to the message store
            messageStore.addMessage(data);
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
        }
    };
}

// Function to start the WebSocket service
export function startWebSocketService() {
    const store = getConnectionManager();
    // Connect the WebSocket
    store.getState().connect();

    // Set up message handling
    initializeWebSocketService();
}

// Function to stop the WebSocket service
export function stopWebSocketService() {
    const store = getConnectionManager();
    // Disconnect the WebSocket
    store.getState().disconnect();
}

// Add event listeners for page visibility changes to manage connection
function setupVisibilityHandling() {
    const store = getConnectionManager();
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            // Reconnect if disconnected when the page becomes visible
            const state = store.getState();
            if (!state.isConnected()) {
                state.reconnect();
            }
        }
    });
}

// Setup visibility handling
setupVisibilityHandling(); 