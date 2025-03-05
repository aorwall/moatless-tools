# WebSocket Implementation

This directory contains the WebSocket implementation for real-time data handling in the application.

## Architecture

### Core Modules

#### Connection Manager (`websocket/connectionManager.ts`)
- Manages WebSocket connection lifecycle
- Implements reconnection with exponential backoff
- Handles connection state and error management
- Accessed via `getConnectionManager()` to ensure singleton instance
- State management with Zustand

#### Message Store (`websocket/messageStore.ts`)
- Handles message queuing and batching
- Manages message subscriptions
- Provides message history
- Direct singleton instance

#### Subscription Manager (`websocket/subscriptionManager.ts`)
- Manages WebSocket subscriptions
- Handles project and trajectory subscriptions
- Uses connection manager for sending messages
- Direct singleton instance

### Integration Layer

#### WebSocket Store (`stores/websocketStore.ts`)
- Facade for the underlying WebSocket modules
- Combines functionality from all core modules
- Provides React hooks for component integration
- Persists connection state
- Uses Zustand for state management

#### WebSocket Provider (`providers/WebSocketProvider.tsx`)
- React context provider for WebSocket functionality
- Initializes WebSocket connection
- Manages connection lifecycle in React components
- Provides connection status to component tree

### Dependency Flow
```
WebSocketProvider
    └── WebSocketStore
        ├── ConnectionManager
        ├── MessageStore
        └── SubscriptionManager
```

## Usage

```typescript
import { useRealtimeQuery } from '@/lib/hooks';

function MyComponent() {
  const { data, isLoading, lastMessage, connectionState } = useRealtimeQuery({
    queryKey: ['my-data'],
    queryFn: fetchData,
    eventFilter: {
      type: 'trajectory',
      action: 'update'
    }
  });

  // Use the data and connection state in your component
}
```

## Creating Real-Time Queries

To create a new real-time query hook, follow these steps:

### 1. Define Query Keys

First, define your query keys using a consistent pattern:

```typescript
export const myFeatureKeys = {
    all: ["feature-name"] as const,
    lists: () => [...myFeatureKeys.all, "list"] as const,
    list: (filters: Record<string, unknown>) => [...myFeatureKeys.lists(), filters] as const,
    details: () => [...myFeatureKeys.all, "detail"] as const,
    detail: (id: string) => [...myFeatureKeys.details(), id] as const,
};
```

### 2. Create the Real-Time Hook

Create a new hook using `useRealtimeQuery`:

```typescript
import { useRealtimeQuery } from "@/lib/hooks/useRealtimeQuery";
import { MyDataType } from "@/lib/types";
import { UseQueryOptions } from "@tanstack/react-query";

export function useRealtimeMyFeature(
    id: string,
    options?: Omit<UseQueryOptions<MyDataType, Error, MyDataType>, "queryKey" | "queryFn">
) {
    return useRealtimeQuery({
        // Query configuration
        queryKey: myFeatureKeys.detail(id),
        queryFn: () => myFeatureApi.getFeature(id),
        
        // WebSocket subscription configuration
        subscriptionConfig: {
            // Required for project-scoped subscriptions
            projectId?: string,
            
            // Required for trajectory-scoped subscriptions
            trajectoryId?: string,
            
            // Filter events by type, scope, or custom criteria
            eventFilter: {
                // Filter by event scopes (e.g., "flow", "trajectory")
                scopes?: string[],
                
                // Match any of the scopes instead of all
                anyMatch?: boolean,
                
                // Custom filter function
                filter?: (message: WebSocketMessage) => boolean
            }
        },
        
        // Additional React Query options
        options
    });
}
```

### 3. Usage Example

```typescript
function MyFeatureComponent({ id }: { id: string }) {
    const { 
        data,           // The fetched data
        isLoading,      // Loading state
        error,          // Error state
        lastMessage,    // Last WebSocket message received
        connectionState // Current WebSocket connection state
    } = useRealtimeMyFeature(id);

    // Handle loading and error states
    if (isLoading) return <Loading />;
    if (error) return <Error error={error} />;

    // Render your component with real-time data
    return <FeatureDisplay data={data} />;
}
```

### Best Practices

1. **Query Keys**: 
   - Use consistent key structure across features
   - Make keys type-safe using `as const`
   - Export query keys for cache invalidation

2. **Subscription Config**:
   - Only subscribe to relevant events
   - Use scoped subscriptions when possible
   - Implement custom filters for specific needs

3. **Error Handling**:
   - Handle both query errors and WebSocket errors
   - Provide fallback UI for disconnected states
   - Use connection state for user feedback

4. **Performance**:
   - Unsubscribe from events when component unmounts
   - Use `options.enabled` to control when query runs
   - Implement proper cleanup in useEffect hooks

5. **Type Safety**:
   - Define proper types for your data
   - Use TypeScript generics for type inference
   - Export types for component usage

## Configuration

### Environment Variables
```env
# Required
VITE_API_HOST=localhost:8000      # API host for fallback
VITE_API_URL=http://localhost:8000 # Base API URL
VITE_WS_URL=ws://localhost:8000   # WebSocket base URL
VITE_WS_PATH=/api/ws             # WebSocket endpoint path

# Optional
VITE_ENABLE_WEBSOCKET=true       # Enable WebSocket functionality
VITE_ENABLE_POLLING_FALLBACK=true # Enable polling fallback
```

### WebSocket Parameters
```typescript
WS_CONFIG = {
    RETRY_DELAY: 1000,            // Initial retry delay (ms)
    MAX_RETRIES: 5,               // Maximum retry attempts
    RECONNECT_BACKOFF_FACTOR: 1.5, // Exponential backoff multiplier
    PING_INTERVAL: 15000,         // Ping interval (ms)
    PONG_TIMEOUT: 5000,           // Pong timeout (ms)
    EVENT_BATCH_INTERVAL: 500,    // Event batching interval (ms)
    MAX_MESSAGE_QUEUE_SIZE: 100,  // Maximum queued messages
    CONNECTION_TIMEOUT: 10000,    // Connection timeout (ms)
    LONG_RETRY_DELAY: 30000,      // Long retry delay (ms)
}
```

### URL Resolution
1. Uses `VITE_WS_URL` if provided
2. Falls back to computed URL based on `VITE_API_HOST`
3. Uses secure protocol (wss/https) if site is served over HTTPS
4. Combines with `VITE_WS_PATH` for final WebSocket endpoint

## State Management

### Store Initialization
- Connection Manager: Lazy singleton via `getConnectionManager()`
- Message Store: Direct singleton
- Subscription Manager: Direct singleton
- WebSocket Store: Zustand store with persistence

### Connection States
- `CONNECTING`: Initial connection attempt
- `CONNECTED`: Successfully connected
- `DISCONNECTED`: Intentionally disconnected
- `ERROR`: Connection error occurred
- `RECONNECTING`: Attempting to reconnect
- `MAX_RETRIES_EXCEEDED`: Maximum retry attempts reached

## Error Handling
- Automatic reconnection with exponential backoff
- User notifications for connection issues
- Fallback to HTTP polling when WebSocket is unavailable
- Non-blocking connection attempts to prevent UI freezing

## Best Practices
1. Always access connection manager through `getConnectionManager()`
2. Initialize WebSocket in React components via WebSocketProvider
3. Use the useRealtimeQuery hook for data subscriptions
4. Handle connection state changes in UI components
5. Implement proper cleanup in useEffect hooks 