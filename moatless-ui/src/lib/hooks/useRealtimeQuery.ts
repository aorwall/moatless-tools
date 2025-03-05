import { useState, useEffect, useRef, useCallback } from 'react';
import { useQuery, UseQueryOptions, QueryKey, UseQueryResult, useQueryClient } from '@tanstack/react-query';
import { debounce } from 'lodash';
import { useWebSocketStore } from '@/lib/stores/websocketStore';
import { WebSocketMessage, ConnectionState } from '@/lib/websocket';

/**
 * Subscription types supported by the hook
 */
export type SubscriptionType = 'project' | 'trajectory';

/**
 * Event filter configuration for WebSocket messages
 */
export type EventFilter = {
    scopes?: string[];  // Filter by event scope (e.g., "flow", "evaluation")
    types?: string[];   // Filter by event_type (e.g., "started", "completed", "error")
    anyMatch?: boolean; // If true, match if any filter matches; if false, all filters must match
};

/**
 * Configuration for WebSocket message handlers
 */
export type WebSocketHandlerConfig = {
    enabled?: boolean;
    onMessage?: (message: WebSocketMessage) => void;
    debounceMs?: number;
    staleTime?: number;
    refetchInterval?: number | false;
    fallbackPollingInterval?: number | false; // Polling interval to use when WebSocket is disconnected
};

/**
 * A hook that combines React Query with WebSocket subscriptions.
 * It automatically subscribes to WebSocket events and invalidates queries when relevant events are received.
 * If WebSocket is unavailable, it falls back to polling.
 * 
 * @param params - The parameters for the hook
 * @returns The React Query result with additional WebSocket status information
 */
export function useRealtimeQuery<
    TQueryFnData = unknown,
    TError = Error,
    TData = TQueryFnData
>({
    queryKey,
    queryFn,
    subscriptionConfig,
    options = {},
    wsConfig = {}
}: {
    queryKey: QueryKey;
    queryFn: () => Promise<TQueryFnData>;
    subscriptionConfig: {
        projectId: string;
        trajectoryId?: string;
        subscriptionType?: SubscriptionType;
        eventFilter?: EventFilter;
    };
    options?: Omit<UseQueryOptions<TQueryFnData, TError, TData>, "queryKey" | "queryFn">;
    wsConfig?: WebSocketHandlerConfig;
}): UseQueryResult<TData, TError> & {
    lastMessage: WebSocketMessage | null;
    connectionState: ConnectionState;
    usingFallback: boolean;
} {
    const queryClient = useQueryClient();
    const websocketStore = useWebSocketStore();

    // State for tracking the last received WebSocket message
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
    const [connectionState, setConnectionState] = useState<ConnectionState>(
        websocketStore.getConnectionStatus()
    );
    const [usingFallback, setUsingFallback] = useState(false);

    // Extract configuration options
    const { projectId, trajectoryId, subscriptionType = 'trajectory', eventFilter } = subscriptionConfig;
    const {
        enabled = true,
        onMessage,
        debounceMs = 500,
        staleTime = 30000,
        refetchInterval = false,
        fallbackPollingInterval = 5000
    } = wsConfig;

    // Keep track of the effective refetch interval (changes based on connection state)
    const effectiveRefetchIntervalRef = useRef<number | false>(refetchInterval);

    // Create a debounced function to invalidate queries
    const debouncedInvalidate = useCallback(
        debounce(() => {
            queryClient.invalidateQueries({ queryKey });
        }, debounceMs),
        [queryClient, queryKey, debounceMs]
    );

    // Handle WebSocket messages
    const handleMessage = useCallback(
        (message: WebSocketMessage) => {
            // Update the last message state
            setLastMessage(message);

            // Call the onMessage callback if provided
            if (onMessage) {
                onMessage(message);
            }

            // Apply filters to determine if we should invalidate the query
            if (eventFilter) {
                const { scopes, types, anyMatch = false } = eventFilter;

                // Check if the message matches our filters
                const scopeMatches = !scopes || scopes.includes(message.scope || '');
                const typeMatches = !types || types.includes(message.event_type || '');

                // Determine if we should invalidate based on filter match
                const shouldInvalidate = anyMatch
                    ? (scopeMatches || typeMatches)
                    : (scopeMatches && typeMatches);

                if (shouldInvalidate) {
                    debouncedInvalidate();
                }
            } else {
                // No filters, invalidate on any message
                debouncedInvalidate();
            }
        },
        [onMessage, eventFilter, debouncedInvalidate]
    );

    // Subscribe to WebSocket connection state changes
    useEffect(() => {
        const unsubscribe = websocketStore.addStatusListener((status) => {
            setConnectionState(status);

            // Adjust polling strategy based on connection state
            if (status === ConnectionState.CONNECTED) {
                setUsingFallback(false);
                effectiveRefetchIntervalRef.current = refetchInterval;
            } else {
                // Use fallback polling when not connected
                setUsingFallback(true);
                effectiveRefetchIntervalRef.current = fallbackPollingInterval;
            }
        });

        return unsubscribe;
    }, [websocketStore, refetchInterval, fallbackPollingInterval]);

    // Subscribe to WebSocket messages
    useEffect(() => {
        if (!enabled) return;

        let unsubscribe: () => void;

        // Choose subscription method based on type
        if (subscriptionType === 'project') {
            // Subscribe to project events
            unsubscribe = websocketStore.subscribeToProject(
                projectId,
                handleMessage
            );
        } else {
            // Subscribe to trajectory events (requires trajectoryId)
            if (!trajectoryId) {
                console.error('trajectoryId is required for trajectory subscriptions');
                return;
            }

            unsubscribe = websocketStore.subscribeToTrajectory(
                projectId,
                trajectoryId,
                handleMessage
            );
        }

        return unsubscribe;
    }, [websocketStore, projectId, trajectoryId, subscriptionType, handleMessage, enabled]);

    // Execute the query with React Query
    const query = useQuery<TQueryFnData, TError, TData>({
        queryKey,
        queryFn,
        staleTime,               // Use provided stale time
        refetchInterval: effectiveRefetchIntervalRef.current,  // Use dynamic refetch interval
        refetchOnMount: true,    // Always refetch when component mounts
        refetchOnWindowFocus: true, // Refetch on window focus
        retry: 3,                // Retry failed requests 3 times
        ...options
    });

    // Return query results with additional WebSocket-related properties
    return {
        ...query,
        lastMessage,
        connectionState,
        usingFallback
    };
} 