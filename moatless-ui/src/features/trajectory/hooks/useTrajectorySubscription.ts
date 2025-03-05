import { WebSocketMessage, useWebSocketStore } from "@/lib/stores/websocketStore";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";

/**
 * Hook to subscribe to trajectory updates, handling both server-side and client-side subscriptions.
 *
 * @param trajectoryId - The ID of the trajectory to subscribe to
 * @param projectId - The ID of the project that contains the trajectory
 * @param options - Optional configuration options
 * @returns Object containing subscription status
 */
export function useTrajectorySubscription(
  trajectoryId: string,
  projectId: string,
  options?: {
    onEvent?: (message: WebSocketMessage) => void;
    showToasts?: boolean;
  },
) {
  const {
    onEvent,
    showToasts = false
  } = options || {};

  const queryClient = useQueryClient();
  const {
    subscribeToTrajectory,
    isConnected,
  } = useWebSocketStore();

  // Keep track of the unsubscribe function
  const unsubscribeRef = useRef<(() => void) | null>(null);

  // Store the callback in a ref to avoid dependency changes
  const onEventRef = useRef(onEvent);
  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  /**
   * Handle incoming messages from the WebSocket
   * Using useCallback to memoize the function
   */
  const handleMessage = useCallback((message: WebSocketMessage) => {
    // Call the onEvent callback if provided
    if (onEventRef.current) {
      onEventRef.current(message);
    }

    // Show toast notifications if enabled
    if (showToasts) {
      if (message.type === "error") {
        toast.error(message.message || "An error occurred");
      } else if (message.type === "success") {
        toast.success(message.message || "Operation completed successfully");
      } else if (message.type === "info") {
        toast.info(message.message || "Information update");
      }
    }

    // Invalidate queries based on message type
    if (message.type === "trajectory_updated") {
      queryClient.invalidateQueries({ queryKey: ["trajectory", trajectoryId] });
    }
  }, [trajectoryId, queryClient, showToasts]);

  // Store previous values to compare and avoid unnecessary resubscriptions
  const prevTrajectoryId = useRef(trajectoryId);
  const prevProjectId = useRef(projectId);

  useEffect(() => {
    if (!trajectoryId || !projectId) return;

    // Only resubscribe if the IDs have changed
    if (
      trajectoryId !== prevTrajectoryId.current ||
      projectId !== prevProjectId.current ||
      !unsubscribeRef.current
    ) {
      // Clean up previous subscription if it exists
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
        unsubscribeRef.current = null;
      }

      // Update previous values
      prevTrajectoryId.current = trajectoryId;
      prevProjectId.current = projectId;

      // The enhanced subscribeToTrajectory handles both client and server subscriptions
      unsubscribeRef.current = subscribeToTrajectory(projectId, trajectoryId, handleMessage);
    }

    // Clean up when the component unmounts or dependencies change
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
        unsubscribeRef.current = null;
      }
    };
  }, [
    trajectoryId,
    projectId,
    subscribeToTrajectory,
    handleMessage,
  ]);

  return {
    isConnected: isConnected(),
  };
}
