import { WebSocketMessage, useWebSocketStore } from "@/lib/stores/websocketStore";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
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

  /**
   * Handle incoming messages from the WebSocket
   */
  const handleMessage = (message: WebSocketMessage) => {
    // Call the onEvent callback if provided
    if (onEvent) {
      onEvent(message);
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
  };

  useEffect(() => {
    if (!trajectoryId || !projectId) return;

    // The enhanced subscribeToTrajectory handles both client and server subscriptions
    unsubscribeRef.current = subscribeToTrajectory(projectId, trajectoryId, handleMessage);

    // Clean up when the component unmounts
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
    // Dependencies for handleMessage
    queryClient,
    onEvent,
    showToasts,
  ]);

  return {
    isConnected: isConnected(),
  };
}
