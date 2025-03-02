import { useWebSocket } from '@/lib/hooks/useWebSocket';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import { toast } from 'sonner';

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
        onEvent?: (message: any) => void;
        showToasts?: boolean;
        queryInvalidation?: boolean;
    }
) {
    const {
        onEvent,
        showToasts = false,
        queryInvalidation = true,
    } = options || {};

    const queryClient = useQueryClient();
    const {
        subscribe,
        subscribeToTrajectory,
        unsubscribeFromTrajectory,
        subscribeToProject,
        unsubscribeFromProject,
        isConnected
    } = useWebSocket();

    // Handle client-side subscription for local UI updates
    useEffect(() => {
        if (!trajectoryId) return;

        const handleMessage = (message: any) => {
            // Invalidate queries if enabled
            if (queryInvalidation && message.type === 'event') {
                queryClient.invalidateQueries({ queryKey: ['trajectory', trajectoryId] });
            }

            // Call custom event handler if provided
            if (onEvent) {
                onEvent(message);
            }
        };

        // Subscribe to local event handlers for both trajectory and project
        const unsubscribeTrajectory = subscribe(`trajectory.${trajectoryId}`, handleMessage);
        const unsubscribeProject = subscribe(`project.${projectId}`, handleMessage);

        return () => {
            unsubscribeTrajectory();
            unsubscribeProject();
        };
    }, [trajectoryId, projectId, subscribe, queryClient, onEvent, queryInvalidation]);

    // Handle server-side WebSocket subscription
    useEffect(() => {
        if (!trajectoryId || !projectId) return;

        const setupSubscriptions = async () => {
            try {
                if (!isConnected) {
                    // We'll subscribe when the connection is established via the reconnection logic
                    if (showToasts) {
                        toast.info("Waiting for WebSocket connection to subscribe to updates...");
                    }
                    return;
                }

                // Always subscribe to both the trajectory and its project
                const [trajectorySuccess, projectSuccess] = await Promise.all([
                    subscribeToTrajectory(trajectoryId),
                    subscribeToProject(projectId)
                ]);

                if (!trajectorySuccess || !projectSuccess) {
                    console.warn(
                        `Subscription partial or failed: trajectory=${trajectoryId} (${trajectorySuccess}), ` +
                        `project=${projectId} (${projectSuccess})`
                    );
                    if (showToasts) {
                        toast.error("Failed to subscribe to some real-time updates");
                    }
                } else {
                    console.log(`Successfully subscribed to trajectory ${trajectoryId} and project ${projectId}`);
                    if (showToasts) {
                        toast.success("Subscribed to real-time updates");
                    }
                }
            } catch (error) {
                console.error("Error subscribing to updates:", error);
                if (showToasts) {
                    toast.error("Failed to subscribe to real-time updates");
                }
            }
        };

        setupSubscriptions();

        // Unsubscribe when component unmounts
        return () => {
            if (isConnected) {
                Promise.all([
                    unsubscribeFromTrajectory(trajectoryId).catch(error => {
                        console.error("Error unsubscribing from trajectory:", error);
                    }),
                    unsubscribeFromProject(projectId).catch(error => {
                        console.error("Error unsubscribing from project:", error);
                    })
                ]);
            }
        };
    }, [
        trajectoryId,
        projectId,
        subscribeToTrajectory,
        subscribeToProject,
        unsubscribeFromTrajectory,
        unsubscribeFromProject,
        isConnected,
        showToasts
    ]);

    return {
        isConnected,
    };
} 