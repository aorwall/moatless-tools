import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRealtimeQuery } from '../hooks/useRealtimeQuery';
import { WebSocketProvider } from '../WebSocketProvider';
import { ConnectionState } from '../types';

// Mock WebSocket provider context
vi.mock('../WebSocketProvider', () => ({
    WebSocketProvider: ({ children }: { children: React.ReactNode }) => children,
    useWebSocket: () => ({
        isConnected: true,
        connectionState: ConnectionState.CONNECTED,
        error: null
    })
}));

describe('useRealtimeQuery', () => {
    let queryClient: QueryClient;

    beforeEach(() => {
        queryClient = new QueryClient({
            defaultOptions: {
                queries: {
                    retry: false,
                    cacheTime: 0
                }
            }
        });
    });

    afterEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
    });

    const wrapper = ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>
            <WebSocketProvider>{children}</WebSocketProvider>
        </QueryClientProvider>
    );

    it('should fetch data and handle WebSocket updates', async () => {
        const mockQueryFn = vi.fn().mockResolvedValue({ data: 'initial' });
        const mockEventFilter = {
            type: 'trajectory',
            action: 'update'
        };

        const { result } = renderHook(
            () =>
                useRealtimeQuery({
                    queryKey: ['test-query'],
                    queryFn: mockQueryFn,
                    eventFilter: mockEventFilter
                }),
            { wrapper }
        );

        // Initial fetch
        expect(result.current.isLoading).toBe(true);
        await act(async () => {
            await result.current.refetch();
        });
        expect(result.current.data).toEqual({ data: 'initial' });

        // Simulate WebSocket message
        act(() => {
            result.current.onMessage({
                type: 'trajectory',
                action: 'update',
                data: { updated: true }
            });
        });

        // Should trigger a refetch
        expect(mockQueryFn).toHaveBeenCalledTimes(2);
    });

    it('should handle connection state changes', async () => {
        const mockQueryFn = vi.fn().mockResolvedValue({ data: 'test' });
        let isConnected = true;

        vi.mock('../WebSocketProvider', () => ({
            WebSocketProvider: ({ children }: { children: React.ReactNode }) => children,
            useWebSocket: () => ({
                isConnected,
                connectionState: isConnected ? ConnectionState.CONNECTED : ConnectionState.DISCONNECTED,
                error: null
            })
        }));

        const { result, rerender } = renderHook(
            () =>
                useRealtimeQuery({
                    queryKey: ['test-query'],
                    queryFn: mockQueryFn,
                    eventFilter: { type: 'test' }
                }),
            { wrapper }
        );

        // Connected state
        expect(result.current.usingFallback).toBe(false);

        // Simulate disconnection
        isConnected = false;
        rerender();

        // Should switch to fallback mode
        expect(result.current.usingFallback).toBe(true);
    });

    it('should handle message filtering', async () => {
        const mockQueryFn = vi.fn().mockResolvedValue({ data: 'test' });
        const mockEventFilter = {
            type: 'trajectory',
            action: 'update',
            projectId: 'test-project'
        };

        const { result } = renderHook(
            () =>
                useRealtimeQuery({
                    queryKey: ['test-query'],
                    queryFn: mockQueryFn,
                    eventFilter: mockEventFilter
                }),
            { wrapper }
        );

        // Matching message
        act(() => {
            result.current.onMessage({
                type: 'trajectory',
                action: 'update',
                project_id: 'test-project',
                data: { updated: true }
            });
        });
        expect(mockQueryFn).toHaveBeenCalledTimes(2);

        // Non-matching message
        act(() => {
            result.current.onMessage({
                type: 'trajectory',
                action: 'delete',
                project_id: 'test-project',
                data: { updated: true }
            });
        });
        expect(mockQueryFn).toHaveBeenCalledTimes(2); // Should not trigger refetch
    });

    it('should handle polling fallback', async () => {
        const mockQueryFn = vi.fn().mockResolvedValue({ data: 'test' });
        const { result } = renderHook(
            () =>
                useRealtimeQuery({
                    queryKey: ['test-query'],
                    queryFn: mockQueryFn,
                    eventFilter: { type: 'test' },
                    pollingInterval: 1000
                }),
            { wrapper }
        );

        // Force fallback mode
        act(() => {
            result.current.setUsingFallback(true);
        });

        expect(result.current.usingFallback).toBe(true);

        // Advance timer to trigger polling
        vi.advanceTimersByTime(1000);
        expect(mockQueryFn).toHaveBeenCalledTimes(2);
    });

    it('should cleanup subscriptions on unmount', () => {
        const mockQueryFn = vi.fn().mockResolvedValue({ data: 'test' });
        const { unmount } = renderHook(
            () =>
                useRealtimeQuery({
                    queryKey: ['test-query'],
                    queryFn: mockQueryFn,
                    eventFilter: { type: 'test' }
                }),
            { wrapper }
        );

        unmount();
        // Verify cleanup (would need to spy on subscription cleanup)
    });

    it('should handle debounced updates', async () => {
        const mockQueryFn = vi.fn().mockResolvedValue({ data: 'test' });
        const { result } = renderHook(
            () =>
                useRealtimeQuery({
                    queryKey: ['test-query'],
                    queryFn: mockQueryFn,
                    eventFilter: { type: 'test' },
                    debounceInterval: 100
                }),
            { wrapper }
        );

        // Send multiple messages in quick succession
        act(() => {
            result.current.onMessage({ type: 'test', data: { value: 1 } });
            result.current.onMessage({ type: 'test', data: { value: 2 } });
            result.current.onMessage({ type: 'test', data: { value: 3 } });
        });

        // Should only trigger one refetch after debounce
        vi.advanceTimersByTime(100);
        expect(mockQueryFn).toHaveBeenCalledTimes(2); // Initial + 1 debounced update
    });
}); 