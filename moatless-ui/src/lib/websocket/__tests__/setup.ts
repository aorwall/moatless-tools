import { expect, vi } from 'vitest';
import * as matchers from '@testing-library/jest-dom/matchers';
import { JSDOM } from 'jsdom';

// Extend expect with jest-dom matchers
expect.extend(matchers);

// Create a JSDOM instance
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    url: 'http://localhost:5173',
    referrer: 'http://localhost:5173',
    contentType: 'text/html',
});

// Set up the global environment
Object.defineProperty(global, 'window', {
    value: dom.window,
    writable: true
});

Object.defineProperty(global, 'document', {
    value: dom.window.document,
    writable: true
});

Object.defineProperty(global, 'navigator', {
    value: dom.window.navigator,
    writable: true
});

// Mock import.meta.env
vi.stubGlobal('import', {
    meta: {
        env: {
            VITE_WS_URL: 'ws://localhost:8000',
            VITE_WS_PATH: '/api/ws',
            VITE_API_HOST: 'localhost:8000',
            MODE: 'test',
            DEV: true,
            PROD: false,
            SSR: false
        }
    }
});

// Mock WebSocket
class MockWebSocket {
    onopen: ((event: Event) => void) | null = null;
    onclose: ((event: CloseEvent) => void) | null = null;
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: Event) => void) | null = null;
    readyState: number = WebSocket.CONNECTING;
    url: string = '';
    protocol: string = '';
    extensions: string = '';
    bufferedAmount: number = 0;
    binaryType: BinaryType = 'blob';

    constructor(url: string, protocols?: string | string[]) {
        this.url = url;
        if (typeof protocols === 'string') {
            this.protocol = protocols;
        }
        setTimeout(() => {
            this.readyState = WebSocket.OPEN;
            if (this.onopen) {
                this.onopen(new Event('open'));
            }
        }, 0);
    }

    send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
        // Mock send implementation
    }

    close(code?: number, reason?: string): void {
        this.readyState = WebSocket.CLOSED;
        if (this.onclose) {
            this.onclose(new CloseEvent('close', { code, reason }));
        }
    }
}

// Mock global WebSocket
vi.stubGlobal('WebSocket', MockWebSocket);

// Mock console methods
console.error = vi.fn();
console.warn = vi.fn();
console.log = vi.fn();

// Mock timers
vi.useFakeTimers();

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
    observe() { }
    unobserve() { }
    disconnect() { }
};

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
    observe() { }
    unobserve() { }
    disconnect() { }
} as any; 