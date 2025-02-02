import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { RootLayout } from '@/layouts/RootLayout';
import { SettingsLayout } from '@/layouts/SettingsLayout';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import ErrorBoundary from '@/lib/components/ErrorBoundary';
import { WebSocketProvider } from '@/lib/providers/WebSocketProvider';
import { Toaster } from "@/lib/components/ui/sonner"

// Import page components
import { Home } from '@/pages/home/index';
import { Trajectory } from '@/pages/trajectory';
import { ModelsLayout } from '@/pages/settings/models/layout';
import { ModelsPage } from '@/pages/settings/models';
import { ModelDetailPage } from '@/pages/settings/models/[id]';
import { AgentsLayout } from '@/pages/settings/agents/layout';
import { AgentsPage } from '@/pages/settings/agents';
import { AgentDetailPage } from '@/pages/settings/agents/[id]';
import { ValidatePage } from '@/pages/validate';
import { RunPage } from '@/pages/runs/[id]';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 minute
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <WebSocketProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<RootLayout />}>
                <Route index element={<Home />} />
                <Route path="trajectory" element={<Trajectory />} />
                <Route path="validate" element={<ValidatePage />} />
                
                {/* Nested settings routes */}
                <Route path="settings" element={<SettingsLayout />}>
                  <Route path="agents" element={<AgentsLayout />}>
                    <Route index element={<AgentsPage />} />
                    <Route path=":id" element={<AgentDetailPage />} />
                  </Route>
                  <Route path="models" element={<ModelsLayout />}>
                    <Route index element={<ModelsPage />} />
                    <Route path=":id" element={<ModelDetailPage />} />
                  </Route>
                </Route>

                <Route path="/runs/:id" element={<RunPage />} />
              </Route>
            </Routes>
          </BrowserRouter>
          <ReactQueryDevtools initialIsOpen={false} />
          <Toaster />
        </WebSocketProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
