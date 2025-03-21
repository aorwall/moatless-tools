import { RootLayout } from "@/layouts/RootLayout";
import ErrorBoundary from "@/lib/components/ErrorBoundary";
import { Toaster } from "@/lib/components/ui/sonner";
import { WebSocketProvider } from "@/lib/providers/WebSocketProvider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { TrajectoryPage } from "@/features/trajectory/pages/TrajectoryPage";
import { AgentDetailPage } from "@/features/settings/agents/AgentDetailPage";
import { AgentNewPage } from "@/features/settings/agents/AgentNewPage";
import { AgentsPage } from "@/features/settings/agents/AgentsPage";
import { FlowDetailPage } from "@/features/settings/flows/FlowDetailPage";
import { FlowsPage } from "@/features/settings/flows/FlowsPage";
import { NewFlowPage } from "@/features/settings/flows/NewFlowPage";
import { ModelsPage } from "@/features/settings/models/ModelsPage";
import { ModelDetailPage } from "@/features/settings/models/ModelDetailPage";

import { EvaluationLayout } from "@/features/swebench/components/EvaluationLayout";
import { EvaluationsPage } from "@/pages/swebench/evaluation";
import { CreateEvaluationPage } from "@/pages/swebench/evaluation/create";
import RunLoopPage from "./features/loop/pages/RunLoopPage";
import { RunnerDashboardPage } from "./features/runner/pages/RunnerDashboardPage";
import { EvaluationInstancePage } from "./features/swebench/pages/EvaluationInstancePage";
import { TrajectoryListPage } from "./features/trajectory/pages/TrajectoryListPage";
import { BaseModelsPage } from "./pages/settings/models/base";
import { CreateModelPage } from "./pages/settings/models/create";
import { EvaluationPage } from "./features/swebench/pages/EvaluationPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 minute
      refetchOnWindowFocus: false,
      retry: false, // Disable retries by default
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
                <Route
                  index
                  element={<Navigate to="/swebench/evaluation" replace />}
                />
                <Route path="loop" element={<RunLoopPage />} />
                <Route path="runner" element={<RunnerDashboardPage />} />

                <Route path="settings">
                  <Route path="agents">
                    <Route index element={<AgentsPage />} />
                    <Route path="new" element={<AgentNewPage />} />
                    <Route path=":id" element={<AgentDetailPage />} />
                  </Route>
                  <Route path="models">
                    <Route index element={<ModelsPage />} />
                    <Route path="base" element={<BaseModelsPage />} />
                    <Route path="create" element={<CreateModelPage />} />
                    <Route path=":id" element={<ModelDetailPage />} />
                  </Route>
                  <Route path="flows">
                    <Route index element={<FlowsPage />} />
                    <Route path="new" element={<NewFlowPage />} />
                    <Route path=":id" element={<FlowDetailPage />} />
                  </Route>
                </Route>

                <Route path="/trajectories" element={<TrajectoryListPage />} />
                <Route
                  path="/trajectories/:projectId/:trajectoryId"
                  element={<TrajectoryPage />}
                />
                <Route path="/swebench">
                  <Route path="evaluation">
                    <Route index element={<EvaluationsPage />} />
                    <Route path="create" element={<CreateEvaluationPage />} />
                    <Route
                      path=":evaluationId"
                      element={<EvaluationPage />}
                    />
                    <Route
                      path=":evaluationId/:instanceId"
                      element={<EvaluationLayout />}
                    >
                      <Route index element={<EvaluationInstancePage />} />
                    </Route>
                  </Route>
                </Route>
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
