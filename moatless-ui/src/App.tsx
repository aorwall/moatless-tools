import { RootLayout } from "@/layouts/RootLayout";
import { SettingsLayout } from "@/layouts/SettingsLayout";
import ErrorBoundary from "@/lib/components/ErrorBoundary";
import { Toaster } from "@/lib/components/ui/sonner";
import { WebSocketProvider } from "@/lib/providers/WebSocketProvider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { TrajectoryPage } from "@/features/trajectory/pages/TrajectoryPage";
import { AgentsPage } from "@/pages/settings/agents";
import { AgentDetailPage } from "@/pages/settings/agents/[id]";
import { AgentsLayout } from "@/pages/settings/agents/layout";
import { NewAgentPage } from "@/pages/settings/agents/new";
import { FlowsPage } from "@/pages/settings/flows";
import { FlowDetailPage } from "@/pages/settings/flows/[id]";
import { FlowsLayout } from "@/pages/settings/flows/layout";
import { NewFlowPage } from "@/pages/settings/flows/new";
import { ModelsPage } from "@/pages/settings/models";
import { ModelDetailPage } from "@/pages/settings/models/[id]";
import { ModelsLayout } from "@/pages/settings/models/layout";

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

                <Route path="settings" element={<SettingsLayout />}>
                  <Route path="agents" element={<AgentsLayout />}>
                    <Route index element={<AgentsPage />} />
                    <Route path="new" element={<NewAgentPage />} />
                    <Route path=":id" element={<AgentDetailPage />} />
                  </Route>
                  <Route path="models" element={<ModelsLayout />}>
                    <Route index element={<ModelsPage />} />
                    <Route path="base" element={<BaseModelsPage />} />
                    <Route path="create" element={<CreateModelPage />} />
                    <Route path=":id" element={<ModelDetailPage />} />
                  </Route>
                  <Route path="flows" element={<FlowsLayout />}>
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
                    <Route element={<EvaluationLayout />}>
                      <Route
                        path=":evaluationId/:instanceId"
                        element={<EvaluationInstancePage />}
                      />
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
