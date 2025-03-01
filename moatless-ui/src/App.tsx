import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { RootLayout } from "@/layouts/RootLayout";
import { SettingsLayout } from "@/layouts/SettingsLayout";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import ErrorBoundary from "@/lib/components/ErrorBoundary";
import { WebSocketProvider } from "@/lib/providers/WebSocketProvider";
import { Toaster } from "@/lib/components/ui/sonner";

import { Trajectory } from "@/pages/trajectory";
import { ModelsLayout } from "@/pages/settings/models/layout";
import { ModelsPage } from "@/pages/settings/models";
import { ModelDetailPage } from "@/pages/settings/models/[id]";
import { AgentsLayout } from "@/pages/settings/agents/layout";
import { AgentsPage } from "@/pages/settings/agents";
import { AgentDetailPage } from "@/pages/settings/agents/[id]";
import { ValidatePage } from "@/pages/validate";
import { TrajectoryPage } from "@/features/trajectory/pages/TrajectoryPage";
import { NewAgentPage } from "@/pages/settings/agents/new";
import { FlowsLayout } from "@/pages/settings/flows/layout";
import { FlowsPage } from "@/pages/settings/flows";
import { FlowDetailPage } from "@/pages/settings/flows/[id]";
import { NewFlowPage } from "@/pages/settings/flows/new";

import { EvaluationsPage } from "@/pages/swebench/evaluation";
import { CreateEvaluationPage } from "@/pages/swebench/evaluation/create";
import { EvaluationDetailsPage } from "@/pages/swebench/evaluation/[id]";
import { EvaluationInstancePage } from "@/pages/swebench/evaluation/[id]/index";
import { EvaluationLayout } from "@/features/swebench/components/EvaluationLayout";
import { BaseModelsPage } from "./pages/settings/models/base";
import { CreateModelPage } from "./pages/settings/models/create";
import RunLoopPage, { RunPage } from "./features/loop/pages/RunLoopPage";
import { TrajectoryListPage } from "./features/trajectory/pages/TrajectoryListPage";
import { RunnerDashboardPage } from "./features/runner/pages/RunnerDashboardPage";

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
                <Route index element={<Navigate to="/swebench/evaluation" replace />} />
                <Route path="trajectory" element={<Trajectory />} />
                <Route path="validate" element={<ValidatePage />} />
                <Route path="loop" element={<RunLoopPage />} />
                <Route path="runner" element={<RunnerDashboardPage />} />

                {/* Nested settings routes */}
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
                <Route path="/trajectories/:projectId/:trajectoryId" element={<TrajectoryPage />} />
                <Route path="/swebench">
                  <Route path="evaluation">
                    <Route index element={<EvaluationsPage />} />
                    <Route path="create" element={<CreateEvaluationPage />} />
                    <Route path=":evaluationId" element={<EvaluationDetailsPage />} />
                    <Route element={<EvaluationLayout />}>
                      <Route path=":evaluationId/:instanceId" element={<EvaluationInstancePage />} />
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
