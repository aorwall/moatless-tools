import { RunnerStatusBar } from "@/features/runner/components/RunnerStatusBar";
import { Link, Outlet, useLocation } from "react-router-dom";

export function RootLayout() {
  const location = useLocation();

  const isActivePath = (path: string) => {
    return location.pathname.startsWith(path);
  };

  // Check if the current route is an evaluation instance page
  const isEvaluationInstancePage = /^\/swebench\/evaluation\/[^/]+\/[^/]+$/.test(location.pathname);

  const showBetaFeatures = true;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-10 border-b bg-background">
        <div className="px-4">
          <nav className="flex h-14 items-center justify-between">
            <div className="flex items-center space-x-8">
              <Link to="/" className="text-lg font-semibold">
                Moatless Tools
              </Link>
              <div className="flex items-center space-x-4">
                {showBetaFeatures && (
                  <>
                    <Link
                      to="/trajectories"
                      className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${isActivePath("/trajectories")
                          ? "text-primary"
                          : "text-muted-foreground"
                        }`}
                    >
                      Trajectories
                    </Link>
                    <Link
                      to="/loop"
                      className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${isActivePath("/loop")
                          ? "text-primary"
                          : "text-muted-foreground"
                        }`}
                    >
                      Run Loop
                    </Link>
                  </>
                )}
                <Link
                  to="/swebench/evaluation"
                  className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${isActivePath("/swebench/evaluation")
                      ? "text-primary"
                      : "text-muted-foreground"
                    }`}
                >
                  Evaluation
                </Link>
                <Link
                  to="/settings/agents"
                  className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${isActivePath("/settings/agents")
                      ? "text-primary"
                      : "text-muted-foreground"
                    }`}
                >
                  Agents
                </Link>
                <Link
                  to="/settings/models"
                  className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${isActivePath("/settings/models")
                      ? "text-primary"
                      : "text-muted-foreground"
                    }`}
                >
                  Models
                </Link>
                <Link
                  to="/settings/flows"
                  className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${isActivePath("/settings/flows")
                      ? "text-primary"
                      : "text-muted-foreground"
                    }`}
                >
                  Flows
                </Link>
              </div>
            </div>

            {/* Runner Status Bar */}
            <div className="flex-shrink-0">
              <RunnerStatusBar />
            </div>
          </nav>
        </div>
      </header>

      <main>
        <Outlet />
      </main>
    </div>
  );
}
