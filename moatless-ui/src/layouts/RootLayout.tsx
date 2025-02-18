import { Link, Outlet, useLocation } from "react-router-dom";

export function RootLayout() {
  const location = useLocation();

  const isActivePath = (path: string) => {
    return location.pathname.startsWith(path);
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <header className="flex-none border-b bg-background">
        <div className="px-4">
          <nav className="flex h-14 items-center space-x-8">
            <Link to="/" className="text-lg font-semibold">
              Moatless Tools
            </Link>
            <div className="flex items-center space-x-4">
              <Link
                to="/trajectories"
                className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${
                  isActivePath("/trajectories")
                    ? "text-primary"
                    : "text-muted-foreground"
                }`}
              >
                Trajectories
              </Link>
              <Link
                to="/loop"
                className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${
                  isActivePath("/loop")
                    ? "text-primary"
                    : "text-muted-foreground"
                }`}
              >
                Loop
              </Link>
              <Link
                to="/swebench/evaluation"
                className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${
                  isActivePath("/swebench/evaluation")
                    ? "text-primary"
                    : "text-muted-foreground"
                }`}
              >
                Evaluation
              </Link>
              <Link
                to="/settings/agents"
                className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${
                  isActivePath("/settings/agents")
                    ? "text-primary"
                    : "text-muted-foreground"
                }`}
              >
                Agents
              </Link>
              <Link
                to="/settings/models"
                className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${
                  isActivePath("/settings/models")
                    ? "text-primary"
                    : "text-muted-foreground"
                }`}
              >
                Models
              </Link>
              <Link
                to="/settings/flows"
                className={`px-3 py-2 text-sm font-medium transition-colors hover:text-primary ${
                  isActivePath("/settings/flows")
                    ? "text-primary"
                    : "text-muted-foreground"
                }`}
              >
                Flows
              </Link>
            </div>
          </nav>
        </div>
      </header>

      <main className="flex-1 min-h-0">
        <Outlet />
      </main>
    </div>
  );
}
