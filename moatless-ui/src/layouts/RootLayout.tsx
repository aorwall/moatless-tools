import { RunnerStatusBar } from "@/features/runner/components/RunnerStatusBar";
import { Outlet } from "react-router-dom";
import { MainSidebar } from "@/lib/components/layouts/MainSidebar";
import { SidebarProvider, SidebarTrigger } from "@/lib/components/ui/sidebar";
import { PageBreadcrumbs } from "@/lib/components/layouts/PageBreadcrumbs";
import { PageContainer } from "@/lib/components/layouts/PageContainer";

export function RootLayout() {
  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full">
        {/* Main Sidebar */}
        <MainSidebar />

        {/* Main Content */}
        <div className="flex-1 flex flex-col min-h-0">
          {/* Header - Full Width */}
          <header className="sticky top-0 z-10 border-b bg-background w-full">
            <div className="w-full px-4 h-14 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <SidebarTrigger />
                <PageBreadcrumbs />
              </div>
              {/* Runner Status Bar */}
              <RunnerStatusBar />
            </div>
          </header>

          {/* Main Content - Uses PageContainer for consistent width */}
          <main className="flex-1 overflow-auto">
            <PageContainer>
              <Outlet />
            </PageContainer>
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
}
