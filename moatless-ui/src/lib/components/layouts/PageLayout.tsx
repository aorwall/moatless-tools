import { cn } from "@/lib/utils";

interface PageLayoutProps {
  children: React.ReactNode;
  className?: string;
}

export function PageLayout({ children, className }: PageLayoutProps) {
  return (
    <div className="min-h-[calc(100vh-56px)] bg-background">
      <div className={cn(
        "container mx-auto max-w-7xl py-8",
        className
      )}>
        {children}
      </div>
    </div>
  );
} 