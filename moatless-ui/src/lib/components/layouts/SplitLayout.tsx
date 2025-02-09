import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface SplitLayoutProps {
  left: ReactNode;
  right: ReactNode;
  className?: string;
}

export function SplitLayout({ left, right, className }: SplitLayoutProps) {
  return (
    <div className={cn("flex h-full min-h-0", className)}>
      <div className="w-80 flex-none border-r min-h-0">{left}</div>
      <div className="flex-1 min-h-0">{right}</div>
    </div>
  );
}
