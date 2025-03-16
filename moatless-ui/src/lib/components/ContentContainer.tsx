import type React from "react";
import { cn } from "@/lib/utils";

interface ContentContainerProps {
  children: React.ReactNode;
  className?: string;
}

export function ContentContainer({ children, className }: ContentContainerProps) {
  return (
    <div className={cn("bg-white shadow-lg rounded-lg", className)}>
      {children}
    </div>
  );
}
