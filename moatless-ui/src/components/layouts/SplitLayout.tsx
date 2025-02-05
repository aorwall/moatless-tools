import { ReactNode } from "react";

interface SplitLayoutProps {
  left: ReactNode;
  right: ReactNode;
}

export function SplitLayout({ left, right }: SplitLayoutProps) {
  return (
    <div className="flex h-full min-h-0">
      <aside className="w-64 h-full min-h-0 border-r">{left}</aside>
      <main className="flex-1 h-full min-h-0">{right}</main>
    </div>
  );
}
