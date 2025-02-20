import React from "react";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface ExpandableItemProps {
  label: string;
  icon: LucideIcon;
  isExpandable: boolean;
  expandedState: boolean;
  onExpandChange: (value: boolean) => void;
  children: React.ReactNode;
}

export const ExpandableItem = ({
  label,
  icon: Icon,
  isExpandable,
  expandedState,
  onExpandChange,
  children,
}: ExpandableItemProps) => {
  const handleClick = () => {
    if (isExpandable) {
      onExpandChange(!expandedState);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (isExpandable && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      onExpandChange(!expandedState);
    }
  };

  return (
    <div className={cn(
      "mb-4 flex items-start sm:mb-8",
      isExpandable && "group/expandable",
      "relative"
    )}>
      <div className={cn(
        "absolute inset-0 rounded-lg transition-colors duration-200",
        "opacity-0 group-hover/expandable:opacity-100",
        isExpandable && "bg-gray-50/50"
      )} />

      <div className="flex w-[80px] shrink-0 items-start justify-end sm:w-[150px] relative">
        <div className="mr-3 flex h-8 flex-col justify-center text-right sm:mr-6">
          <div>
            <button
              className={cn(
                "text-xs font-medium",
                "transition-colors duration-200",
                isExpandable ? [
                  "text-primary-600 group-hover/expandable:text-primary-700",
                  "cursor-pointer"
                ] : [
                  "text-gray-600",
                  "cursor-default"
                ],
                "max-w-[60px] truncate sm:max-w-[120px]",
              )}
              onClick={() => isExpandable && onExpandChange(!expandedState)}
            >
              {label}
            </button>
          </div>
        </div>
      </div>

      <div
        className={cn(
          "relative z-10 -ml-2 flex h-8 min-w-[2rem] items-center justify-center",
          "rounded-full border-2 border-gray-200 bg-white",
          "transition-colors duration-150 sm:-ml-4",
          isExpandable ? "cursor-pointer" : "cursor-default",
          "group-hover:border-gray-300 group-hover:bg-gray-50",
        )}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={isExpandable ? 0 : -1}
      >
        <Icon className="h-4 w-4 text-gray-400 group-hover:text-gray-500" />
      </div>

      <div className="min-w-0 flex-1 overflow-x-auto pl-4 sm:pl-8">
        <div className="min-w-[300px]">{children}</div>
      </div>
    </div>
  );
};
