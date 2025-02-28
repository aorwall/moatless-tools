import React from "react";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface ExpandableItemProps {
  label: string;
  icon: LucideIcon;
  onExpandChange: (value: boolean) => void;
  children: React.ReactNode;
}

export const ExpandableItem = ({
  label,
  icon: Icon,
  onExpandChange,
  children,
}: ExpandableItemProps) => {
  const handleClick = () => {
    onExpandChange(true);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    e.preventDefault();
    onExpandChange(true);
  
  };

  return (
    <div
      className={cn(
        "flex items-start gap-2",
        "group/expandable transition-all duration-150",
        "cursor-pointer"
      )}
    >
      <div className="flex w-[80px] shrink-0 items-start justify-end sm:w-[150px]">
        <div className="mr-3 flex h-8 flex-col justify-center text-right sm:mr-6">
          <div>
            <button
              className={cn(
                "text-xs font-medium",
                "max-w-[60px] truncate sm:max-w-[120px]",
                "transition-colors duration-150",
              
                  "text-gray-600 group-hover/expandable:text-gray-900"
                
              )}
              onClick={handleClick}
            >
              {label}
            </button>
          </div>
        </div>
      </div>

      <div
        className={cn(
          "relative z-10 -ml-2 flex h-8 min-w-[2rem] items-center justify-center",
          "rounded-full border-2 bg-white transition-all duration-150",
          "border-gray-200 group-hover/expandable:border-gray-300 group-hover/expandable:bg-gray-50 group-hover/expandable:shadow-sm"
          
        )}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
      >
        <Icon className={cn(
          "h-4 w-4",
          "text-gray-400 group-hover/expandable:text-gray-500"
          
        )} />
      </div>

      <div className="min-w-0 flex-1 overflow-x-auto pl-4 sm:pl-8">
        <div className="min-w-[300px]">{children}</div>
      </div>
    </div>
  );
};
