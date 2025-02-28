import { FC } from "react";
import { Badge } from "@/lib/components/ui/badge";
import { cn } from "@/lib/utils";

export interface RewardTimelineContent {
  value: number;
  explanation?: string;
}

export interface RewardTrajectoryItemProps {
  content: RewardTimelineContent;
}

// Define reward color type for better type safety
type RewardColor = "red" | "yellow" | "green" | "gray";

export const RewardTrajectoryItem: FC<RewardTrajectoryItemProps> = ({
  content,
}) => {
  
  // Clamp the reward value between -100 and 100
  const clampedValue = Math.max(-100, Math.min(100, content.value));
  
  // Determine color based on reward value
  const getRewardColor = (value: number): RewardColor => {
    if (value === 0) return "yellow";
    
    if (value < 0) {
      // For negative values, interpolate between red (-100) and yellow (0)
      return value <= -50 ? "red" : "yellow";
    }
    
    // For positive values, interpolate between yellow (0) and green (100)
    return value >= 50 ? "green" : "yellow";
  };
  
  const rewardColor = getRewardColor(clampedValue);
  
  // Get badge variant and styles based on reward color
  const getBadgeStyles = (color: RewardColor) => {
    switch (color) {
      case "red":
        return {
          variant: "destructive" as const,
          className: "bg-red-600 hover:bg-red-700"
        };
      case "yellow":
        return {
          variant: "outline" as const,
          className: "border-yellow-500 text-yellow-700 bg-yellow-50"
        };
      case "green":
        return {
          variant: "default" as const,
          className: "bg-green-600 hover:bg-green-700"
        };
      case "gray":
        return {
          variant: "outline" as const,
          className: ""
        };
    }
  };
  
  const badgeStyles = getBadgeStyles(rewardColor);

  // Format explanation text for better display
  const formatExplanation = (text?: string) => {
    if (!text) return "";
    
    // Replace single newlines with breaks to preserve formatting
    return text
      .split("\n")
      .map((line, i) => (
        <span key={i} className="block">
          {line || "\u00A0"}
        </span>
      ));
  };

  return (
    <div className="space-y-4">
      {/* Reward Value Summary */}
      <div className="flex items-center gap-4">
        <Badge
          variant={badgeStyles.variant}
          className={cn("px-2 py-0.5 text-xs font-medium", badgeStyles.className)}
        >
          {clampedValue > 0 ? "+" : ""}{clampedValue}
        </Badge>
        
        {content.explanation && (
          <span className="text-sm text-gray-700 truncate max-w-[400px]">
            {content.explanation.split("\n")[0]}
            {content.explanation.includes("\n") && "..."}
          </span>
        )}
      </div>

  
    </div>
  );
}; 