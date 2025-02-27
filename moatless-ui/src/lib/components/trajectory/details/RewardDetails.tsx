import { cn } from "@/lib/utils";

interface RewardDetailsProps {
  content: {
    value: number;
    explanation?: string;
  };
}

// Define reward color type for better type safety
type RewardColor = "red" | "yellow" | "green" | "gray";

export const RewardDetails = ({ content }: RewardDetailsProps) => {
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
  
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className={cn(
          "px-3 py-1.5 rounded-md font-medium text-base",
          {
            "bg-red-100 border border-red-200": rewardColor === "red",
            "bg-yellow-100 border border-yellow-200": rewardColor === "yellow",
            "bg-green-100 border border-green-200": rewardColor === "green",
            "bg-gray-100 border border-gray-200": rewardColor === "gray"
          }
        )}>
          {clampedValue > 0 ? "+" : ""}{clampedValue}
        </div>
        <span className="text-sm text-muted-foreground">
          Reward Value
        </span>
      </div>
      
      {content.explanation && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Explanation</h3>
          <div className={cn(
            "prose prose-sm max-w-none rounded-lg p-4",
            {
              "bg-red-50 border border-red-100": rewardColor === "red",
              "bg-yellow-50 border border-yellow-100": rewardColor === "yellow",
              "bg-green-50 border border-green-100": rewardColor === "green",
              "bg-gray-50 border border-gray-100": rewardColor === "gray"
            }
          )}>
            <pre className="whitespace-pre-wrap text-sm font-mono text-gray-900">
              {content.explanation}
            </pre>
          </div>
        </div>
      )}
      
      {!content.explanation && (
        <div className="text-sm text-muted-foreground italic">
          No explanation provided for this reward.
        </div>
      )}
    </div>
  );
}; 