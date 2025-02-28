import React, { useState } from "react";
import { MessageSquare, ChevronDown } from "lucide-react";
import { truncateMessage } from "@/lib/utils/text";
import { cn } from "@/lib/utils";

interface UserMessageItemProps {
  message: string;
  maxLength?: number;
}

export const UserMessageItem = ({ 
  message, 
  maxLength = 200 
}: UserMessageItemProps) => {
  const isLongMessage = message.length > maxLength;
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm p-3 mb-3">
      <div className="flex items-start gap-2.5">
        <div className="mt-0.5 p-1.5 rounded-md">
          <MessageSquare className="h-4 w-4 text-primary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm text-gray-700 whitespace-pre-wrap break-words">
            {isExpanded || !isLongMessage 
              ? message 
              : truncateMessage(message, maxLength)}
          </div>
          
          {isLongMessage && (
            <button 
              onClick={() => setIsExpanded(!isExpanded)}
              className="mt-2 text-xs text-primary hover:text-primary/80 flex items-center gap-1"
            >
              {isExpanded ? "Show less" : "Show more"}
              <ChevronDown className={cn(
                "h-3 w-3 transition-transform",
                { "transform rotate-180": isExpanded }
              )} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}; 