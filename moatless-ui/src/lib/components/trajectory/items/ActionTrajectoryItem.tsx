import { useEffect } from "react";

export interface ActionTimelineContent extends Record<string, any> {
  errors?: string[];
  warnings?: string[];
  name?: string;
  action?: {
    name: string;
    properties?: Record<string, any>;
  };
}

export interface ActionTrajectoryItemProps {
  name: string;
  content: ActionTimelineContent;
}

export const ActionTrajectoryItem = ({
  name,
  content,
}: ActionTrajectoryItemProps) => {
  // Get the correct properties object based on the content structure
  let actionName = name || '';
  console.log(content);
  
  const isSpecialAction = ["StringReplace", "ViewCode", "str_replace_editor"].includes(actionName);
  const path = content.path;
  const command = content.command;

  const formatValue = (value: any): string => {
    if (value === null) return "null";
    if (value === undefined) return "undefined";
    if (typeof value === "number") return value.toString();
    if (typeof value === "boolean") return value.toString();
    if (typeof value === "string") {
      if (!value.includes("\n")) return value;
      return value.split("\n").join("\n");
    }
    return JSON.stringify(value, null, 2);
  };

  const truncateString = (str: string, maxLength: number = 30): string => {
    if (typeof str !== "string") return str;
    const firstLine = str.split("\n")[0];
    if (firstLine.length <= maxLength) return firstLine;
    return firstLine.slice(0, maxLength) + "...";
  };

  const getFilteredProperties = () => {
    const filtered = {...content};
    // Always remove common metadata
    delete filtered.errors;
    delete filtered.warnings;
    delete filtered.name;
    if (filtered.action) delete filtered.action;
    
    // For display purposes, we'll handle command and path separately
    delete filtered.command;
    delete filtered.path;
    
    return filtered;
  };

  const filteredProperties = getFilteredProperties();
  const otherProperties = Object.entries(filteredProperties);
  
  // For vertical layout, we can show more properties (up to 4)
  const displayProperties = otherProperties.slice(0, 4);
  const remainingPropertiesCount = Math.max(0, otherProperties.length - 4);

  return (
      <div className="text-xs text-gray-600 space-y-1">
        {/* Properties list */}
        <div className="space-y-1">
          {/* Command (highest priority) */}
          {command && (
            <div className="flex items-start">
              <span className="font-medium min-w-16">command:</span>
              <span className="font-mono break-words">"{truncateString(command, 80)}"</span>
            </div>
          )}
          
          {/* Path (second priority) */}
          {path && (
            <div className="flex items-start">
              <span className="font-medium min-w-16">path:</span>
              <span className="font-mono break-words">"{truncateString(path, 80)}"</span>
            </div>
          )}
          
          {/* Other properties */}
          {displayProperties.map(([key, value]) => (
            <div key={key} className="flex items-start">
              <span className="font-medium min-w-16">{key}:</span>
              <span className="font-mono break-words">
                {typeof value === "string"
                  ? `"${truncateString(value, 80)}"`
                  : JSON.stringify(value)}
              </span>
            </div>
          ))}
          
          {/* Show count of remaining properties */}
          {remainingPropertiesCount > 0 && (
            <div className="text-gray-400">
              and {remainingPropertiesCount} more properties
            </div>
          )}
          
          {/* Show if no properties */}
          {!command && !path && displayProperties.length === 0 && (
            <div>No properties</div>
          )}
        </div>

        {/* Errors and warnings */}
        {(content.errors || content.warnings) && (
          <div className="mt-2 space-y-1">
            {content.errors?.map((error, index) => (
              <div key={index} className="text-red-600">
                {error}
              </div>
            ))}
            {content.warnings?.map((warning, index) => (
              <div key={index} className="text-yellow-600">
                {warning}
              </div>
            ))}
          </div>
        )}
      </div>
  );
};
