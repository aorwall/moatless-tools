import { FC, useState } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, ChevronRight } from "lucide-react";

interface JsonViewProps {
  data: any;
  defaultExpanded?: boolean;
  level?: number;
  showBraces?: boolean;
  expanded?: boolean;
}

const isExpandable = (value: any): boolean => {
  return value !== null && (typeof value === "object" || Array.isArray(value));
};

const getValueClass = (value: any): string => {
  if (value === null) return "text-gray-400";
  if (typeof value === "string") return "text-emerald-600";
  if (typeof value === "number") return "text-blue-600";
  if (typeof value === "boolean") return "text-purple-600";
  return "text-gray-600";
};

const formatValue = (value: any): string => {
  if (value === null) return "null";
  if (typeof value === "string") {
    return `"${value.replace(/\n/g, "\n")}"`;
  }
  return String(value);
};

const getPreview = (value: any): string => {
  if (Array.isArray(value)) {
    return `Array(${value.length})`;
  }
  if (typeof value === "object" && value !== null) {
    return `Object(${Object.keys(value).length})`;
  }
  return formatValue(value);
};

export const JsonView: FC<JsonViewProps> = ({
  data,
  defaultExpanded = false,
  level = 0,
  showBraces = true,
  expanded: expandAll = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded || expandAll);

  const toggle = () => {
    setIsExpanded(!isExpanded);
  };

  const paddingLeft = `${level * 1}rem`;

  return (
    <div className="text-xs font-mono" style={{ paddingLeft }}>
      {showBraces && (
        <div className="flex items-start">
          {isExpandable(data) && (
            <button
              className="p-0.5 hover:bg-gray-50 rounded-sm transition-colors duration-200 -ml-1"
              onClick={toggle}
            >
              <span className="text-gray-400 flex items-center justify-center w-4 h-4">
                {isExpanded ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
              </span>
            </button>
          )}
          <span className="text-gray-500">
            {Array.isArray(data) ? "[" : "{"}
          </span>
        </div>
      )}

      {isExpanded || !isExpandable(data) ? (
        <>
          {Array.isArray(data) ? (
            data.map((item, i) => (
              <div key={i} className="flex items-start py-0.5">
                {isExpandable(item) ? (
                  <JsonView
                    data={item}
                    level={level + 1}
                    defaultExpanded={false}
                    expanded={expandAll}
                  />
                ) : (
                  <span
                    className={cn(
                      getValueClass(item),
                      "whitespace-pre-wrap font-mono",
                    )}
                    style={{ paddingLeft: `${level + 1}rem` }}
                  >
                    {formatValue(item)}
                  </span>
                )}
                {i < data.length - 1 && (
                  <span className="text-gray-400">,</span>
                )}
              </div>
            ))
          ) : typeof data === "object" && data !== null ? (
            Object.entries(data).map(([key, value], i, arr) => (
              <div key={key} className="flex items-start py-0.5">
                <span
                  className="text-gray-600"
                  style={{ paddingLeft: `${level + 1}rem` }}
                >
                  {key}:
                </span>
                {isExpandable(value) ? (
                  <JsonView
                    data={value}
                    level={level + 1}
                    defaultExpanded={false}
                    expanded={expandAll}
                  />
                ) : (
                  <span
                    className={cn(
                      "ml-2",
                      getValueClass(value),
                      "whitespace-pre-wrap",
                    )}
                  >
                    {formatValue(value)}
                  </span>
                )}
                {i < arr.length - 1 && <span className="text-gray-400">,</span>}
              </div>
            ))
          ) : (
            <span
              className={cn(getValueClass(data), "whitespace-pre-wrap")}
              style={{ paddingLeft: `${level + 1}rem` }}
            >
              {formatValue(data)}
            </span>
          )}
        </>
      ) : (
        <span className="text-gray-400 ml-2 hover:text-gray-600 transition-colors duration-200">
          {getPreview(data)}
        </span>
      )}

      {showBraces && (
        <div style={{ paddingLeft: `${level}rem` }}>
          <span className="text-gray-500">
            {Array.isArray(data) ? "]" : "}"}
          </span>
        </div>
      )}
    </div>
  );
};
