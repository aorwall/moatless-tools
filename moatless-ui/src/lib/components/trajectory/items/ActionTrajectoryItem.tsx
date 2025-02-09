import { useEffect } from "react";

export interface ActionTimelineContent {
  properties: Record<string, any>;
  errors?: string[];
  warnings?: string[];
}

export interface ActionTrajectoryItemProps {
  content: ActionTimelineContent;
  expandedState: boolean;
}

export const ActionTrajectoryItem = ({
  content,
  expandedState,
}: ActionTrajectoryItemProps) => {
  const isExpandable = Object.keys(content.properties || {}).length > 0;

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

  const isMultilineString = (value: any): boolean => {
    return typeof value === "string" && value.includes("\n");
  };

  const isObject = (value: any): boolean => {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  };

  const truncateString = (str: string, maxLength: number = 30): string => {
    if (typeof str !== "string") return str;
    const firstLine = str.split("\n")[0];
    if (firstLine.length <= maxLength) return firstLine;
    return firstLine.slice(0, maxLength) + "...";
  };

  const firstTwoProperties = Object.entries(content.properties || {}).slice(
    0,
    2,
  );
  const remainingPropertiesCount =
    Object.keys(content.properties || {}).length - 2;

  if (expandedState) {
    return (
      <>
        <div className="overflow-x-auto rounded-md bg-gray-50 p-2 sm:p-3">
          {content.properties && (
            <div className="min-w-[300px] space-y-1.5 sm:space-y-2">
              {Object.entries(content.properties).map(([key, value]) => (
                <div
                  key={key}
                  className="grid grid-cols-[120px,minmax(200px,1fr)] gap-1 sm:gap-2"
                >
                  <div className="truncate text-xs font-medium text-gray-600">
                    {key}:
                  </div>
                  <div className="overflow-x-auto font-mono text-xs">
                    {typeof value === "number" ||
                    typeof value === "boolean" ||
                    value === null ||
                    value === undefined ? (
                      <span className="text-blue-600">
                        {formatValue(value)}
                      </span>
                    ) : typeof value === "string" &&
                      !isMultilineString(value) ? (
                      <span className="whitespace-nowrap text-green-600">
                        "{value}"
                      </span>
                    ) : typeof value === "string" &&
                      isMultilineString(value) ? (
                      <div className="whitespace-pre border-l-2 border-gray-200 pl-2 text-green-600 sm:pl-4">
                        {formatValue(value)}
                      </div>
                    ) : isObject(value) ? (
                      <div className="whitespace-pre pl-2 sm:pl-4">
                        {Object.entries(value).map(([objKey, objValue]) => (
                          <div key={objKey} className="py-0.5">
                            <span className="text-purple-600">{objKey}</span>:
                            {typeof objValue === "string" ? (
                              <span className="whitespace-nowrap text-green-600">
                                "{objValue}"
                              </span>
                            ) : typeof objValue === "number" ||
                              typeof objValue === "boolean" ? (
                              <span className="text-blue-600">{objValue}</span>
                            ) : (
                              <span className="whitespace-nowrap">
                                {JSON.stringify(objValue)}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <pre className="whitespace-pre">{formatValue(value)}</pre>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {(content.errors?.length || content.warnings?.length) && (
          <div className="mt-2 space-y-1">
            {content.errors?.map((error, index) => (
              <div key={index} className="text-xs text-red-600">
                {error}
              </div>
            ))}
            {content.warnings?.map((warning, index) => (
              <div key={index} className="text-xs text-yellow-600">
                {warning}
              </div>
            ))}
          </div>
        )}
      </>
    );
  }

  return (
    <>
      <div className="overflow-x-auto whitespace-nowrap text-xs text-gray-600">
        {firstTwoProperties.length ? (
          <>
            {firstTwoProperties.map(([key, value], i) => (
              <span key={key}>
                {i > 0 && <span className="mx-1 sm:mx-2">|</span>}
                <span className="font-medium">{key}:</span>
                <span className="font-mono">
                  {typeof value === "string"
                    ? `"${truncateString(value, 40)}"`
                    : JSON.stringify(value)}
                </span>
              </span>
            ))}
            {remainingPropertiesCount > 0 && (
              <span className="ml-1 text-gray-400 sm:ml-2">
                and {remainingPropertiesCount} more
              </span>
            )}
          </>
        ) : (
          "No properties"
        )}
      </div>

      {(content.errors?.length || content.warnings?.length) && (
        <div className="mt-2 space-y-1">
          {content.errors?.map((error, index) => (
            <div key={index} className="text-xs text-red-600">
              {error}
            </div>
          ))}
          {content.warnings?.map((warning, index) => (
            <div key={index} className="text-xs text-yellow-600">
              {warning}
            </div>
          ))}
        </div>
      )}
    </>
  );
};
