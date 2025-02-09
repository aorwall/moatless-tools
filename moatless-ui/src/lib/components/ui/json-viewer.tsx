import { useState } from "react"
import { ChevronRight, ChevronDown, Copy, Check } from "lucide-react"
import { cn } from "@/lib/utils"

interface JsonViewerProps {
  data: any
  level?: number
  expanded?: boolean
}

export function JsonViewer({ data, level = 0, expanded = true }: JsonViewerProps) {
  const [isExpanded, setIsExpanded] = useState(expanded)
  const [copied, setCopied] = useState(false)

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const getValueColor = (value: any) => {
    if (typeof value === "string") return "text-green-600 dark:text-green-400"
    if (typeof value === "number") return "text-blue-600 dark:text-blue-400"
    if (typeof value === "boolean") return "text-purple-600 dark:text-purple-400"
    if (value === null) return "text-gray-500 dark:text-gray-400"
    return "text-foreground"
  }

  if (typeof data !== "object" || data === null) {
    return (
      <span className={cn("font-mono break-all", getValueColor(data))}>
        {JSON.stringify(data)}
      </span>
    )
  }

  const isArray = Array.isArray(data)
  const isEmpty = Object.keys(data).length === 0

  return (
    <div className="font-mono text-xs">
      {level === 0 && (
        <button
          onClick={copyToClipboard}
          className="mb-2 flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted"
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied!" : "Copy"}
        </button>
      )}
      <div className="flex items-start gap-1">
        {!isEmpty && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="mt-0.5 rounded-sm hover:bg-muted flex-shrink-0"
            aria-label={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </button>
        )}
        <div className="min-w-0 break-words">
          <span>{isArray ? "[" : "{"}</span>
          {!isEmpty && !isExpanded && <span>...</span>}
          {isExpanded && !isEmpty && (
            <div className="ml-4">
              {Object.entries(data).map(([key, value], index) => (
                <div key={key} className="my-1">
                  <span className="text-foreground break-all">
                    {!isArray && (
                      <>
                        &quot;{key}&quot;
                        <span className="text-muted-foreground">: </span>
                      </>
                    )}
                  </span>
                  <JsonViewer data={value} level={level + 1} />
                  {index < Object.entries(data).length - 1 && <span>,</span>}
                </div>
              ))}
            </div>
          )}
          <span>{isArray ? "]" : "}"}</span>
        </div>
      </div>
    </div>
  )
}

