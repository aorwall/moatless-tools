"use client"

import type React from "react"

import { useState } from "react"
import { ChevronDown, ChevronUp, Maximize2, Minimize2 } from "lucide-react"
import { Textarea } from "@/lib/components/ui/textarea"
import { Button } from "@/lib/components/ui/button"
import type { ExpandableTextareaField } from "@/lib/components/form/types"
import { FormField } from "@/lib/components/form/form-field"
import { cn } from "@/lib/utils"

interface ExpandableTextareaProps {
  field: ExpandableTextareaField
  value: string
  onChange: (id: string, value: string) => void
}

export function ExpandableTextarea({ field, value, onChange }: ExpandableTextareaProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(field.id, e.target.value)
  }

  const toggleExpand = () => {
    setIsExpanded(!isExpanded)
  }

  const minRows = field.minRows || 3
  const maxRows = field.maxRows || 12

  return (
    <FormField label={field.label} htmlFor={field.id} tooltip={field.tooltip}>
      <div className="space-y-2">
        <div className="relative">
          <Textarea
            id={field.id}
            value={value ?? field.defaultValue ?? ""}
            onChange={handleChange}
            placeholder={field.placeholder}
            className={cn(
              "resize-none transition-all duration-200",
              isExpanded ? "min-h-[300px]" : `min-h-[${minRows * 1.5}rem]`,
            )}
            rows={isExpanded ? maxRows : minRows}
          />
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="absolute top-2 right-2 h-6 w-6 p-0 opacity-70 hover:opacity-100"
            onClick={toggleExpand}
            title={isExpanded ? "Minimize" : "Maximize"}
          >
            {isExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </Button>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="w-full flex items-center justify-center text-xs text-muted-foreground hover:text-foreground"
          onClick={toggleExpand}
        >
          {isExpanded ? (
            <>
              <ChevronUp className="h-3 w-3 mr-1" />
              Show less
            </>
          ) : (
            <>
              <ChevronDown className="h-3 w-3 mr-1" />
              Show more
            </>
          )}
        </Button>
      </div>
    </FormField>
  )
}

