"use client"

import { useState } from "react"
import type { Field, ComponentSelectField } from "@/lib/components/form/types"
import { Input } from "@/lib/components/ui/input"
import { Textarea } from "@/lib/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/lib/components/ui/select"
import { FormField } from "@/lib/components/form/form-field"
import { ToggleSetting } from "@/lib/components/form/toggle-setting"
import { DynamicItemList } from "@/lib/components/form/dynamic-item-list"
import { ExpandableTextarea } from "@/lib/components/form/expandable-textarea"

interface DynamicFieldProps {
  field: Field
  value: any
  onChange: (id: string, value: any) => void
}

export function DynamicField({ field, value, onChange }: DynamicFieldProps) {
  const handleChange = (newValue: any) => {
    onChange(field.id, newValue)
  }

  switch (field.type) {
    case "text":
      return (
        <FormField label={field.label} htmlFor={field.id} tooltip={field.tooltip}>
          <Input
            id={field.id}
            value={value ?? field.defaultValue ?? ""}
            onChange={(e) => handleChange(e.target.value)}
            placeholder={field.placeholder}
          />
        </FormField>
      )

    case "number":
      return (
        <FormField label={field.label} htmlFor={field.id} tooltip={field.tooltip}>
          <Input
            id={field.id}
            type="number"
            value={value ?? field.defaultValue ?? ""}
            onChange={(e) => handleChange(Number(e.target.value))}
            min={field.min}
            max={field.max}
            step={field.step}
          />
        </FormField>
      )

    case "textarea":
      return (
        <FormField label={field.label} htmlFor={field.id} tooltip={field.tooltip}>
          <Textarea
            id={field.id}
            value={value ?? field.defaultValue ?? ""}
            onChange={(e) => handleChange(e.target.value)}
            placeholder={field.placeholder}
            rows={field.rows}
          />
        </FormField>
      )

    case "expandable-textarea":
      return <ExpandableTextarea field={field} value={value} onChange={onChange} />

    case "select":
      return (
        <FormField label={field.label} htmlFor={field.id} tooltip={field.tooltip}>
          <Select value={value ?? field.defaultValue ?? ""} onValueChange={handleChange}>
            <SelectTrigger id={field.id}>
              <SelectValue placeholder="Select an option" />
            </SelectTrigger>
            <SelectContent>
              {field.options.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FormField>
      )

    case "toggle":
      return (
        <ToggleSetting
          label={field.label}
          description={field.description}
          checked={value ?? field.defaultValue ?? false}
          onCheckedChange={handleChange}
        />
      )

    case "component-select":
      return <DynamicComponentSelectField field={field} value={value} onChange={onChange} />

    case "dynamic-item-list":
      return <DynamicItemList field={field} value={value} onChange={onChange} />

    default:
      return <div>Unknown field type</div>
  }
}

interface DynamicComponentSelectFieldProps {
  field: ComponentSelectField
  value: any
  onChange: (id: string, value: any) => void
}

function DynamicComponentSelectField({ field, value, onChange }: DynamicComponentSelectFieldProps) {
  const [selectedOption, setSelectedOption] = useState<string>(
    value?.type ?? field.defaultValue ?? field.options[0]?.value ?? "",
  )

  const handleSelectChange = (newValue: string) => {
    setSelectedOption(newValue)
    onChange(field.id, { type: newValue })
  }

  // Get conditional fields for the selected option
  const conditionalFields = field.conditionalFields[selectedOption] || []

  return (
    <div className="space-y-4">
      <FormField label={field.label} htmlFor={field.id} tooltip={field.tooltip}>
        <Select value={selectedOption} onValueChange={handleSelectChange}>
          <SelectTrigger id={field.id}>
            <SelectValue placeholder="Select an option" />
          </SelectTrigger>
          <SelectContent>
            {field.options.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </FormField>

      <div className="text-sm font-medium text-muted-foreground">Selected: {selectedOption}</div>

      {conditionalFields.length > 0 && (
        <div className="space-y-4 pt-4 border-t">
          {conditionalFields.map((conditionalField) => (
            <DynamicField
              key={conditionalField.id}
              field={conditionalField}
              value={value?.[conditionalField.id]}
              onChange={(id, fieldValue) => {
                onChange(field.id, {
                  ...value,
                  type: selectedOption,
                  [id]: fieldValue,
                })
              }}
            />
          ))}
        </div>
      )}
    </div>
  )
}

