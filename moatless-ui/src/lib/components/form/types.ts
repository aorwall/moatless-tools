export type FieldType =
  | "text"
  | "number"
  | "textarea"
  | "select"
  | "toggle"
  | "component-select"
  | "dynamic-item-list"
  | "expandable-textarea"

// Base field interface
export interface BaseField {
  id: string
  type: FieldType
  label: string
  tooltip?: string
  required?: boolean
}

// Text input field
export interface TextField extends BaseField {
  type: "text"
  defaultValue?: string
  placeholder?: string
}

// Number input field
export interface NumberField extends BaseField {
  type: "number"
  defaultValue?: number
  min?: number
  max?: number
  step?: number
}

// Textarea field
export interface TextareaField extends BaseField {
  type: "textarea"
  defaultValue?: string
  placeholder?: string
  rows?: number
}

// Expandable textarea field
export interface ExpandableTextareaField extends BaseField {
  type: "expandable-textarea"
  defaultValue?: string
  placeholder?: string
  minRows?: number
  maxRows?: number
}

// Select field
export interface SelectField extends BaseField {
  type: "select"
  options: { value: string; label: string }[]
  defaultValue?: string
}

// Toggle field
export interface ToggleField extends BaseField {
  type: "toggle"
  defaultValue?: boolean
  description?: string
}

// Component selector field with conditional fields
export interface ComponentSelectField extends BaseField {
  type: "component-select"
  options: { value: string; label: string }[]
  defaultValue?: string
  conditionalFields: {
    [key: string]: Field[] // Fields to show when a specific option is selected
  }
}

export interface DynamicListItem {
  id: string
  name: string
  description: string
  fields: Field[]
  [key: string]: any // For additional properties
}

export interface DynamicListField extends BaseField {
  type: "dynamic-item-list"
  addButtonText?: string // Add this property
  availableItems: {
    id: string
    name: string
    description: string
    fields: Field[]
  }[]
  defaultValue?: DynamicListItem[]
}

// Union type of all field types
export type Field =
  | TextField
  | NumberField
  | TextareaField
  | SelectField
  | ToggleField
  | ComponentSelectField
  | DynamicListField
  | ExpandableTextareaField

// Section of fields
export interface FormSection {
  id: string
  title: string
  description?: string
  fields: Field[]
}

export interface FormSchema {
  id: string
  title: string
  sections: FormSection[]
}

// Settings values stored as a record
export type FormValues = Record<string, any>

