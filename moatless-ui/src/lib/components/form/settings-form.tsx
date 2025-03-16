import { useState, useEffect, ReactNode } from "react"
import { SectionCard } from "@/lib/components/form/section-card"
import { DynamicField } from "@/lib/components/form/dynamic-field"
import { Button } from "@/lib/components/ui/button"
import { FormValues } from "@/lib/components/form/types"
import { FormSchema } from "@/lib/components/form/types"
import { useForm, FormProvider } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { z } from "zod"

interface FormContainerProps {
  schema: FormSchema
  initialValues?: FormValues
  onSave?: (values: FormValues) => void
  actionButtons?: ReactNode
  showDuplicateButton?: boolean
  onDuplicate?: () => void
  zodSchema?: z.ZodType<any>
}

export function FormContainer({
  schema,
  initialValues = {},
  onSave,
  actionButtons,
  showDuplicateButton = false,
  onDuplicate,
  zodSchema
}: FormContainerProps) {
  const [isDirty, setIsDirty] = useState(false)

  // Initialize React Hook Form
  const methods = useForm({
    defaultValues: initialValues,
    resolver: zodSchema ? zodResolver(zodSchema) : undefined,
  })

  const { handleSubmit, reset, formState, watch } = methods

  // Watch for form changes to set isDirty state
  useEffect(() => {
    const subscription = watch(() => {
      setIsDirty(true)
    })
    return () => subscription.unsubscribe()
  }, [watch])

  // Reset form when initialValues change
  useEffect(() => {
    reset(initialValues)
    setIsDirty(false)
  }, [initialValues, reset])

  const handleFieldChange = (id: string, value: any) => {
    methods.setValue(id, value, {
      shouldDirty: true,
      shouldValidate: true
    })
  }

  const onSubmit = (data: FormValues) => {
    console.log('FormContainer onSubmit called with data:', data);
    console.log('isDirty:', isDirty);
    console.log('formState:', formState);

    if (onSave) {
      console.log('Calling onSave callback');
      onSave(data);
      setIsDirty(false);
    } else {
      console.log('No onSave callback provided');
    }
  }

  return (
    <FormProvider {...methods}>
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6 pb-24">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold">{schema.title}</h1>
          <div className="flex gap-2">
            {showDuplicateButton && (
              <Button variant="outline" onClick={onDuplicate} type="button">Duplicate</Button>
            )}
            {actionButtons}
            <Button type="submit" disabled={!isDirty || formState.isSubmitting}>
              {formState.isSubmitting ? "Saving..." : "Save Changes"}
            </Button>
          </div>
        </div>

        {schema.sections.map((section) => (
          <SectionCard key={section.id} title={section.title} description={section.description}>
            <div className="space-y-4">
              {section.fields.map((field) => (
                <DynamicField
                  key={field.id}
                  field={field}
                  value={methods.watch(field.id)}
                  onChange={handleFieldChange}
                />
              ))}
            </div>
          </SectionCard>
        ))}

        {/* Sticky Save Button */}
        <div className="fixed bottom-0 left-0 right-0 bg-background border-t p-4 flex justify-end">
          <div className="w-full flex justify-end gap-2">
            {actionButtons}
            <Button
              type="submit"
              disabled={!isDirty || formState.isSubmitting}
            >
              {formState.isSubmitting ? "Saving..." : "Save Changes"}
            </Button>
          </div>
        </div>
      </form>
    </FormProvider>
  )
}

