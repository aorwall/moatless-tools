import { Field } from '@/lib/components/form/types';

/**
 * Converts a JSON Schema property to a form field
 */
export function schemaPropertyToField(propName: string, propSchema: any): Field {
    // Get property type
    let propType = propSchema.type;

    // Handle anyOf case (common in the API response)
    if (propSchema.anyOf) {
        // Find the first type that's not null
        const nonNullType = propSchema.anyOf.find((t: any) => t.type !== 'null');
        if (nonNullType) {
            propType = nonNullType.type;
        }
    }

    // Get description and title
    const description = propSchema.description || '';
    const label = propSchema.title || propName;

    // Create appropriate field based on property type
    switch (propType) {
        case 'string':
            // Handle enum type (dropdown)
            if (propSchema.enum) {
                return {
                    id: propName,
                    type: 'select',
                    label,
                    tooltip: description,
                    options: propSchema.enum.map((value: string) => ({
                        value,
                        label: value
                    })),
                    defaultValue: propSchema.default || '',
                };
            }

            // Handle const type (fixed value)
            if (propSchema.const) {
                return {
                    id: propName,
                    type: 'text',
                    label,
                    tooltip: description,
                    defaultValue: propSchema.const,
                };
            }

            // Regular string
            return {
                id: propName,
                type: 'text',
                label,
                tooltip: description,
                defaultValue: propSchema.default || '',
            };

        case 'integer':
        case 'number':
            return {
                id: propName,
                type: 'number',
                label,
                tooltip: description,
                defaultValue: propSchema.default !== undefined ? propSchema.default : 0,
                min: propSchema.minimum,
                max: propSchema.maximum,
                step: propSchema.multipleOf || 1,
            };

        case 'boolean':
            return {
                id: propName,
                type: 'toggle',
                label,
                tooltip: description,
                defaultValue: propSchema.default || false,
            };

        case 'object':
            return {
                id: propName,
                type: 'expandable-textarea',
                label,
                tooltip: description,
                defaultValue: JSON.stringify(propSchema.default || {}, null, 2),
            };

        case 'array':
            return {
                id: propName,
                type: 'expandable-textarea',
                label,
                tooltip: description,
                defaultValue: JSON.stringify(propSchema.default || [], null, 2),
            };

        default:
            // Default to text field
            return {
                id: propName,
                type: 'text',
                label,
                tooltip: description,
            };
    }
}

/**
 * Converts a JSON Schema object to form fields
 */
export function schemaToFields(schema: any): Field[] {
    if (!schema || !schema.properties) {
        return [];
    }

    return Object.entries(schema.properties)
        .filter(([propName]) => !propName.startsWith('_') && propName !== 'type')
        .map(([propName, propSchema]) => schemaPropertyToField(propName, propSchema));
} 