import { Input } from "@/lib/components/ui/input"
import { Switch } from "@/lib/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/lib/components/ui/select"
import { FormField } from "@/lib/components/form/form-field"
import { Textarea } from "@/lib/components/ui/textarea"
import { Badge } from "@/lib/components/ui/badge"

interface PropertyType {
    type?: string;
    $ref?: string;
}

interface ComponentPropertyProps {
    id: string
    property: any
    value: any
    onChange: (id: string, value: any) => void
}

export function ComponentProperty({ id, property, value, onChange }: ComponentPropertyProps) {
    // Determine property type
    const getPropertyType = () => {
        // Check for enum first (select dropdown)
        if (property.enum?.length) {
            return 'enum';
        }

        // Check for anyOf with integer and null (optional number)
        if (property.anyOf?.some((item: PropertyType) => item.type === 'integer') &&
            property.anyOf?.some((item: PropertyType) => item.type === 'null')) {
            return 'integer';
        }

        // Check for anyOf with number and null (optional number)
        if (property.anyOf?.some((item: PropertyType) => item.type === 'number') &&
            property.anyOf?.some((item: PropertyType) => item.type === 'null')) {
            return 'number';
        }

        // Check for anyOf with string and null (optional string)
        if (property.anyOf?.some((item: PropertyType) => item.type === 'string') &&
            property.anyOf?.some((item: PropertyType) => item.type === 'null')) {
            return 'string';
        }

        // Check for const value (fixed value)
        if (property.const !== undefined) {
            return 'const';
        }

        // Use the direct type if available
        if (property.type) {
            return property.type;
        }

        // Default to string if we can't determine
        return 'string';
    };

    const propertyType = getPropertyType();
    const label = property.title || id;
    const description = property.description || '';
    const isRequired = property.required === true;

    // Handle change based on property type
    const handleChange = (newValue: any) => {
        onChange(id, newValue);
    };

    // For const values, just show a disabled input with a badge
    if (propertyType === 'const') {
        return (
            <FormField label={label} htmlFor={id} tooltip={description}>
                <div className="flex items-center gap-2">
                    <Input
                        id={id}
                        value={property.const}
                        disabled
                        className="flex-1"
                    />
                    <Badge variant="outline">Constant</Badge>
                </div>
            </FormField>
        );
    }

    // For enum values, show a select dropdown
    if (propertyType === 'enum') {
        return (
            <FormField label={label} htmlFor={id} tooltip={description}>
                <Select
                    value={value !== undefined ? String(value) : String(property.default || '')}
                    onValueChange={handleChange}
                >
                    <SelectTrigger id={id}>
                        <SelectValue placeholder={`Select ${label}`} />
                    </SelectTrigger>
                    <SelectContent>
                        {property.enum.map((option: string) => (
                            <SelectItem key={option} value={option}>
                                {option}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
            </FormField>
        );
    }

    // For boolean values, show a switch
    if (propertyType === 'boolean') {
        return (
            <FormField label={label} htmlFor={id} tooltip={description}>
                <div className="flex items-center">
                    <Switch
                        id={id}
                        checked={value !== undefined ? Boolean(value) : Boolean(property.default || false)}
                        onCheckedChange={handleChange}
                    />
                    <span className="ml-2 text-sm text-muted-foreground">
                        {value !== undefined ? (Boolean(value) ? 'Enabled' : 'Disabled') : (Boolean(property.default || false) ? 'Enabled' : 'Disabled')}
                    </span>
                </div>
            </FormField>
        );
    }

    // For integer or number values, show a number input
    if (propertyType === 'integer' || propertyType === 'number') {
        return (
            <FormField label={label} htmlFor={id} tooltip={description}>
                <Input
                    id={id}
                    type="number"
                    value={value !== undefined ? value : (property.default !== undefined ? property.default : '')}
                    onChange={(e) => handleChange(Number(e.target.value))}
                    step={propertyType === 'integer' ? 1 : 0.1}
                    min={property.minimum}
                    max={property.maximum}
                    placeholder={property.placeholder || `Enter ${label}`}
                />
                {(property.minimum !== undefined || property.maximum !== undefined) && (
                    <p className="text-xs text-muted-foreground mt-1">
                        {property.minimum !== undefined && property.maximum !== undefined
                            ? `Range: ${property.minimum} to ${property.maximum}`
                            : property.minimum !== undefined
                                ? `Minimum: ${property.minimum}`
                                : `Maximum: ${property.maximum}`}
                    </p>
                )}
            </FormField>
        );
    }

    // For object values, show a textarea with JSON
    if (propertyType === 'object' || propertyType === 'array') {
        const stringValue = typeof value === 'object'
            ? JSON.stringify(value, null, 2)
            : (value || (propertyType === 'array' ? '[]' : '{}'));

        return (
            <FormField label={label} htmlFor={id} tooltip={description}>
                <Textarea
                    id={id}
                    value={stringValue}
                    onChange={(e) => {
                        try {
                            handleChange(JSON.parse(e.target.value));
                        } catch {
                            // If not valid JSON, just store as string
                            handleChange(e.target.value);
                        }
                    }}
                    rows={5}
                    placeholder={`Enter ${propertyType === 'array' ? 'array' : 'object'} as JSON`}
                />
            </FormField>
        );
    }

    // Default to string input for all other types
    return (
        <FormField label={label} htmlFor={id} tooltip={description}>
            <Input
                id={id}
                value={value !== undefined ? value : (property.default !== undefined ? property.default : '')}
                onChange={(e) => handleChange(e.target.value)}
                placeholder={property.placeholder || `Enter ${label}`}
            />
        </FormField>
    );
} 