import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";

export interface OptionType {
  id: string;
  label: string;
}

interface GenericSelectorProps {
  value: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  options: OptionType[];
  renderAdditionalInfo?: (
    selectedOption: OptionType | undefined,
  ) => React.ReactNode;
}

export const GenericSelector: React.FC<GenericSelectorProps> = ({
  value,
  onValueChange,
  placeholder,
  options,
  renderAdditionalInfo,
}) => {
  const selectedOption = options.find((option) => option.id === value);

  return (
    <div className="space-y-4">
      <Select value={value} onValueChange={onValueChange}>
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.id} value={option.id}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {renderAdditionalInfo && selectedOption && (
        <div className="text-sm text-muted-foreground">
          {renderAdditionalInfo(selectedOption)}
        </div>
      )}
    </div>
  );
};
