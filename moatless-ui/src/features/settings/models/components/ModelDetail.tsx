import { type ModelConfig } from '../types';
import { ModelForm } from './ModelForm';

interface ModelDetailProps {
    model: ModelConfig;
    onSubmit: (data: ModelConfig) => Promise<void>;
    [key: string]: any; // Allow additional props
}

export function ModelDetail({ model, onSubmit, ...rest }: ModelDetailProps) {
    return <ModelForm model={model} onSubmit={onSubmit} {...rest} />;
} 