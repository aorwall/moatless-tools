import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface FormPageLayoutProps {
    children: ReactNode;
    className?: string;
}

/**
 * A layout component that constrains the width of the content for better form display.
 * 
 * @param children The content to display
 * @param className Additional classes to apply to the container (including max-width)
 */
export function FormPageLayout({
    children,
    className
}: FormPageLayoutProps) {
    return (
        <div className="flex h-full w-full overflow-auto">
            <div className={cn(
                'mx-auto w-full p-6 max-w-3xl', // Default max-width
                className
            )}>
                {children}
            </div>
        </div>
    );
} 