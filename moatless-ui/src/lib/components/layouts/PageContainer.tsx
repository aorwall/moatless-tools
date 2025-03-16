import { useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';

interface PageContainerProps {
    children: React.ReactNode;
    className?: string;
}

export function PageContainer({ children, className }: PageContainerProps) {
    const location = useLocation();
    const path = location.pathname;

    // Only evaluation instance pages should be full width
    // These have the pattern /swebench/evaluation/{evaluationId}/{instanceId}
    const isFullWidthPage =
        path.match(/^\/swebench\/evaluation\/[^\/]+\/[^\/]+$/);

    return (
        <div
            className={cn(
                "h-full w-full",
                isFullWidthPage
                    ? "p-0" // Full width with minimal padding for evaluation instance pages
                    : "container py-6", // Centered with consistent width for other pages
                className
            )}
        >
            {children}
        </div>
    );
} 