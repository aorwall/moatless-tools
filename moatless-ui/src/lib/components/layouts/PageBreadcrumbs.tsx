import { useLocation, Link } from 'react-router-dom';
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from '@/lib/components/ui/breadcrumb';

// Map of route segments to display names
const routeNameMap: Record<string, string> = {
    settings: 'Settings',
    agents: 'Agents',
    models: 'Models',
    base: 'Base Models',
    create: 'Create',
    new: 'New',
    flows: 'Flows',
    trajectories: 'Trajectories',
    swebench: 'SWEBench',
    evaluation: 'Evaluation',
    loop: 'Loop',
    runner: 'Runner',
};

export function PageBreadcrumbs() {
    const location = useLocation();
    const pathSegments = location.pathname.split('/').filter(Boolean);

    // Don't show breadcrumbs on the home page
    if (pathSegments.length === 0) {
        return null;
    }

    // Build breadcrumb items
    const breadcrumbItems = pathSegments.map((segment, index) => {
        // Check if this segment is an ID (doesn't have a mapping)
        const isId = !routeNameMap[segment];
        const displayName = routeNameMap[segment] || segment;

        // Build the path up to this segment
        const path = `/${pathSegments.slice(0, index + 1).join('/')}`;

        // If it's the last segment, render as a page
        if (index === pathSegments.length - 1) {
            return (
                <BreadcrumbItem key={path}>
                    <BreadcrumbPage>{displayName}</BreadcrumbPage>
                </BreadcrumbItem>
            );
        }

        // Otherwise, render as a link
        return (
            <BreadcrumbItem key={path}>
                <BreadcrumbLink asChild>
                    <Link to={path}>{displayName}</Link>
                </BreadcrumbLink>
                <BreadcrumbSeparator />
            </BreadcrumbItem>
        );
    });

    return (
        <Breadcrumb>
            <BreadcrumbList>
                <BreadcrumbItem>
                    <BreadcrumbLink asChild>
                        <Link to="/">Home</Link>
                    </BreadcrumbLink>
                    <BreadcrumbSeparator />
                </BreadcrumbItem>
                {breadcrumbItems}
            </BreadcrumbList>
        </Breadcrumb>
    );
} 