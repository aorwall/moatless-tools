import { Link, useLocation } from "react-router-dom";
import {
    LayoutDashboard,
    Settings,
    GitBranch,
    Play,
    Layers,
    Bot,
    Cpu,
    FlaskConical,
} from "lucide-react";
import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarFooter,
} from "@/lib/components/ui/sidebar";

// Menu items for main navigation
const mainItems = [
    {
        title: "Evaluation",
        url: "/swebench/evaluation",
        icon: FlaskConical,
    },
    {
        title: "Trajectories",
        url: "/trajectories",
        icon: GitBranch,
    },
    {
        title: "Run Loop",
        url: "/loop",
        icon: Play,
    },
    {
        title: "Runner Dashboard",
        url: "/runner",
        icon: LayoutDashboard,
    },
];

// Menu items for settings
const settingsItems = [
    {
        title: "Agents",
        url: "/settings/agents",
        icon: Bot,
    },
    {
        title: "Models",
        url: "/settings/models",
        icon: Cpu,
    },
    {
        title: "Flows",
        url: "/settings/flows",
        icon: Layers,
    },
];

export function MainSidebar() {
    const location = useLocation();

    const isActive = (url: string) => {
        return location.pathname.startsWith(url);
    };

    return (
        <Sidebar>
            <SidebarHeader className="border-b">
                <div className="flex h-14 items-center px-4">
                    <Link to="/" className="flex items-center">
                        <h1 className="text-xl font-bold">Moatless Tools</h1>
                    </Link>
                </div>
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupLabel>Main</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {mainItems.map((item) => (
                                <SidebarMenuItem key={item.title}>
                                    <SidebarMenuButton asChild isActive={isActive(item.url)}>
                                        <Link to={item.url}>
                                            <item.icon className="h-4 w-4" />
                                            <span>{item.title}</span>
                                        </Link>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>
                            ))}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
                <SidebarGroup>
                    <SidebarGroupLabel>Settings</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {settingsItems.map((item) => (
                                <SidebarMenuItem key={item.title}>
                                    <SidebarMenuButton asChild isActive={isActive(item.url)}>
                                        <Link to={item.url}>
                                            <item.icon className="h-4 w-4" />
                                            <span>{item.title}</span>
                                        </Link>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>
                            ))}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
            </SidebarContent>
            <SidebarFooter className="border-t p-4">
                <div className="text-xs text-muted-foreground">
                    Moatless Tools v1.0
                </div>
            </SidebarFooter>
        </Sidebar>
    );
} 