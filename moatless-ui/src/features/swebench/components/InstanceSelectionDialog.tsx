import React from "react";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "@/lib/components/ui/dialog";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/lib/components/ui/table";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/lib/components/ui/tooltip";
import {
    Pagination,
    PaginationContent,
    PaginationItem,
    PaginationLink,
    PaginationNext,
    PaginationPrevious,
} from "@/lib/components/ui/pagination";
import { Input } from "@/lib/components/ui/input";
import { Label } from "@/lib/components/ui/label";
import { Button } from "@/lib/components/ui/button";
import { Checkbox } from "@/lib/components/ui/checkbox";
import { Badge } from "@/lib/components/ui/badge";
import { FileIcon, InfoIcon } from "lucide-react";
import { useInstances } from "../hooks/useInstances";
import { SWEBenchInstance } from "../api/instances";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/lib/components/ui/select";
import { InstanceDetailsDialog } from "./InstanceDetailsDialog";

interface InstanceSelectionDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onInstancesSelect: (selectedInstanceIds: string[]) => void;
    selectedInstanceIds: string[];
}

interface InstanceFilterState {
    search: string;
    repo: string;
    dataset: string;
    minResolved: number;
    maxResolved: number;
    minFiles: number;
    maxFiles: number;
    page: number;
    limit: number;
    selectedInstances: Record<string, boolean>;
}

// Dataset options
const DATASET_OPTIONS = [
    { value: "all", label: "All Datasets" },
    { value: "princeton-nlp/SWE-bench_Verified", label: "SWE-bench Verified" },
    { value: "princeton-nlp/SWE-bench_Lite", label: "SWE-bench Lite" },
    { value: "SWE-Gym/SWE-Gym", label: "SWE-Gym" },
];

export function InstanceSelectionDialog({
    open,
    onOpenChange,
    onInstancesSelect,
    selectedInstanceIds,
}: InstanceSelectionDialogProps) {
    // Initialize filter state with pre-selected instances
    const [filterState, setFilterState] = React.useState<InstanceFilterState>(() => {
        const selectedInstancesMap: Record<string, boolean> = {};
        selectedInstanceIds.forEach((id) => {
            selectedInstancesMap[id] = true;
        });

        return {
            search: "",
            repo: "",
            dataset: "all",
            minResolved: 0,
            maxResolved: 40,
            minFiles: 0,
            maxFiles: 40,
            page: 1,
            limit: 10,
            selectedInstances: selectedInstancesMap,
        };
    });

    // State for instance details dialog
    const [detailsDialogOpen, setDetailsDialogOpen] = React.useState(false);
    const [selectedInstanceId, setSelectedInstanceId] = React.useState<string | null>(null);

    // Sync selected instances when prop changes
    React.useEffect(() => {
        const selectedInstancesMap: Record<string, boolean> = {};
        selectedInstanceIds.forEach((id) => {
            selectedInstancesMap[id] = true;
        });

        setFilterState((prev) => ({
            ...prev,
            selectedInstances: selectedInstancesMap,
        }));
    }, [selectedInstanceIds]);

    // Get instances with filtering - now we pass more filters to the backend
    const queryParams = React.useMemo(() => ({
        page: filterState.page,
        limit: filterState.limit,
        search: filterState.search || undefined,
        dataset: filterState.dataset !== "all" ? filterState.dataset : undefined,
        repo: filterState.repo || undefined,
        min_resolved: filterState.minResolved,
        max_resolved: filterState.maxResolved,
        min_files: filterState.minFiles,
        max_files: filterState.maxFiles,
    }), [
        filterState.page,
        filterState.limit,
        filterState.search,
        filterState.dataset,
        filterState.repo,
        filterState.minResolved,
        filterState.maxResolved,
        filterState.minFiles,
        filterState.maxFiles
    ]);

    const {
        data: instancesResponse,
        isLoading: instancesLoading,
        isError: instancesError,
    } = useInstances(queryParams);

    // Get selected instance IDs from filter state
    const getSelectedInstanceIds = (): string[] => {
        return Object.entries(filterState.selectedInstances)
            .filter(([_, isSelected]) => isSelected)
            .map(([id]) => id);
    };

    const filteredInstances = instancesResponse?.instances || [];

    // Handle instance selection toggling
    const toggleInstanceSelection = (instanceId: string) => {
        setFilterState((prev) => ({
            ...prev,
            selectedInstances: {
                ...prev.selectedInstances,
                [instanceId]: !prev.selectedInstances[instanceId],
            },
        }));
    };

    // Reset filters
    const resetFilters = () => {
        setFilterState((prev) => ({
            ...prev,
            search: "",
            repo: "",
            dataset: "all",
            minResolved: 0,
            maxResolved: 100,
            minFiles: 0,
            maxFiles: 100,
            page: 1,
        }));
    };

    // Handle dialog close and update parent component
    const handleDone = () => {
        onInstancesSelect(getSelectedInstanceIds());
        onOpenChange(false);
    };

    // Show instance details dialog
    const showInstanceDetails = (instanceId: string) => {
        setSelectedInstanceId(instanceId);
        setDetailsDialogOpen(true);
    };

    return (
        <>
            <Dialog open={open} onOpenChange={onOpenChange}>
                <DialogContent className="sm:max-w-[90vw] h-[90vh] flex flex-col">
                    <DialogHeader>
                        <DialogTitle>Select Instances</DialogTitle>
                    </DialogHeader>

                    <div className="flex-1 overflow-hidden flex flex-col">
                        {/* Filters */}
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
                            <div className="space-y-2">
                                <Label htmlFor="search">Search</Label>
                                <Input
                                    id="search"
                                    placeholder="Search by ID or problem statement"
                                    value={filterState.search}
                                    onChange={(e) =>
                                        setFilterState((prev) => ({
                                            ...prev,
                                            search: e.target.value,
                                            page: 1
                                        }))
                                    }
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="dataset">Dataset</Label>
                                <Select
                                    value={filterState.dataset}
                                    onValueChange={(value) =>
                                        setFilterState((prev) => ({
                                            ...prev,
                                            dataset: value,
                                            page: 1
                                        }))
                                    }
                                >
                                    <SelectTrigger id="dataset">
                                        <SelectValue placeholder="All Datasets" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {DATASET_OPTIONS.map((option) => (
                                            <SelectItem key={option.value} value={option.value}>
                                                {option.label}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="repo">Repository</Label>
                                <Input
                                    id="repo"
                                    placeholder="Filter by repository"
                                    value={filterState.repo}
                                    onChange={(e) =>
                                        setFilterState((prev) => ({
                                            ...prev,
                                            repo: e.target.value,
                                            page: 1
                                        }))
                                    }
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="resolvedRange">
                                    Resolved Count: {filterState.minResolved} - {filterState.maxResolved}
                                </Label>
                                <div className="flex items-center gap-2">
                                    <Input
                                        type="number"
                                        min={0}
                                        max={filterState.maxResolved}
                                        value={filterState.minResolved}
                                        onChange={(e) =>
                                            setFilterState((prev) => ({
                                                ...prev,
                                                minResolved: Math.max(0, parseInt(e.target.value) || 0),
                                                page: 1
                                            }))
                                        }
                                        className="w-20"
                                    />
                                    <span>to</span>
                                    <Input
                                        type="number"
                                        min={filterState.minResolved}
                                        value={filterState.maxResolved}
                                        onChange={(e) =>
                                            setFilterState((prev) => ({
                                                ...prev,
                                                maxResolved: Math.max(prev.minResolved, parseInt(e.target.value) || 0),
                                                page: 1
                                            }))
                                        }
                                        className="w-20"
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="filesRange">
                                    File Count: {filterState.minFiles} - {filterState.maxFiles}
                                </Label>
                                <div className="flex items-center gap-2">
                                    <Input
                                        type="number"
                                        min={0}
                                        max={filterState.maxFiles}
                                        value={filterState.minFiles}
                                        onChange={(e) =>
                                            setFilterState((prev) => ({
                                                ...prev,
                                                minFiles: Math.max(0, parseInt(e.target.value) || 0),
                                                page: 1
                                            }))
                                        }
                                        className="w-20"
                                    />
                                    <span>to</span>
                                    <Input
                                        type="number"
                                        min={filterState.minFiles}
                                        value={filterState.maxFiles}
                                        onChange={(e) =>
                                            setFilterState((prev) => ({
                                                ...prev,
                                                maxFiles: Math.max(prev.minFiles, parseInt(e.target.value) || 0),
                                                page: 1
                                            }))
                                        }
                                        className="w-20"
                                    />
                                </div>
                            </div>

                            <div className="space-y-2 col-span-1 md:col-span-2 flex items-end">
                                <Button variant="outline" onClick={resetFilters}>
                                    Reset Filters
                                </Button>
                            </div>
                        </div>

                        {/* Table Container - flex-1 makes it take available space */}
                        <div className="flex-1 overflow-hidden">
                            {/* Instances Table */}
                            <div className="border rounded-md h-full overflow-hidden">
                                <div className="h-full overflow-y-auto">
                                    <Table>
                                        <TableHeader className="sticky top-0 bg-background z-10">
                                            <TableRow>
                                                <TableHead className="w-12">Select</TableHead>
                                                <TableHead>Instance ID</TableHead>
                                                <TableHead>Dataset</TableHead>
                                                <TableHead>Repository</TableHead>
                                                <TableHead>Files</TableHead>
                                                <TableHead>Resolved</TableHead>
                                                <TableHead className="w-12">Details</TableHead>
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {instancesLoading ? (
                                                <TableRow>
                                                    <TableCell colSpan={8} className="text-center py-6">
                                                        Loading instances...
                                                    </TableCell>
                                                </TableRow>
                                            ) : instancesError ? (
                                                <TableRow>
                                                    <TableCell colSpan={8} className="text-center py-6 text-destructive">
                                                        Error loading instances. Please try again.
                                                    </TableCell>
                                                </TableRow>
                                            ) : filteredInstances.length === 0 ? (
                                                <TableRow>
                                                    <TableCell colSpan={8} className="text-center py-6">
                                                        No instances match your filters.
                                                    </TableCell>
                                                </TableRow>
                                            ) : (
                                                filteredInstances.map((instance) => {
                                                    return (
                                                        <TableRow key={instance.instance_id}>
                                                            <TableCell>
                                                                <Checkbox
                                                                    checked={!!filterState.selectedInstances[instance.instance_id]}
                                                                    onCheckedChange={() => toggleInstanceSelection(instance.instance_id)}
                                                                />
                                                            </TableCell>
                                                            <TableCell>{instance.instance_id}</TableCell>
                                                            <TableCell>{instance.dataset}</TableCell>
                                                            <TableCell>{instance.repo}</TableCell>

                                                            <TableCell>
                                                                <div className="flex items-center">
                                                                    <FileIcon className="w-4 h-4 mr-1" />
                                                                    {instance.file_count}
                                                                </div>
                                                            </TableCell>
                                                            <TableCell>
                                                                <Badge variant={instance.resolved_count > 0 ? "default" : "secondary"}>
                                                                    {instance.resolved_count}
                                                                </Badge>
                                                            </TableCell>
                                                            <TableCell>
                                                                <Button
                                                                    variant="ghost"
                                                                    size="icon"
                                                                    onClick={() => showInstanceDetails(instance.instance_id)}
                                                                    title="View instance details"
                                                                >
                                                                    <InfoIcon className="w-4 h-4" />
                                                                </Button>
                                                            </TableCell>
                                                        </TableRow>
                                                    );
                                                })
                                            )}
                                        </TableBody>
                                    </Table>
                                </div>
                            </div>
                        </div>

                        {/* Pagination */}
                        <Pagination className="mt-4">
                            <PaginationContent>
                                <PaginationItem>
                                    <PaginationPrevious
                                        onClick={() =>
                                            setFilterState((prev) => ({
                                                ...prev,
                                                page: Math.max(1, prev.page - 1)
                                            }))
                                        }
                                        className={filterState.page <= 1 ? "pointer-events-none opacity-50" : ""}
                                    />
                                </PaginationItem>
                                <PaginationItem>
                                    <PaginationLink>
                                        Page {filterState.page}
                                    </PaginationLink>
                                </PaginationItem>
                                <PaginationItem>
                                    <PaginationNext
                                        onClick={() =>
                                            setFilterState((prev) => ({
                                                ...prev,
                                                page: prev.page + 1
                                            }))
                                        }
                                        className={
                                            !instancesResponse ||
                                                filteredInstances.length < filterState.limit
                                                ? "pointer-events-none opacity-50"
                                                : ""
                                        }
                                    />
                                </PaginationItem>
                            </PaginationContent>
                        </Pagination>
                    </div>

                    <DialogFooter className="flex justify-between items-center mt-4">
                        <div>
                            {Object.values(filterState.selectedInstances).filter(Boolean).length} instance(s) selected
                        </div>
                        <div>
                            <Button
                                variant="outline"
                                onClick={handleDone}
                                className="mr-2"
                            >
                                Done
                            </Button>
                        </div>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Instance details dialog */}
            <InstanceDetailsDialog
                open={detailsDialogOpen}
                onOpenChange={setDetailsDialogOpen}
                instanceId={selectedInstanceId}
            />
        </>
    );
} 