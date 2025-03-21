import { apiRequest } from "@/lib/api/config";

export interface SWEBenchInstance {
    instance_id: string;
    repo: string;
    dataset: string;
    problem_statement?: string;
    resolved_count: number;
    file_count: number;

}

export interface FullSWEBenchInstance extends SWEBenchInstance {
    golden_patch?: string;
    test_patch?: string;
    expected_spans?: Record<string, any>;
    test_file_spans?: Record<string, any>;
    base_commit?: string;
    fail_to_pass?: string[];
    pass_to_pass?: string[];
    resolved_by?: Array<{
        name: string;
        updated_spans: Record<string, string[]>;
        alternative_spans?: Record<string, string[]>;
    }>;
}

export interface InstancesResponse {
    instances: SWEBenchInstance[];
}

export interface InstancesQueryParams {
    page?: number;
    limit?: number;
    sort_by?: string;
    order?: 'asc' | 'desc';
    search?: string;
    dataset?: string;
    repo?: string;
    min_resolved?: number;
    max_resolved?: number;
    min_files?: number;
    max_files?: number;
}

export const instancesApi = {
    getInstances: (params: InstancesQueryParams = {}) => {
        const queryParams = new URLSearchParams();

        if (params.page) queryParams.append('page', params.page.toString());
        if (params.limit) queryParams.append('limit', params.limit.toString());
        if (params.sort_by) queryParams.append('sort_by', params.sort_by);
        if (params.order) queryParams.append('order', params.order);
        if (params.search) queryParams.append('search', params.search);
        if (params.dataset) queryParams.append('dataset', params.dataset);
        if (params.repo) queryParams.append('repo', params.repo);
        if (params.min_resolved !== undefined) queryParams.append('min_resolved', params.min_resolved.toString());
        if (params.max_resolved !== undefined) queryParams.append('max_resolved', params.max_resolved.toString());
        if (params.min_files !== undefined) queryParams.append('min_files', params.min_files.toString());
        if (params.max_files !== undefined) queryParams.append('max_files', params.max_files.toString());

        const queryString = queryParams.toString();
        const url = `/swebench/instances${queryString ? `?${queryString}` : ''}`;

        return apiRequest<InstancesResponse>(url);
    },

    getInstance: (instanceId: string) => {
        return apiRequest<FullSWEBenchInstance>(`/swebench/instances/${instanceId}`);
    }
}; 