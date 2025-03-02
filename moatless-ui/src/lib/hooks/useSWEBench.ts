import { useQuery, useMutation } from "@tanstack/react-query";
import { swebenchApi } from "@/lib/api/swebench";
import { AttachmentData } from "@/types/attachments";

export const swebenchKeys = {
  all: ["swebench"] as const,
  instances: () => [...swebenchKeys.all, "instances"] as const,
  validations: () => [...swebenchKeys.all, "validations"] as const,
};

export function useSWEBenchInstances(
  page: number,
  limit: number,
  sortBy: string = "instance_id",
  order: string = "asc",
) {
  return useQuery({
    queryKey: [swebenchKeys.instances(), page, limit, sortBy, order],
    queryFn: () => swebenchApi.getInstances(page, limit, sortBy, order),
  });
}

export function useStartValidation() {
  return useMutation({
    mutationFn: swebenchApi.startValidation,
  });
}

export function useStartLoop() {
  return useMutation({
    mutationFn: (data: {
      agent_id: string;
      model_id: string;
      message: string;
      attachments?: AttachmentData[];
      repository_path?: string;
    }) => swebenchApi.startLoop(data),
  });
}
