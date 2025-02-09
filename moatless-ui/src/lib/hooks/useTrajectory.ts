import { useQuery, useMutation } from "@tanstack/react-query";
import { trajectoryApi } from "@/lib/api/trajectory";

export const trajectoryKeys = {
  all: ["trajectory"] as const,
  byPath: (path: string) => [...trajectoryKeys.all, path] as const,
};

export const useTrajectory = (path: string | null) => {
  return useQuery({
    queryKey: path ? trajectoryKeys.byPath(path) : trajectoryKeys.all,
    queryFn: () => {
      if (!path) throw new Error("Path is required");
      return trajectoryApi.getTrajectory(path);
    },
    enabled: !!path,
  });
};

export const useTrajectoryUpload = () => {
  return useMutation({
    mutationFn: trajectoryApi.uploadTrajectory,
  });
};
