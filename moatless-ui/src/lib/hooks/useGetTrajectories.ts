import { useQuery } from "@tanstack/react-query";
import { trajectoriesApi } from "@/lib/api/trajectories";

export function useGetTrajectories() {
  const {
    data: trajectories,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["trajectories"],
    queryFn: trajectoriesApi.getTrajectories,
  });

  const sortedTrajectories = trajectories
    ? [...trajectories].sort((a, b) => {
        const aDate = a.finished_at
          ? new Date(a.finished_at)
          : new Date(a.started_at);
        const bDate = b.finished_at
          ? new Date(b.finished_at)
          : new Date(b.started_at);
        return bDate.getTime() - aDate.getTime();
      })
    : [];

  return {
    trajectories: sortedTrajectories,
    isLoading,
    error,
  };
}
