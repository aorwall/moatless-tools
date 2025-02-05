import { useSearchParams } from "react-router-dom";
import { TrajectoryHeader } from "@/pages/trajectory/components/TrajectoryHeader";
import { TrajectoryUpload } from "@/pages/trajectory/components/TrajectoryUpload";
import { TrajectoryViewer } from "@/lib/components/trajectory/TrajectoryViewer";

export function Trajectory() {
  const [searchParams, setSearchParams] = useSearchParams();
  const path = searchParams.get("path");

  const handleLoadTrajectory = (path: string) => {
    setSearchParams({ path });
  };

  return (
    <div className="flex h-screen flex-col">
      <div className="flex-none p-6 bg-white border-b">
        <TrajectoryHeader />
        <TrajectoryUpload
          onLoadTrajectory={handleLoadTrajectory}
          searchParams={searchParams}
          setSearchParams={setSearchParams}
        />
      </div>
      {path && (
        <div className="flex-1 overflow-auto">
          <div className="container mx-auto py-6">
            <TrajectoryViewer trajectoryId={path} />
          </div>
        </div>
      )}
    </div>
  );
}
