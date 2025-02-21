import { createContext, useContext, PropsWithChildren } from 'react';
import { Trajectory } from '@/lib/types/trajectory';

interface TrajectoryContextValue {
  trajectory: Trajectory;
  trajectoryId: string;
}

const TrajectoryContext = createContext<TrajectoryContextValue | null>(null);

interface TrajectoryProviderProps {
  trajectory: Trajectory;
}

export function TrajectoryProvider({ 
  trajectory, 
  children 
}: PropsWithChildren<TrajectoryProviderProps>) {
  if (!trajectory?.id) {
    throw new Error('TrajectoryProvider requires a trajectory with an id');
  }

  const value: TrajectoryContextValue = {
    trajectory,
    trajectoryId: trajectory.id
  };

  return (
    <TrajectoryContext.Provider value={value}>
      {children}
    </TrajectoryContext.Provider>
  );
}

export function useTrajectoryContext() {
  const context = useContext(TrajectoryContext);
  if (!context) {
    throw new Error('useTrajectoryContext must be used within a TrajectoryProvider');
  }
  return context;
}

export function useTrajectoryId() {
  const context = useTrajectoryContext();
  if (!context.trajectoryId) {
    throw new Error('No trajectory ID found in context');
  }
  return { trajectoryId: context.trajectoryId };
}

export function useTrajectory() {
  const context = useTrajectoryContext();
  if (!context.trajectory) {
    throw new Error('No trajectory found in context');
  }
  return { trajectory: context.trajectory };
} 