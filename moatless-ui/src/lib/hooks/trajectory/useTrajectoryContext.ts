import { createContext, useContext } from 'react';
import { Trajectory } from '@/lib/types/trajectory';

interface TrajectoryContextType {
  trajectory: Trajectory;
}

const TrajectoryContext = createContext<TrajectoryContextType | null>(null);

export function useTrajectoryContext() {
  const context = useContext(TrajectoryContext);
  if (!context) {
    throw new Error('useTrajectoryContext must be used within a TrajectoryProvider');
  }
  return context;
}

export const TrajectoryProvider = TrajectoryContext.Provider; 