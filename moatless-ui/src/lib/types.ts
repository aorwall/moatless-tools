export interface SystemStatus {
  run_id: string;
  started_at: string;
  finished_at?: string;
  status: string;
  error?: string;
  error_trace?: string;
  metadata: Record<string, any>;
  restart_count: number;
  last_restart?: string;
}

export interface TrajectoryListItem extends SystemStatus {
  trajectory_id: string;
} 