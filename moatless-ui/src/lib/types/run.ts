import { TrajectoryDTO } from "./trajectory";

export interface RunEvent {
  timestamp: number;
  event_type: string;
  node_id?: number;
  agent_id?: string;
  action_name?: string;
}

export interface RunStatus {
  status: string;
  error?: string;
  started_at: string;
  finished_at?: string;
  restart_count: number;
  last_restart?: string;
  metadata: Record<string, any>;
  current_attempt?: number;
}

export interface Run {
  id: string;
  status: "running" | "error" | "finished" | "unknown";
  system_status: RunStatus;
  trajectory?: TrajectoryDTO;
  events: RunEvent[];
}
