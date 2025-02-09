export interface RunStatus {
  run_id: string;
  status: "pending" | "running" | "finished" | "error";
  iterations: number;
  cost: number;
  current_action: string | null;
  error: string | null;
  result: any | null;
}
