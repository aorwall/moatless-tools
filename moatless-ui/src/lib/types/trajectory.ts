export interface ResumeTrajectoryRequest {
  agent_id: string;
  model_id: string;
  message: string;
}

export interface TimelineItem {
  label: string;
  icon: string;  // Now a string matching lucide-react icon names
  type: TimelineItemType;
  content: TimelineContent;
}

export type TimelineItemType =
  | "user_message"
  | "assistant_message"
  | "completion"
  | "thought"
  | "action"
  | "observation"
  | "error"
  | "workspace"
  | "workspace_files"
  | "workspace_context"
  | "workspace_tests"
  | "artifact"
  | "reward";

export type TimelineContent = 
  | UserMessageContent
  | AssistantMessageContent
  | CompletionContent
  | ThoughtContent
  | ActionContent
  | ObservationContent
  | ErrorContent
  | WorkspaceContent
  | ArtifactChangeContent;

export interface UserMessageContent {
  message: string;
}

export interface AssistantMessageContent {
  message: string;
}

export interface CompletionContent {
  usage?: Usage;
  tokens: string;
  retries?: number;
  input?: string;
  response?: string;
  warnings: string[];
  errors: string[];
}

export interface ThoughtContent {
  message: string;
}

export interface ActionContent {
  name: string;
  shortSummary: string;
  thoughts?: string;
  properties: Record<string, any>;
}

export interface ObservationContent {
  message?: string;
  summary?: string;
  properties: Record<string, any>;
  expectCorrection: boolean;
}

export interface ErrorContent {
  error: string;
}

export interface WorkspaceContent {
  summary: string;
  testResults?: Record<string, any>[];
}

export interface Usage {
  completionCost?: number;
  promptTokens?: number;
  completionTokens?: number;
  cachedTokens?: number;
}

export interface Completion {
  type: string;
  usage?: Usage;
  tokens: string;
  retries?: number;
  input?: string;
  response?: string;
}

export interface Observation {
  message?: string;
  summary?: string;
  properties: Record<string, any>;
  expectCorrection: boolean;
}

export interface Action {
  name: string;
  shortSummary: string;
  thoughts?: string;
  properties: Record<string, any>;
}

export interface ActionStep {
  thoughts?: string;
  action: Action;
  observation?: Observation;
  completion?: Completion;
  warnings: string[];
  errors: string[];
}

export interface FileContextSpan {
  span_id: string;
  start_line?: number;
  end_line?: number;
  tokens?: number;
  pinned: boolean;
}

export interface FileContextFile {
  file_path: string;
  content?: string;
  patch?: string;
  spans: FileContextSpan[];
  show_all_spans: boolean;
  tokens?: number;
  is_new: boolean;
  was_edited: boolean;
}

export interface UpdatedFile {
  file_path: string;
  status: string;
  tokens?: number;
  patch?: string;
}

export interface FileContext {
  summary: string;
  testResults?: Record<string, any>[];
  patch?: string;
  files: FileContextFile[];
  warnings: string[];
  errors: string[];
  updatedFiles: UpdatedFile[];
}

export interface TestResultsSummary {
  total: number;
  passed: number;
  failed: number;
  errors: number;
  skipped: number;
}

export interface Reward {
  value: number;
  explanation?: string;
}

export interface Node {
  nodeId: number;
  reward?: Reward;
  children: Node[];
  executed: boolean;
  userMessage?: string;
  assistantMessage?: string;
  actionSteps: ActionStep[];
  error?: string;
  fileContext?: FileContext;
  warnings: string[];
  errors: string[];
  terminal: boolean;
  allNodeErrors: string[];
  allNodeWarnings: string[];
  testResultsSummary?: TestResultsSummary;
  items: TimelineItem[];
}


export interface ArtifactChangeContent {
  artifact_id: string;
  artifact_type: string;
  change_type: "added" | "updated" | "removed";
  diff_details?: string;
  actor: "user" | "assistant";
}


export interface TrajectoryEvent {
  timestamp: number;
  scope?: string;
  event_type: string;
  node_id?: number;
  agent_id?: string;
  action_name?: string;
  data?: Record<string, any>;
}

export interface TrajectoryStatus {
  status: string;
  error?: string;
  error_trace?: string;
  started_at: string;
  finished_at?: string;
  restart_count: number;
  last_restart?: string;
  metadata: Record<string, any>;
  current_attempt?: number;
}

export interface Trajectory {
  id: string;
  project_id: string;
  status: "running" | "error" | "finished" | "unknown";
  agent_id: string;
  model_id: string;
  system_status: TrajectoryStatus;
  duration?: number;
  error?: string;
  completionCost?: number;
  promptTokens?: number;
  completionTokens?: number;
  cachedTokens?: number;
  totalTokens?: number;
  flags?: string[];
  failedActions?: number;
  duplicatedActions?: number;
  nodes: Node[];
  events: TrajectoryEvent[];
}
