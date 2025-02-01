import type { LucideIcon } from 'lucide-react';

// Timeline specific types
export interface TimelineItem {
  label: string;
  icon: LucideIcon;
  type: 'user_message' | 'assistant_message' | 'completion' | 'thought' | 'action' | 'observation' | 'error' | 'workspace';
  content: any; // This could be further typed based on the type property
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

export interface Node {
  nodeId: number;
  userMessage?: string;
  assistantMessage?: string;
  actionCompletion?: Completion;
  actionSteps: ActionStep[];
  fileContext?: FileContext;
  error?: string;
  warnings: string[];
  errors: string[];
  terminal: boolean;
  allNodeErrors: string[];
  allNodeWarnings: string[];
  testResultsSummary?: TestResultsSummary;
}

export interface TrajectoryDTO {
  duration?: number;
  error?: string;
  iterations?: number;
  completionCost?: number;
  totalTokens?: number;
  promptTokens?: number;
  completionTokens?: number;
  cachedTokens?: number;
  flags: string[];
  failedActions: number;
  duplicatedActions: number;
  nodes: Node[];
} 

