import { TimelineItem as TrajectoryTimelineItem } from "@/lib/types/trajectory";

export interface MessageContent {
  message: string;
}

export interface ActionContent {
  [key: string]: any; // For dynamic properties
  errors?: string[];
  warnings?: string[];
}

export interface ArtifactContent {
  artifact_id: string;
  artifact_type: string;
  change_type: string;
  diff_details: any;
  actor: string;
}

export type TimelineItemType =
  | "user_message"
  | "assistant_message"
  | "thought"
  | "action"
  | "artifact";

export type ChatMessage = TrajectoryTimelineItem & {
  nodeId: number;
  timestamp?: string;
  id: string;
  trajectoryId: string;
};

export interface ChatMessageGroup {
  messages: ChatMessage[];
  isUser: boolean;
}
