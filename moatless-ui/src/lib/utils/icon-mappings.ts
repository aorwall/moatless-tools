import {
  MessageSquare,
  Bot,
  Lightbulb,
  Terminal,
  Eye,
  Folder,
  AlertTriangle,
  Cpu,
  FileEdit,
  type LucideIcon
} from "lucide-react";

export const Icons: Record<string, LucideIcon> = {
  // Timeline item types
  user_message: MessageSquare,
  assistant_message: Bot,
  thought: Lightbulb,
  action: Terminal,
  observation: Eye,
  workspace: Folder,
  error: AlertTriangle,
  completion: Cpu,
  artifact: FileEdit,
};

export type IconType = keyof typeof Icons; 