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
  Award,
  Files,
  Code,
  TestTube,
  type LucideIcon,
} from "lucide-react";

export const Icons: Record<string, LucideIcon> = {
  // Timeline item types
  user_message: MessageSquare,
  assistant_message: Bot,
  thought: Lightbulb,
  action: Terminal,
  observation: Eye,
  workspace: Folder,
  workspace_files: Files,
  workspace_context: Code,
  workspace_tests: TestTube,
  error: AlertTriangle,
  completion: Cpu,
  artifact: FileEdit,
  reward: Award,
};

export type IconType = keyof typeof Icons;
