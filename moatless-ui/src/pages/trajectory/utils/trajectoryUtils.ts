import { Node, TimelineItem, TestResultsSummary, WorkspaceTimelineContent } from '../types';
import { MessageSquare, Bot, Lightbulb, Terminal, Eye, Folder, AlertTriangle, Cpu } from 'lucide-react';

export function getTestResultsSummary(
  testResults: Array<{ status: string; name?: string; message?: string }> = []
): TestResultsSummary {
  return {
    total: testResults.length,
    passed: testResults.filter((t) => t.status === 'passed').length,
    failed: testResults.filter((t) => t.status === 'failed').length,
    errors: testResults.filter((t) => t.status === 'error').length,
    skipped: testResults.filter((t) => t.status === 'skipped').length
  };
}

export function getTimelineItems(node: Node): TimelineItem[] {
  const items: TimelineItem[] = [];

  if (node.userMessage) {
    items.push({
      label: 'User Message',
      icon: MessageSquare,
      type: 'user_message',
      content: { message: node.userMessage }
    });
  }

  if (node.assistantMessage) {
    items.push({
      label: 'Assistant Message',
      icon: Bot,
      type: 'assistant_message',
      content: { message: node.assistantMessage }
    });
  }

  if (node.actionCompletion) {
    items.push({
      label: 'Completion',
      icon: Cpu,
      type: 'completion',
      content: {
        usage: node.actionCompletion.usage,
        input: node.actionCompletion.input,
        response: node.actionCompletion.response
      }
    });
  }

  node.actionSteps?.forEach((step) => {
    if (step.thoughts) {
      items.push({
        label: 'Thought',
        icon: Lightbulb,
        type: 'thought',
        content: { message: step.thoughts }
      });
    }

    items.push({
      label: step.action.name,
      icon: Terminal,
      type: 'action',
      content: {
        ...step.action,
        errors: step.errors,
        warnings: step.warnings
      }
    });

    if (step.completion) {
      items.push({
        label: 'Action Completion',
        icon: Cpu,
        type: 'completion',
        content: {
          usage: step.completion.usage,
          input: step.completion.input,
          response: step.completion.response
        }
      });
    }

    if (step.observation) {
      items.push({
        label: 'Observation',
        icon: Eye,
        type: 'observation',
        content: step.observation
      });
    }
  });

  if (node.error) {
    items.push({
      label: 'Error',
      icon: AlertTriangle,
      type: 'error',
      content: { error: node.error }
    });
  } else if (node.fileContext) {
    const testResults = node.fileContext.testResults as
      | Array<{ status: string; name?: string; message?: string }>
      | undefined;
    const content: WorkspaceTimelineContent = {
      ...node.fileContext,
      testResults,
      testResultsSummary: testResults ? getTestResultsSummary(testResults) : undefined
    };
    items.push({
      label: 'Workspace',
      icon: Folder,
      type: 'workspace',
      content
    });
  }

  return items;
}

export function hasSuccessfulChanges(node: Node): boolean {
  const hasPatches = node.fileContext?.files?.some((f) => f.patch) ?? false;
  const hasTestErrors =
    node.fileContext?.testResults?.some((t) => t.status === 'failed' || t.status === 'error') ??
    false;
  return hasPatches && !hasTestErrors;
}

export function getNodeColor(node: Node): string {
  if (node.nodeId === 0) return 'blue';
  if (node.allNodeErrors.length > 0) return 'red';
  if (node.allNodeWarnings.length > 0) return 'yellow';

  const lastAction = node.actionSteps[node.actionSteps.length - 1]?.action.name;
  if (lastAction === 'Finish') {
    return hasSuccessfulChanges(node) ? 'green' : 'red';
  }

  if (node.fileContext?.updatedFiles?.length) return 'green';
  return 'gray';
} 