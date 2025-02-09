import { Circle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Node } from '@/lib/types/trajectory';

interface NodeCircleProps {
  node: Node;
  isLastNode: boolean;
  isRunning: boolean;
  onClick: () => void;
}

const COLOR_MAPPINGS = {
  blue: {
    border: 'border-blue-500',
    hover: 'hover:border-blue-600 hover:bg-blue-50',
    text: 'text-blue-500',
    hoverText: 'group-hover:text-blue-600',
  },
  red: {
    border: 'border-red-500',
    hover: 'hover:border-red-600 hover:bg-red-50',
    text: 'text-red-500',
    hoverText: 'group-hover:text-red-600',
  },
  yellow: {
    border: 'border-yellow-500',
    hover: 'hover:border-yellow-600 hover:bg-yellow-50',
    text: 'text-yellow-500',
    hoverText: 'group-hover:text-yellow-600',
  },
  green: {
    border: 'border-green-500',
    hover: 'hover:border-green-600 hover:bg-green-50',
    text: 'text-green-500',
    hoverText: 'group-hover:text-green-600',
  },
  default: {
    border: 'border-gray-300',
    hover: 'hover:border-gray-300 hover:bg-gray-50',
    text: 'text-gray-300',
    hoverText: 'group-hover:text-gray-500',
  },
} as const;


function getNodeColor(node: Node, isRunning: boolean): string {
    if (node.nodeId === 0) return "blue";
    if (node.error) return "red";
    if (node.allNodeErrors.length > 0) return "red";
    if (node.allNodeWarnings.length > 0) return "yellow";
    if (node.executed) return "green";
    return "gray";
  }
  
export function NodeCircle({ node, isLastNode, isRunning, onClick }: NodeCircleProps) {
  const nodeColor = getNodeColor(node, isRunning);
  const colors = COLOR_MAPPINGS[nodeColor as keyof typeof COLOR_MAPPINGS] || COLOR_MAPPINGS.default;

  const showSpinner = node.nodeId !== 0 && isRunning && isLastNode;

  return (
    <button
      className={cn(
        'relative z-10 flex h-8 min-w-[2rem] cursor-pointer items-center justify-center',
        'rounded-full border-2 bg-white transition-colors duration-150 -ml-4 -mr-4',
        colors.border,
        colors.hover
      )}
      onClick={onClick}
    >
      {node.nodeId !== 0 && (
        showSpinner ? (
          <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
        ) : (
          <Circle
            className={cn(
              'h-4 w-4 transition-colors duration-150',
              colors.text,
              colors.hoverText
            )}
          />
        )
      )}
    </button>
  );
} 