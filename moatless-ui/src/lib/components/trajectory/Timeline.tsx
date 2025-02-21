import { Node, Trajectory } from "@/lib/types/trajectory";
import { TimelineItem } from "@/lib/components/trajectory/TimelineItem";
import { TrajectoryNode } from "@/lib/components/trajectory/TrajectoryNode";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { NodeCircle } from '@/lib/components/trajectory/NodeCircle';
import { cn } from "@/lib/utils";
import { useTrajectoryActions } from "@/pages/trajectory/stores/trajectoryStore";
import { ChevronDown } from "lucide-react";

export const TIMELINE_CONFIG = {
  // Horizontal spacing
  nodePadding: {
    sm: '12px',  // For wider screens
    default: '12px' // For mobile
  },
  // Vertical spacing between nodes
  nodeSpacing: {
    sm: '32px', // 8rem
    default: '16px' // 4rem
  },
  // Left offset for the main timeline
  timelineOffset: {
    sm: '150px',
    default: '80px'
  },
  // New configuration for lines
  lines: {
    circleSize: '40px',
    verticalOffset: '16px',
    horizontalConnector: '24px',
    verticalLine: {
      offset: '20px',  // Change from 50% to fixed pixel value
      parentOffset: '48px',
    },
    item: {
      offset: '20px', // Aligns with the circle's center
      verticalOffset: '24px'
    }
  }
} as const;

interface TimelineProps {
  trajectory: Trajectory;
  isRunning: boolean;
}

// First create a new component for the lines
interface ConnectionLinesProps {
  level: number;
  hasChildren: boolean;
  isLastInLevel: boolean;
  parentNodes: Array<{
    hasNextSibling: boolean;
    level: number;
  }>;
}

const ConnectionLines = ({ level, hasChildren, isLastInLevel, parentNodes }: ConnectionLinesProps) => {
  return (
    <>
      {/* Main vertical line for current node */}
      {!isLastInLevel && (
        <div 
          className="absolute w-px bg-gray-200"
          style={{
            left: level === 0 ? '20px' : `calc(${TIMELINE_CONFIG.timelineOffset.default} + 20px)`,
            top: '0',
            height: '100%'
          }}
        />
      )}

      {/* Parent connection lines - only for non-root levels */}
      {level > 0 && parentNodes.map((parent, index) => (
        <div 
          key={index}
          className="absolute w-px bg-gray-200"
          style={{
            right: `calc(${TIMELINE_CONFIG.lines.verticalLine.parentOffset} + ${parent.level * 48}px)`,
            top: '-32px',
            bottom: '0'
          }}
        />
      ))}

      {/* Horizontal connector - only for non-root levels */}
      {level > 0 && (
        <div 
          className="absolute h-px bg-gray-200"
          style={{
            right: '100%',
            top: TIMELINE_CONFIG.lines.verticalOffset,
            width: TIMELINE_CONFIG.lines.horizontalConnector
          }}
        />
      )}
    </>
  );
};

function renderNodes(
  trajectory: Trajectory,
  nodes: Node[], 
  level: number = 0, 
  isRunning: boolean,
  isExpanded: (nodeId: number) => boolean,
  handleNodeClick: (nodeId: number) => void,
  parentNodes: Array<{hasNextSibling: boolean, level: number}> = []
) {
  return nodes.map((node, index) => {
    const isLastNode = index === nodes.length - 1;
    const hasChildren = node.children && node.children.length > 0;
    
    // Only include parent nodes for non-root levels
    const currentParentNodes = level > 0 ? [
      ...parentNodes,
      { hasNextSibling: !isLastNode, level: level - 1 } // Adjust level to start from 0
    ] : [];

    return (
      <li key={node.nodeId} className={cn(
        "mb-4 sm:mb-8 relative",
        { [`ml-[${TIMELINE_CONFIG.nodePadding.default}] sm:ml-[${TIMELINE_CONFIG.nodePadding.sm}]`]: level > 0 }
      )}>
        <div className={cn("group relative", {
          "ml-[24px] sm:ml-[48px]": level > 0
        })}>
          {/* Node header */}
          <div className="relative rounded-lg">
            <div className="flex items-start">
              {/* Step number */}
              <div className="flex w-[80px] shrink-0 items-start justify-end sm:w-[150px]">
                <div className="mr-3 flex h-8 flex-col justify-center text-right sm:mr-6">
                  <button
                    className={cn("max-w-[60px] cursor-pointer truncate text-xs font-medium sm:max-w-[120px]", {
                      "text-gray-600 group-hover:text-gray-900": !hasChildren,
                      "text-primary-600 group-hover:text-primary-900": hasChildren,
                    })}
                    onClick={() => handleNodeClick(node.nodeId)}
                  >
                  </button>
                </div>
              </div>

              {/* Node circle with connection lines */}
              <div className="relative">
                <ConnectionLines 
                  level={level}
                  hasChildren={hasChildren}
                  isLastInLevel={isLastNode}
                  parentNodes={currentParentNodes}
                />

                <NodeCircle
                  node={node}
                  isLastNode={isLastNode && !hasChildren}
                  isRunning={isRunning}
                  onClick={() => handleNodeClick(node.nodeId)}
                />
              </div>

              {/* Node content */}
              <div className="min-w-0 flex-1 pl-8">
                <div className="flex items-start justify-between">
                  <div className="min-w-0 flex-1">
                    <TrajectoryNode
                      node={node}
                      expanded={isExpanded(node.nodeId)}
                      level={level}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Timeline items */}
          {isExpanded(node.nodeId) && (
            <div className={cn(
              "mt-8 transition-all duration-200 ease-in-out",
              "relative pl-6 pr-4 py-4",
              "ml-[150px] mr-4",
              "border border-gray-200 bg-gray-50/50 rounded-lg",
              "shadow-sm",
              "animate-in fade-in slide-in-from-left-1",
              "before:absolute before:-left-[1px] before:top-4 before:h-px before:w-6 before:bg-gray-200",
              "after:absolute after:-left-2 after:top-[14px] after:h-3 after:w-3 after:rotate-45 after:border-l after:border-t after:border-gray-200 after:bg-gray-50/50"
            )}>
              {node.items.map((item, index) => (
                <TimelineItem
                  key={index}
                  type={item.type}
                  content={item.content}
                  label={item.label}
                  nodeId={node.nodeId}
                  instanceId={trajectory.id}
                  itemId={index.toString()}
                  isLast={index === node.items.length - 1}
                  hasNextSibling={!isLastNode || hasChildren}
                />
              ))}
            </div>
          )}

          {/* Child nodes */}
          {hasChildren && (
            <div className="relative mt-8">
              <ol className="relative">
                {renderNodes(
                  trajectory,
                  node.children, 
                  level + 1, 
                  isRunning,
                  isExpanded,
                  handleNodeClick,
                  currentParentNodes
                )}
              </ol>
            </div>
          )}
        </div>
      </li>
    );
  });
}


export function Timeline({ trajectory, isRunning = false }: TimelineProps) {
  const { isNodeExpanded, toggleNode } = useTrajectoryActions();
  
  const lastNode = trajectory.nodes[trajectory.nodes.length - 1];
  const isTerminal = lastNode?.terminal;

  const isExpanded = (nodeId: number) => isNodeExpanded(nodeId);
  const handleNodeClick = (nodeId: number) => {
    toggleNode(nodeId);
  };

  return (
    <div className="w-full">
      <div className="relative">
        <ol className="relative">
          {renderNodes(trajectory, trajectory.nodes, 0, isRunning, isExpanded, handleNodeClick)}

          {/* Terminal node */}
          {isTerminal && !isRunning && (
            <li className="mb-4 sm:mb-8">
              <div className="flex items-start">
                <div className="flex w-[80px] shrink-0 items-start justify-end sm:w-[150px]">
                  <div className="mr-3 flex h-8 flex-col justify-center text-right sm:mr-6">
                    <div>
                      <span className="text-xs font-medium text-gray-600">
                        End
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div
                className={`relative z-10 -ml-4 flex h-8 min-w-[2rem] items-center justify-center rounded-full border-2 bg-white 
                  ${
                    lastNode.error
                      ? "border-red-500 bg-red-500"
                      : "border-green-500 bg-green-500"
                  }`}
              />

              <div className="min-w-0 flex-1 pl-8">
                <div className="text-xs text-gray-600">
                  {lastNode.error
                    ? "Terminated with error"
                    : "Successfully completed"}
                </div>
              </div>
            </li>
          )}
        </ol>
      </div>
    </div>
  );
}
