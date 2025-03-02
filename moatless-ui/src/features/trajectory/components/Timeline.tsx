import { Node, Trajectory } from "@/lib/types/trajectory.ts";
import { TimelineItem } from "@/features/trajectory/components/TimelineItem.tsx";
import { TrajectoryNode } from "@/features/trajectory/components/TrajectoryNode.tsx";
import { NodeCircle } from '@/features/trajectory/components/NodeCircle.tsx';
import { UserMessageItem } from '@/features/trajectory/components/UserMessageItem.tsx';
import { cn } from "@/lib/utils.ts";
import { useTrajectoryStore } from "@/features/trajectory/stores/trajectoryStore.ts";
import './timeline.css';

// Feature flags configuration - kept in JS since it's used for logic
export const TIMELINE_CONFIG = {
  features: {
    enableConnectionLines: false, // Feature flag for connection lines
  }
} as const;

// Helper function to get CSS variable values
const getTimelineVar = (name: string): string => {
  return `var(--timeline-${name})`;
};

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
  // If connection lines are disabled, return null
  if (!TIMELINE_CONFIG.features.enableConnectionLines) {
    return null;
  }
  
  return (
    <>
      {/* Main vertical line for current node */}
      {!isLastInLevel && (
        <div 
          className="absolute w-px bg-gray-200"
          style={{
            left: level === 0 ? getTimelineVar('item-offset') : `calc(${getTimelineVar('offset-default')} + ${getTimelineVar('item-offset')})`,
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
            right: `calc(${getTimelineVar('vertical-line-parent-offset')} + ${parent.level * 48}px)`,
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
            top: getTimelineVar('vertical-offset'),
            width: getTimelineVar('horizontal-connector')
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

    // Special handling for the initial user message node
    const isUserMessageNode = node.nodeId === 0 && node.userMessage;

    return (
      <li key={node.nodeId} className={cn(
        isUserMessageNode ? "mb-8" : "mb-4 sm:mb-8",
        "relative",
        { [`ml-[${getTimelineVar('node-padding-default')}] sm:ml-[${getTimelineVar('node-padding-sm')}]`]: level > 0 }
      )}>
        <div className={cn(
          "group relative", 
          {
            "ml-[24px] sm:ml-[48px]": level > 0,
            "ml-0": isUserMessageNode
          }
        )}>
          {/* Node header */}
          <div className="relative rounded-lg">
            <div className={cn(
              "flex items-start",
              isUserMessageNode && "ml-0"
            )}>
              {/* Node circle with connection lines */}
              <div className={cn(
                "relative",
                isUserMessageNode && "hidden"
              )}>
                <ConnectionLines 
                  level={level}
                  hasChildren={hasChildren}
                  isLastInLevel={isLastNode}
                  parentNodes={currentParentNodes}
                />

                {!isUserMessageNode && (
                  <NodeCircle
                    trajectory={trajectory}
                    node={node}
                    isLastNode={isLastNode && !hasChildren}
                    isRunning={isRunning}
                    onClick={() => handleNodeClick(node.nodeId)}
                  />
                )}
              </div>

              {/* Node content */}
              <div className={cn(
                "min-w-0 flex-1",
                isUserMessageNode ? "pl-0" : "pl-8"
              )}>
                <div className="flex items-start justify-between">
                  <div className={cn(
                    "min-w-0 flex-1",
                    isUserMessageNode && "w-full"
                  )}>
                    {isUserMessageNode ? (
                      <UserMessageItem message={node.userMessage!} />
                    ) : (
                      <TrajectoryNode
                        node={node}
                        expanded={isExpanded(node.nodeId)}
                        level={level}
                      />
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Timeline items - only show for non-user-message nodes */}
          {!isUserMessageNode && isExpanded(node.nodeId) && (
            <div className={cn(
              "mt-8 transition-all duration-200 ease-in-out",
              "relative pl-1 pr-4 py-4",
              `ml-[${getTimelineVar('items-margin-left')}] mr-4`,
              "border border-gray-200 bg-gray-50/50 rounded-lg",
              "shadow-sm",
              "animate-in fade-in slide-in-from-left-1",
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
  const { isNodeExpanded, toggleNode } = useTrajectoryStore();
  
  const isExpanded = (nodeId: number) => isNodeExpanded(trajectory.id, nodeId);
  const handleNodeClick = (nodeId: number) => {
    toggleNode(trajectory.id, nodeId);
  };

  return (
    <div className="w-full">
      <div className="relative">
        <ol className="relative">
          {renderNodes(trajectory, trajectory.nodes, 0, isRunning, isExpanded, handleNodeClick)}

        </ol>
      </div>
    </div>
  );
}
