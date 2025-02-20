import { Node } from "@/lib/types/trajectory";
import { TimelineItem } from "@/lib/components/trajectory/TimelineItem";
import { TrajectoryNode } from "@/lib/components/trajectory/TrajectoryNode";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { NodeCircle } from '@/lib/components/trajectory/NodeCircle';
import { cn } from "@/lib/utils";

interface TimelineProps {
  nodes: Node[];
  instanceId?: string;
  isRunning: boolean;
}

function getAllDescendants(node: Node): number {
  let count = 0;
  const processNode = (n: Node) => {
    if (n.children) {
      n.children.forEach(child => {
        count++;
        processNode(child);
      });
    }
  };
  processNode(node);
  return count;
}

function renderNodes(
  nodes: Node[], 
  level: number = 0, 
  instanceId: string, 
  isRunning: boolean,
  isExpanded: (nodeId: number) => boolean,
  handleNodeClick: (nodeId: number) => void
) {
  return nodes.map((node, index) => {
    const isLastNode = index === nodes.length - 1;
    const hasChildren = node.children && node.children.length > 0;
    const descendantCount = getAllDescendants(node);
    
    return (
      <li key={node.nodeId} className="mb-4 sm:mb-8">
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
                {/* Parent's vertical line - extends down to last descendant */}
                {hasChildren && (
                  <div 
                    className="absolute left-1/2 top-[16px] w-px -translate-x-1/2 bg-gray-200"
                    style={{ 
                      height: `calc(100% + ${descendantCount * 48}px)` 
                    }}
                  />
                )}

                {/* Horizontal line connecting to parent's vertical line */}
                {level > 0 && (
                  <div 
                    className="absolute right-full top-[16px] h-px w-[24px] bg-gray-200"
                  />
                )}

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
            <div className="mt-8 transition-all duration-200 ease-in-out">
              {node.items.map((item, index) => (
                <TimelineItem
                  key={index}
                  type={item.type}
                  content={item.content}
                  label={item.label}
                  nodeId={node.nodeId}
                  instanceId={instanceId}
                  itemId={index.toString()}
                />
              ))}
            </div>
          )}

          {/* Child nodes */}
          {hasChildren && (
            <div className="relative mt-8">
              <ol className="relative">
                {renderNodes(
                  node.children, 
                  level + 1, 
                  instanceId, 
                  isRunning,
                  isExpanded,
                  handleNodeClick
                )}
              </ol>
            </div>
          )}
        </div>
      </li>
    );
  });
}

export function Timeline({ nodes, instanceId = "standalone", isRunning = false }: TimelineProps) {
  const { isNodeExpanded, toggleNode } = useTrajectoryStore();
  
  const lastNode = nodes[nodes.length - 1];
  const isTerminal = lastNode?.terminal;

  const isExpanded = (nodeId: number) => isNodeExpanded(instanceId, nodeId);
  const handleNodeClick = (nodeId: number) => {
    toggleNode(instanceId, nodeId);
  };

  return (
    <div className="w-full">
      <div className="relative">
        <div
          className="absolute bottom-0 left-[80px] top-0 w-px bg-gray-200 sm:left-[150px]"
          style={isTerminal ? { bottom: "2rem" } : undefined}
        />

        <ol className="relative">
          {renderNodes(nodes, 0, instanceId, isRunning, isExpanded, handleNodeClick)}

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
              </div>
            </li>
          )}
        </ol>
      </div>
    </div>
  );
}
