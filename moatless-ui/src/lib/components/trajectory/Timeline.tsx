import { Node } from "@/lib/types/trajectory";
import { TimelineItem } from "@/lib/components/trajectory/TimelineItem";
import { TrajectoryNode } from "@/lib/components/trajectory/TrajectoryNode";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { NodeCircle } from '@/lib/components/trajectory/NodeCircle';

interface TimelineProps {
  nodes: Node[];
  instanceId?: string;
  isRunning: boolean;
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
          {nodes.map((node, index) => {
            const isLastNode = index === nodes.length - 1;
            return (
              <li key={node.nodeId} className="mb-4 sm:mb-8">
                <div className="group relative">
                  {/* Node header */}
                  <div className="relative rounded-lg">
                    <div className="flex items-start">
                      {/* Step number */}
                      <div className="flex w-[80px] shrink-0 items-start justify-end sm:w-[150px]">
                        <div className="mr-3 flex h-8 flex-col justify-center text-right sm:mr-6">
                          <button
                            className="max-w-[60px] cursor-pointer truncate text-xs font-medium text-gray-600 group-hover:text-gray-900 sm:max-w-[120px]"
                            onClick={() => handleNodeClick(node.nodeId)}
                          >
                            {node.nodeId === 0
                              ? "Start"
                              : `Step ${node.nodeId}`}
                          </button>
                        </div>
                      </div>

                      {/* Node circle */}
                      <NodeCircle
                        node={node}
                        isLastNode={isLastNode}
                        isRunning={isRunning}
                        onClick={() => handleNodeClick(node.nodeId)}
                      />

                      {/* Node content */}
                      <div className="min-w-0 flex-1 pl-8">
                        <div className="flex items-start justify-between">
                          <div className="min-w-0 flex-1">
                            <TrajectoryNode
                              node={node}
                              expanded={isExpanded(node.nodeId)}
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
                          itemId={index}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </li>
            );
          })}

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
