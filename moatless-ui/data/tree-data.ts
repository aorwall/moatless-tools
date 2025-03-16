import type { TreeItem } from "@/types/tree-types"

// Sample tree data
export const treeData: TreeItem[] = [
  {
    id: "node1",
    type: "node",
    label: "Node 1",
    timestamp: "11:12:30",
    children: [
      {
        id: "completion1",
        type: "completion",
        label: "Completion",
        detail: "(gpt-4o)",
        time: "0.95s",
        tokens: 350,
        nodeId: "node1",
      },
      {
        id: "thought1",
        type: "thought",
        label: "Thought",
        detail: '("thinking")',
        nodeId: "node1",
      },
      {
        id: "action1",
        type: "action",
        actionType: "semanticSearch",
        actionIndex: 0,
        label: "SemanticSearch",
        detail: '("query")',
        time: "1s",
        nodeId: "node1",
        children: [
          {
            id: "completion2",
            type: "completion",
            label: "Completion",
            detail: "(gpt-4o-mini)",
            time: "0.8s",
            nodeId: "node1",
            parentId: "action1",
          },
        ],
      },
      // Nested Node 3
      {
        id: "node3",
        type: "node",
        label: "Node 3",
        timestamp: "11:12:45",
        parentNodeId: "node1",
        children: [
          {
            id: "completion4",
            type: "completion",
            label: "Completion",
            detail: "(gpt-4o)",
            time: "0.7s",
            tokens: 280,
            nodeId: "node3",
          },
          // First action in Node 3 (index 0)
          {
            id: "node3-action-0",
            type: "action",
            actionType: "semanticSearch",
            actionIndex: 0,
            label: "SemanticSearch",
            detail: '("first query")',
            time: "0.5s",
            nodeId: "node3",
            children: [
              {
                id: "node3-action-0-completion",
                type: "completion",
                label: "Completion",
                detail: "(gpt-4o-mini)",
                time: "0.3s",
                tokens: 150,
                nodeId: "node3",
                parentId: "node3-action-0",
              },
            ],
          },
          // Second action in Node 3 (index 1)
          {
            id: "node3-action-1",
            type: "action",
            actionType: "semanticSearch",
            actionIndex: 1,
            label: "SemanticSearch",
            detail: '("second query")',
            time: "0.6s",
            nodeId: "node3",
            children: [
              {
                id: "node3-action-1-completion",
                type: "completion",
                label: "Completion",
                detail: "(gpt-4o)",
                time: "0.4s",
                tokens: 200,
                nodeId: "node3",
                parentId: "node3-action-1",
              },
            ],
          },
        ],
      },
      // Nested Node 4
      {
        id: "node4",
        type: "node",
        label: "Node 4",
        timestamp: "11:13:00",
        parentNodeId: "node1",
        children: [
          {
            id: "completion5",
            type: "completion",
            label: "Completion",
            detail: "(gpt-4o-mini)",
            time: "0.6s",
            tokens: 180,
            nodeId: "node4",
          },
          {
            id: "node4-action-0",
            type: "action",
            actionType: "stringReplace",
            actionIndex: 0,
            label: "StringReplace",
            detail: "(nested/file/path)",
            time: "0.4s",
            nodeId: "node4",
          },
        ],
      },
    ],
  },
  {
    id: "node2",
    type: "node",
    label: "Node 2",
    timestamp: "11:13:45",
    children: [
      {
        id: "completion3",
        type: "completion",
        label: "Completion",
        detail: "(gpt-4o)",
        time: "1.2s",
        tokens: 420,
        nodeId: "node2",
      },
      {
        id: "node2-action-0",
        type: "action",
        actionType: "stringReplace",
        actionIndex: 0,
        label: "StringReplace",
        detail: "(path/to/file)",
        time: "1.5s",
        nodeId: "node2",
      },
    ],
  },
]

