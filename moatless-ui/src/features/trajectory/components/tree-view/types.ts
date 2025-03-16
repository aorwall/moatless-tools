export type NodeType = "node"
export type ItemType = "completion" | "thought" | "action" | "error" | "reward"

export interface BaseItem {
  id: string
  type: ItemType | NodeType
  label: string
  node_id: number
  detail?: string
  time?: string
}

export interface NodeItem extends BaseItem {
  type: "node"
  timestamp: string
  parent_node_id?: number
  children?: (NodeItem | CompletionItem | ThoughtItem | ActionItem | ErrorItem)[]
}

export interface CompletionItem extends BaseItem {
  type: "completion"
  tokens?: number
  action_step_id?: number
}

export interface ThoughtItem extends BaseItem {
  type: "thought"
}

export interface ActionItem extends BaseItem {
  type: "action"
  action_name: string
  action_index: number
  children?: CompletionItem[]
}

export interface RewardItem extends BaseItem {
  type: "reward"
  reward: number
}

export interface ErrorItem extends BaseItem {
  type: "error"
  error: string
}

export type TreeItem = NodeItem | CompletionItem | ThoughtItem | ActionItem | ErrorItem | RewardItem

export interface SelectedItem {
  id: string
  node_id: number
  type: ItemType | NodeType
  parent_id?: number
}
