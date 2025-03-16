import { Box, Cpu, BrainCircuit, Search, Replace, Code, FileTerminal, Settings, AlertCircle, Medal } from "lucide-react"
import type { TreeItem, ActionItem } from "@/features/trajectory/components/tree-view/types"

interface TreeIconProps {
  item: TreeItem
}

export function TreeIcon({ item }: TreeIconProps) {
  switch (item.type) {
    case "node":
      return (
        <div className="w-5 h-5 rounded-md bg-purple-600 flex items-center justify-center">
          <Box className="w-3.5 h-3.5 text-white" />
        </div>
      )
    case "completion":
      return (
        <div className="w-5 h-5 rounded-md bg-blue-500 flex items-center justify-center">
          <Cpu className="w-3.5 h-3.5 text-white" />
        </div>
      )
    case "thought":
      return (
        <div className="w-5 h-5 rounded-md bg-amber-500 flex items-center justify-center">
          <BrainCircuit className="w-3.5 h-3.5 text-white" />
        </div>
      )
    case "action":
      const actionItem = item as ActionItem
      // Map action names to icons
      let ActionIcon = Code // Default icon
      let bgColor = "bg-green-500" // Default background color

      // Map common action types to specific icons
      if (actionItem.action_name in ["SemanticSearch", "FindClass", "FindFunction"]) {
        ActionIcon = Search
      } else if (actionItem.action_name === "StringReplace") {
        ActionIcon = Replace
      }

      return (
        <div className={`w-5 h-5 rounded-md ${bgColor} flex items-center justify-center`}>
          <ActionIcon className="w-3.5 h-3.5 text-white" />
        </div>
      )

    case "reward":
      return (
        <div className="w-5 h-5 rounded-md bg-amber-400 flex items-center justify-center">
          <Medal className="w-3.5 h-3.5 text-white" />
        </div>
      )

    case "error":
      return (
        <div className="w-5 h-5 rounded-md bg-red-500 flex items-center justify-center">
          <AlertCircle className="w-3.5 h-3.5 text-white" />
        </div>
      )
  }
}

