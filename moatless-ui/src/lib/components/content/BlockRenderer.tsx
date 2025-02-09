import type { ContentBlock } from "@/lib/types/content"
import { TextBlock } from "./blocks/TextBlock"
import { ListBlock } from "./blocks/ListBlock"
import { TableBlock } from "./blocks/TableBlock"

interface BlockRendererProps {
  block: ContentBlock
}

export function BlockRenderer({ block }: BlockRendererProps) {
  switch (block.type) {
    case "text":
      return <TextBlock {...block} />
    case "list":
      return <ListBlock {...block} />
    case "table":
      return <TableBlock {...block} />
    default:
      console.warn(`Unknown block type: ${(block as any).type}`)
      return null
  }
}

