export type BlockType = "text" | "list" | "table"

export interface BaseBlock {
  type: BlockType
  id: string
}

export interface TextBlock extends BaseBlock {
  type: "text"
  content: string
  variant?: "default" | "heading" | "subheading"
}

export interface ListBlock extends BaseBlock {
  type: "list"
  items: string[]
  variant?: "unordered" | "ordered"
}

export interface TableBlock extends BaseBlock {
  type: "table"
  headers: string[]
  rows: (string | number)[][]
}

export type ContentBlock = TextBlock | ListBlock | TableBlock

export interface ContentSection {
  id: string
  title: string
  blocks: ContentBlock[]
  sections?: ContentSection[]
}

export interface ContentStructure {
  title: string
  sections: ContentSection[]
}

