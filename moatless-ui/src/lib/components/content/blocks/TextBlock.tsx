import type { TextBlock as TextBlockType } from "@/lib/types/content"

export function TextBlock({ content, variant = "default" }: TextBlockType) {
  switch (variant) {
    case "heading":
      return <h2 className="text-lg font-bold text-gray-900 mb-2">{content}</h2>
    case "subheading":
      return <h3 className="text-base font-semibold text-gray-800 mb-1.5">{content}</h3>
    default:
      return <p className="text-sm leading-normal text-gray-700">{content}</p>
  }
}

