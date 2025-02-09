import type { ContentStructure } from "@/lib/types/content"
import { SectionRenderer } from "./SectionRenderer"

interface ContentViewProps {
  content: ContentStructure
}

export function ContentView({ content }: ContentViewProps) {
  return (
    <div className="w-full">
      <h1 className="text-2xl font-bold mb-4">{content.title}</h1>
      <div className="space-y-4">
        {content.sections.map((section) => (
          <SectionRenderer key={section.id} section={section} />
        ))}
      </div>
    </div>
  )
}

