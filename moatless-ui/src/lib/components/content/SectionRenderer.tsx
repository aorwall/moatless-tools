import type { ContentSection } from "@/lib/types/content"
import { BlockRenderer } from "./BlockRenderer"

interface SectionRendererProps {
  section: ContentSection
  level?: number
}

export function SectionRenderer({ section, level = 1 }: SectionRendererProps) {
  const HeaderComponent = level === 1 ? "h2" : "h3"
  const headerClass = level === 1 
    ? "text-xl font-bold text-gray-800" 
    : "text-lg font-semibold text-gray-700"

  return (
    <div className="w-full">
      <div className="border-b border-gray-200 mb-3">
        <HeaderComponent className={headerClass}>{section.title}</HeaderComponent>
      </div>
      <div className="space-y-2">
        {section.blocks.map((block) => (
          <BlockRenderer key={block.id} block={block} />
        ))}
        {section.sections && section.sections.length > 0 && (
          <div className="space-y-4 mt-6 pl-4">
            {section.sections.map((subSection) => (
              <SectionRenderer key={subSection.id} section={subSection} level={level + 1} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

