import type React from "react"

interface ContentContainerProps {
  children: React.ReactNode
}

export function ContentContainer({ children }: ContentContainerProps) {
  return <div className="max-w-4xl mx-auto p-6 bg-white shadow-lg rounded-lg">{children}</div>
}

