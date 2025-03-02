import type { ListBlock as ListBlockType } from "@/lib/types/content";

export function ListBlock({ items, variant = "unordered" }: ListBlockType) {
  const ListComponent = variant === "ordered" ? "ol" : "ul";
  return (
    <ListComponent
      className={`space-y-1 ${variant === "unordered" ? "list-disc" : "list-decimal"} pl-4 text-sm text-gray-700`}
    >
      {items.map((item, index) => (
        <li key={index} className="leading-normal">
          {item}
        </li>
      ))}
    </ListComponent>
  );
}
