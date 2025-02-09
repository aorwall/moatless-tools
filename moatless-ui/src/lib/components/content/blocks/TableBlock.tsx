import type { TableBlock as TableBlockType } from "@/lib/types/content"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/lib/components/ui/table"

export function TableBlock({ headers, rows }: TableBlockType) {
  return (
    <div className="w-full overflow-x-auto">
      <div className="min-w-full">
        <Table>
          <TableHeader>
            <TableRow className="border-b border-gray-200">
              {headers.map((header, index) => (
                <TableHead key={index} className="py-2 px-2 text-sm font-semibold text-gray-900">
                  {header}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, rowIndex) => (
              <TableRow key={rowIndex} className="border-b border-gray-100">
                {row.map((cell, cellIndex) => (
                  <TableCell key={cellIndex} className="py-2 px-2 text-sm text-gray-700">
                    {cell}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}

