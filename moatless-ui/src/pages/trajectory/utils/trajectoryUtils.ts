import { TestResultsSummary, Node } from "@/lib/types/trajectory";
export function getTestResultsSummary(
  testResults: Array<{ status: string; name?: string; message?: string }> = [],
): TestResultsSummary {
  return {
    total: testResults.length,
    passed: testResults.filter((t) => t.status === "passed").length,
    failed: testResults.filter((t) => t.status === "failed").length,
    errors: testResults.filter((t) => t.status === "error").length,
    skipped: testResults.filter((t) => t.status === "skipped").length,
  };
} 


export function getNodeColor(node: Node): string {
  if (node.nodeId === 0) return "blue";
  if (node.allNodeErrors.length > 0) return "red";
  if (node.allNodeWarnings.length > 0) return "yellow";

  return "green";
}
