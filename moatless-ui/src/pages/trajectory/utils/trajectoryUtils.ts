
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

export function hasSuccessfulChanges(node: Node): boolean {
  const hasPatches = node.fileContext?.files?.some((f) => f.patch) ?? false;
  const hasTestErrors =
    node.fileContext?.testResults?.some(
      (t) => t.status === "failed" || t.status === "error",
    ) ?? false;
  return hasPatches && !hasTestErrors;
}

export function getNodeColor(node: Node): string {
  if (node.nodeId === 0) return "blue";
  if (node.allNodeErrors.length > 0) return "red";
  if (node.allNodeWarnings.length > 0) return "yellow";

  const lastAction = node.actionSteps[node.actionSteps.length - 1]?.action.name;
  if (lastAction === "Finish") {
    return hasSuccessfulChanges(node) ? "green" : "red";
  }

  if (node.fileContext?.updatedFiles?.length) return "green";
  return "gray";
}
