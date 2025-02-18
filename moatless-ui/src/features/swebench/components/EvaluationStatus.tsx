import { Card, CardContent, CardHeader, CardTitle } from "@/lib/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/lib/components/ui/table";
import { Badge } from "@/lib/components/ui/badge";
import { useEvaluationStatus } from "../hooks/useEvaluationStatus";

interface EvaluationStatusProps {
  evaluationName: string;
}

export default function EvaluationStatus({ evaluationName }: EvaluationStatusProps) {
  const { data: evaluation, error, isLoading } = useEvaluationStatus(evaluationName);

  if (error) {
    return <div className="text-red-500">Error: {error instanceof Error ? error.message : 'Failed to fetch status'}</div>;
  }

  if (isLoading || !evaluation) {
    return <div>Loading...</div>;
  }

  const getStatusColor = (status: string): string => {
    switch (status.toLowerCase()) {
      case "running": return "blue";
      case "completed": return "green";
      case "error": return "destructive";
      default: return "secondary";
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Evaluation Details</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <div><strong>Dataset:</strong> {evaluation.dataset_name}</div>
          <div>
            <strong>Status:</strong>{" "}
            <Badge variant={getStatusColor(evaluation.status)}>{evaluation.status}</Badge>
          </div>
          <div><strong>Started:</strong> {new Date(evaluation.started_at).toLocaleString()}</div>
          {evaluation.completed_at && (
            <div><strong>Finished:</strong> {new Date(evaluation.completed_at).toLocaleString()}</div>
          )}
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-4">Instances</h3>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Instance ID</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Started</TableHead>
                <TableHead>Finished</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {evaluation.instances.map((instance) => (
                <TableRow key={instance.instance_id}>
                  <TableCell>{instance.instance_id}</TableCell>
                  <TableCell>
                    <Badge variant={getStatusColor(instance.status)}>{instance.status}</Badge>
                  </TableCell>
                  <TableCell>{instance.started_at ? new Date(instance.started_at).toLocaleString() : "-"}</TableCell>
                  <TableCell>{instance.completed_at ? new Date(instance.completed_at).toLocaleString() : "-"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
} 