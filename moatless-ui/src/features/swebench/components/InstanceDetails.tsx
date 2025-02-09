import { Card, CardContent, CardHeader, CardTitle } from "@/lib/components/ui/card";
import { Badge } from "@/lib/components/ui/badge";
import { EvaluationInstance } from "../api/evaluation";

interface InstanceDetailsProps {
  instance: EvaluationInstance | null;
  getStatusColor: (status: string) => string;
  formatDate: (date: string) => string;
}

export function InstanceDetails({ instance, getStatusColor, formatDate }: InstanceDetailsProps) {
  if (!instance) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        Select an instance to view details
      </div>
    );
  }

  return (
    <div className="p-6">
      <Card>
        <CardHeader>
          <CardTitle>Instance Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <p className="font-medium">Instance ID</p>
            <p className="text-sm text-muted-foreground">{instance.instance_id}</p>
          </div>
          <div>
            <p className="font-medium">Status</p>
            <Badge variant={getStatusColor(instance.status)}>
              {instance.status}
            </Badge>
          </div>
          {instance.started_at && (
            <div>
              <p className="font-medium">Started</p>
              <p className="text-sm text-muted-foreground">
                {formatDate(instance.started_at)}
              </p>
            </div>
          )}
          {instance.completed_at && (
            <div>
              <p className="font-medium">Completed</p>
              <p className="text-sm text-muted-foreground">
                {formatDate(instance.completed_at)}
              </p>
            </div>
          )}
          {instance.error && (
            <div>
              <p className="font-medium text-destructive">Error</p>
              <p className="text-sm text-destructive">{instance.error}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
} 