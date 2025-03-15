import { AddModelDialog as FeatureAddModelDialog } from '@/features/settings/models/components/AddModelDialog';
import type { ModelConfig } from '@/lib/types/model';

interface AddModelDialogProps {
  baseModel: ModelConfig;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AddModelDialog(props: AddModelDialogProps) {
  return <FeatureAddModelDialog {...props} />;
}
