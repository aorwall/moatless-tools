import { useTrajectoryStore } from '@/pages/trajectory/stores/trajectoryStore';
import { ScrollArea } from '@/lib/components/ui/scroll-area';
import { CompletionDetails } from './details/CompletionDetails';
import { ActionDetails } from './details/ActionDetails';
import { ErrorDetails } from './details/ErrorDetails';
import { MessageDetails } from './details/MessageDetails';
import { ObservationDetails } from './details/ObservationDetails';
import { WorkspaceDetails } from './details/WorkspaceDetails';

export const TimelineItemDetails = () => {
  const selectedItem = useTrajectoryStore((state) => state.selectedItem);

  if (!selectedItem) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-muted-foreground">
        Select an item to view details
      </div>
    );
  }

  const renderContent = () => {
    switch (selectedItem.type) {
      case 'completion':
        return <CompletionDetails content={selectedItem.content} />;
      case 'action':
        return <ActionDetails content={selectedItem.content} />;
      case 'error':
        return <ErrorDetails content={selectedItem.content} />;
      case 'user_message':
      case 'assistant_message':
      case 'thought':
        return <MessageDetails content={selectedItem.content} type={selectedItem.type} />;
      case 'observation':
        return <ObservationDetails content={selectedItem.content} />;
      case 'workspace':
        return <WorkspaceDetails content={selectedItem.content} />;
      default:
        return <div>Unsupported item type</div>;
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="border-b p-4">
        <h2 className="font-semibold">
          {selectedItem.type.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
          ).join(' ')} Details
        </h2>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-4">
          {renderContent()}
        </div>
      </ScrollArea>
    </div>
  );
}; 