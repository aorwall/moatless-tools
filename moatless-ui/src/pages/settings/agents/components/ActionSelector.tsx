import { useState } from 'react';
import { Search, X } from 'lucide-react';
import { Input } from '@/lib/components/ui/input';
import { Badge } from '@/lib/components/ui/badge';
import { Checkbox } from '@/lib/components/ui/checkbox';
import { ScrollArea } from '@/lib/components/ui/scroll-area';
import { useAvailableActions } from '@/lib/hooks/useAgents';
import type { ActionInfo } from '@/lib/types/agent';

interface ActionSelectorProps {
  selectedActions?: string[];
  onChange: (actions: string[]) => void;
}

export function ActionSelector({ selectedActions = [], onChange }: ActionSelectorProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const { data: actions, isLoading, error } = useAvailableActions();

  const filteredActions = actions?.filter((action) => {
    const query = searchQuery.toLowerCase();
    return action.name.toLowerCase().includes(query) || 
           action.description.toLowerCase().includes(query);
  }) ?? [];

  const categories = groupActionsByCategory(filteredActions);

  function groupActionsByCategory(actions: ActionInfo[]) {
    return actions.reduce((acc, action) => {
      const category = action.name.split('_')[0] || 'Other';
      if (!acc[category]) {
        acc[category] = [];
      }
      acc[category].push(action);
      return acc;
    }, {} as Record<string, ActionInfo[]>);
  }

  function toggleAction(actionName: string) {
    const newSelectedActions = [...selectedActions];
    const index = newSelectedActions.indexOf(actionName);
    
    if (index === -1) {
      newSelectedActions.push(actionName);
    } else {
      newSelectedActions.splice(index, 1);
    }
    
    onChange(newSelectedActions);
  }

  if (error) {
    return <div className="text-destructive">Failed to load actions</div>;
  }

  if (isLoading) {
    return <div className="flex items-center justify-center h-32">Loading actions...</div>;
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-4 bg-background z-10 p-2">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search actions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8"
          />
        </div>
        <Badge variant="secondary" className="ml-4">
          {selectedActions.length} action{selectedActions.length === 1 ? '' : 's'} selected
        </Badge>
      </div>

      <div className="flex gap-6 flex-1 min-h-0">
        <ScrollArea className="flex-1 border rounded-lg">
          <div className="p-4 space-y-6">
            {Object.entries(categories).map(([category, categoryActions]) => (
              <div key={category}>
                <h3 className="font-semibold mb-3 text-sm text-muted-foreground">{category}</h3>
                <div className="space-y-2">
                  {categoryActions.map((action) => (
                    <div key={action.name} className="flex items-start space-x-3 p-3 rounded-lg hover:bg-muted/50 transition-colors">
                      <Checkbox
                        checked={selectedActions.includes(action.name)}
                        onCheckedChange={() => toggleAction(action.name)}
                        className="mt-1"
                      />
                      <div>
                        <div className="font-medium">{action.name}</div>
                        <p className="text-sm text-muted-foreground whitespace-pre-line">{action.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>

        {selectedActions.length > 0 && (
          <div className="w-72 border rounded-lg flex flex-col">
            <div className="p-4 border-b bg-muted/50">
              <h3 className="font-semibold">Selected Actions</h3>
              <p className="text-sm text-muted-foreground">Click to remove</p>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-4">
                <div className="space-y-2">
                  {selectedActions.map((actionName) => (
                    <button
                      key={actionName}
                      className="group w-full flex items-center justify-between p-2 rounded-md hover:bg-destructive/10 text-left transition-colors"
                      onClick={() => toggleAction(actionName)}
                    >
                      <span className="text-sm truncate flex-1">{actionName}</span>
                      <X className="h-4 w-4 text-destructive opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                    </button>
                  ))}
                </div>
              </div>
            </ScrollArea>
          </div>
        )}
      </div>
    </div>
  );
} 