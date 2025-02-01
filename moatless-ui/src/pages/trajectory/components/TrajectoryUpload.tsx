import { useRef, useState } from 'react';
import { Upload, FileText, Info } from 'lucide-react';
import { Card } from '@/lib/components/ui/card';
import { Input } from '@/lib/components/ui/input';
import { Button } from '@/lib/components/ui/button';
import { useTrajectoryUpload } from '@/lib/hooks/useTrajectory';

interface TrajectoryUploadProps {
  onLoadTrajectory: (path: string) => void;
  searchParams: URLSearchParams;
  setSearchParams: (params: URLSearchParams) => void;
}

export function TrajectoryUpload({ onLoadTrajectory, searchParams, setSearchParams }: TrajectoryUploadProps) {
  const [filePath, setFilePath] = useState(searchParams.get('path') || '');
  const [activeTab, setActiveTab] = useState<'path' | 'upload'>('path');
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const uploadMutation = useTrajectoryUpload();

  const handleLoadTrajectory = () => {
    if (!filePath) return;
    setSearchParams(new URLSearchParams({ path: filePath }));
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (activeTab === 'path') {
      setFilePath(file.name);
      handleLoadTrajectory();
    } else {
      try {
        const result = await uploadMutation.mutateAsync(file);
        onLoadTrajectory(result.path);
      } catch (error) {
        console.error('Failed to upload trajectory:', error);
      }
    }
  };

  return (
    <Card className="p-6">
      <div className="mb-6 flex space-x-4 border-b">
        <button
          className={`px-4 py-2 -mb-px ${
            activeTab === 'path'
              ? 'border-b-2 border-primary text-primary'
              : 'text-muted-foreground'
          }`}
          onClick={() => setActiveTab('path')}
        >
          <div className="flex items-center space-x-2">
            <FileText className="h-4 w-4" />
            <span>Load from Path</span>
          </div>
        </button>
        <button
          className={`px-4 py-2 -mb-px ${
            activeTab === 'upload'
              ? 'border-b-2 border-primary text-primary'
              : 'text-muted-foreground'
          }`}
          onClick={() => setActiveTab('upload')}
        >
          <div className="flex items-center space-x-2">
            <Upload className="h-4 w-4" />
            <span>Upload File</span>
          </div>
        </button>
      </div>

      {activeTab === 'path' ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Info className="h-4 w-4" />
            <span>Enter the absolute path to a trajectory file on your local machine</span>
          </div>
          <div className="flex items-center gap-4">
            <Input
              type="text"
              placeholder="/absolute/path/to/trajectory.json"
              value={filePath}
              onChange={(e) => setFilePath(e.target.value)}
              className="flex-1 font-mono text-sm"
            />
            <Button onClick={handleLoadTrajectory}>Load Trajectory</Button>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-4">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            className="hidden"
            accept=".json,.jsonl"
          />
          <Button
            variant="outline"
            className="flex-1"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadMutation.isPending}
          >
            <Upload className="mr-2 h-4 w-4" />
            {uploadMutation.isPending ? 'Uploading...' : 'Choose File to Upload'}
          </Button>
        </div>
      )}
    </Card>
  );
} 