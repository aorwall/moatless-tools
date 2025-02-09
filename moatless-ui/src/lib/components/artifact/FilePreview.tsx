import { FC } from 'react';
import { FileText } from 'lucide-react';
import { Button } from '@/lib/components/ui/button';
import { PDFPreview } from '@/lib/components/artifact/PDFPreview';

interface FilePreviewProps {
  mimeType: string;
  content: string;
  fileName?: string;
}

export const FilePreview: FC<FilePreviewProps> = ({ mimeType, content, fileName }) => {
  if (!content) return null;

  if (mimeType.startsWith('image/')) {
    return (
      <div className="relative aspect-video w-full overflow-hidden rounded-lg border bg-gray-100">
        <img
          src={`data:${mimeType};base64,${content}`}
          alt={fileName || 'File preview'}
          className="h-full w-full object-contain"
          loading="lazy"
        />
      </div>
    );
  }

  if (mimeType === 'application/pdf') {
    return <PDFPreview content={content} fileName={fileName} />;
  }

  if (mimeType.startsWith('text/')) {
    try {
      const decodedContent = atob(content);
      return (
        <pre className="max-h-96 overflow-auto rounded-lg border bg-gray-50 p-4 text-sm">
          {decodedContent}
        </pre>
      );
    } catch (e) {
      console.error('Failed to decode text content:', e);
      return (
        <div className="rounded-lg border bg-red-50 p-4 text-sm text-red-600">
          Failed to decode text content
        </div>
      );
    }
  }

  // Default file representation with download option
  const blob = new Blob([Buffer.from(content, 'base64')], { type: mimeType });
  const url = URL.createObjectURL(blob);

  return (
    <div className="flex items-center justify-between rounded-lg border bg-gray-50 p-4">
      <div className="flex items-center gap-2">
        <FileText className="h-5 w-5 text-gray-400" />
        <span className="text-sm text-gray-600">
          File type: {mimeType}
        </span>
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={() => {
          const link = document.createElement('a');
          link.href = url;
          link.download = fileName || 'download';
          link.click();
          setTimeout(() => URL.revokeObjectURL(url), 100);
        }}
      >
        <FileText className="mr-2 h-4 w-4" />
        Download
      </Button>
    </div>
  );
}; 