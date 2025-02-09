import { FC, useEffect, useState } from 'react';
import { FileText } from 'lucide-react';
import { Button } from '@/lib/components/ui/button';
import { Document, pdfjs } from 'react-pdf'


console.log(import.meta.url);
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

interface PDFPreviewProps {
  content: string;
  fileName?: string;
}

export const PDFPreview: FC<PDFPreviewProps> = ({ content, fileName }) => {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    // Create data URL directly from base64 content
    const dataUrl = `data:application/pdf;base64,${content}`;
    setUrl(dataUrl);

    // No cleanup needed since we're not creating object URLs
  }, [content]);

  if (!url) return null;

  return (
    <div className="space-y-2">
      <div className="relative h-[600px] w-full overflow-hidden rounded-lg border bg-white">
        <Document file={url} />
      </div>
      <div className="flex gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => window.open(url, '_blank')}
        >
          <FileText className="mr-2 h-4 w-4" />
          Open in New Tab
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            // Create temporary link to download the PDF
            const link = document.createElement('a');
            link.href = url;
            link.download = fileName || 'document.pdf';
            link.click();
          }}
        >
          <FileText className="mr-2 h-4 w-4" />
          Download PDF
        </Button>
      </div>
    </div>
  );
}; 