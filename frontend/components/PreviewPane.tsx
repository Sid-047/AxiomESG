import { useEffect, useMemo, useState } from "react";

type PreviewPaneProps = {
  files: File[];
  rawText?: string | null;
  onCopyRaw: () => void;
  extractionDone: boolean;
};

export default function PreviewPane({ files, rawText, onCopyRaw, extractionDone }: PreviewPaneProps) {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);

  const firstPdf = useMemo(() => files.find((f) => f.name.toLowerCase().endsWith(".pdf")), [files]);

  useEffect(() => {
    if (!firstPdf) {
      setPdfUrl(null);
      return;
    }
    const url = URL.createObjectURL(firstPdf);
    setPdfUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [firstPdf]);

  if (extractionDone && rawText) {
    return (
      <div className="panel p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="text-xs uppercase tracking-[0.2em]">Extracted Text</div>
          <button
            type="button"
            className="border border-hairline px-3 py-1 text-xs uppercase tracking-[0.2em] focus-ring transition-all"
            onClick={onCopyRaw}
            aria-label="Copy extracted text"
          >
            Copy Raw Text
          </button>
        </div>
        <div className="mono text-xs max-h-[420px] overflow-auto border border-hairline p-3 whitespace-pre-wrap">
          {rawText}
        </div>
      </div>
    );
  }

  return (
    <div className="panel p-4 space-y-3">
      <div className="text-xs uppercase tracking-[0.2em]">Preview</div>
      {pdfUrl ? (
        <div className="border border-hairline">
          <embed src={pdfUrl} type="application/pdf" className="w-full h-[420px]" />
        </div>
      ) : (
        <div className="space-y-3 text-sm text-muted">
          <div>{files.length ? "Preview available after extraction." : "Select files to preview."}</div>
          {files.length ? (
            <div className="space-y-1 text-xs">
              {files.map((file, idx) => (
                <div key={`${file.name}-${idx}`} className="flex justify-between">
                  <span>{file.name}</span>
                  <span>{(file.size / 1024 / 1024).toFixed(2)}MB</span>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      )}
      {files.length ? (
        <div className="text-xs text-muted">
          {files.length} file(s) staged. Preview is available for PDFs only.
        </div>
      ) : null}
    </div>
  );
}
