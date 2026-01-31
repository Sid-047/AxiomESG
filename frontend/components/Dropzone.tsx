import { useCallback, useRef } from "react";

type DropzoneProps = {
  onFilesAdded: (files: File[]) => void;
  maxTotalMB: number;
  currentTotalBytes: number;
  error?: string | null;
};

const ACCEPT = [".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".png", ".jpg", ".jpeg"];

export default function Dropzone({ onFilesAdded, maxTotalMB, currentTotalBytes, error }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (fileList: FileList | null) => {
      if (!fileList) return;
      onFilesAdded(Array.from(fileList));
    },
    [onFilesAdded]
  );

  return (
    <div className="space-y-4">
      <div
        className="panel p-4 relative"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFiles(e.dataTransfer.files);
        }}
      >
        <div className="flex flex-col gap-2">
          <div className="text-xs uppercase tracking-[0.2em] text-muted font-body">Drop files</div>
          <div className="text-sm font-body">
            Drag &amp; drop ESG documents or use the picker. Max total {maxTotalMB}MB.
          </div>
          <button
            type="button"
            className="mt-2 w-fit border border-hairline px-4 py-2 text-xs uppercase tracking-[0.2em] focus-ring transition-all"
            onClick={() => inputRef.current?.click()}
            aria-label="Open file picker"
          >
            Choose Files
          </button>
        </div>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          multiple
          accept={ACCEPT.join(",")}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
      <div className="text-xs text-muted">
        Accepted: {ACCEPT.join(" ")} | Current total: {(currentTotalBytes / (1024 * 1024)).toFixed(2)}MB
      </div>
      {error ? (
        <div className="border border-hairline p-3 text-xs uppercase tracking-[0.2em]">{error}</div>
      ) : null}
    </div>
  );
}
