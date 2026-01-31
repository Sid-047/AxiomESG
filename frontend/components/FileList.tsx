import { Trash2 } from "lucide-react";

type FileListProps = {
  files: File[];
  onRemove: (index: number) => void;
};

export default function FileList({ files, onRemove }: FileListProps) {
  return (
    <div className="space-y-2">
      {files.map((file, idx) => (
        <div key={`${file.name}-${idx}`} className="flex items-center justify-between border border-hairline px-3 py-2">
          <div className="text-xs">
            <div className="uppercase tracking-[0.15em]">{file.name}</div>
            <div className="text-muted">{(file.size / 1024 / 1024).toFixed(2)}MB</div>
          </div>
          <button
            type="button"
            aria-label={`Remove ${file.name}`}
            className="border border-hairline px-2 py-1 focus-ring transition-all"
            onClick={() => onRemove(idx)}
          >
            <Trash2 size={14} strokeWidth={1} />
          </button>
        </div>
      ))}
    </div>
  );
}
