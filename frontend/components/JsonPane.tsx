type JsonPaneProps = {
  data: unknown;
  onCopy: () => void;
};

export default function JsonPane({ data, onCopy }: JsonPaneProps) {
  return (
    <div className="panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-[0.2em]">ESG JSON</div>
        <button
          type="button"
          className="border border-hairline px-3 py-1 text-xs uppercase tracking-[0.2em] focus-ring transition-all"
          onClick={onCopy}
          aria-label="Copy ESG JSON"
        >
          Copy JSON
        </button>
      </div>
      <pre className="mono text-xs max-h-[420px] overflow-auto border border-hairline p-3 whitespace-pre-wrap">
        {data ? JSON.stringify(data, null, 2) : "No ESG JSON yet."}
      </pre>
    </div>
  );
}
