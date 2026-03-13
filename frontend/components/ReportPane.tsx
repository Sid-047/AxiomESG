"use client";

import { useCallback, useEffect, useState } from "react";

type ReportPaneProps = {
  jobId: string | null;
  ready: boolean;
};

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function ReportPane({ jobId, ready }: ReportPaneProps) {
  const [report, setReport] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!jobId || !ready) {
      setReport("");
      return;
    }
    setLoading(true);
    fetch(`${BACKEND_URL}/api/jobs/${jobId}/report`)
      .then((res) => {
        if (!res.ok) throw new Error("Report not available.");
        return res.text();
      })
      .then((text) => setReport(text))
      .catch(() => setReport(""))
      .finally(() => setLoading(false));
  }, [jobId, ready]);

  const copyReport = useCallback(async () => {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(report);
    } catch {
      const area = document.createElement("textarea");
      area.value = report;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      document.body.removeChild(area);
    }
  }, [report]);

  const downloadReport = useCallback(() => {
    if (!report) return;
    const blob = new Blob([report], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `axiomesg-report-${jobId || "export"}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [report, jobId]);

  return (
    <div className="panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-[0.2em]">Report</div>
        <div className="flex gap-2">
          <button
            type="button"
            className="border border-hairline px-3 py-1 text-xs uppercase tracking-[0.2em] focus-ring transition-all"
            onClick={copyReport}
            disabled={!report}
            aria-label="Copy report"
          >
            Copy
          </button>
          <button
            type="button"
            className="border border-hairline px-3 py-1 text-xs uppercase tracking-[0.2em] focus-ring transition-all"
            onClick={downloadReport}
            disabled={!report}
            aria-label="Download report"
          >
            Download
          </button>
        </div>
      </div>
      <div className="mono text-xs max-h-[420px] overflow-auto border border-hairline p-3 whitespace-pre-wrap">
        {loading ? "Generating report..." : report || "No report yet."}
      </div>
    </div>
  );
}
