"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Dropzone from "@/components/Dropzone";
import FileList from "@/components/FileList";
import Stepper from "@/components/Stepper";
import PreviewPane from "@/components/PreviewPane";
import JsonPane from "@/components/JsonPane";
import Toast from "@/components/Toast";

type JobStatus = {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  source_files: string[];
  raw_text_preview?: string;
  result?: any;
  error?: { message: string; detail?: string };
};

const MAX_TOTAL_MB = 50;
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const SYNC_MODE = process.env.NEXT_PUBLIC_SYNC_MODE === "true";

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<"idle" | "uploading" | "processing" | "done" | "error">("idle");
  const [stage, setStage] = useState<string>("UPLOAD");
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [rawText, setRawText] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<string | null>(null);
  const [toastVisible, setToastVisible] = useState(false);

  const totalBytes = useMemo(() => files.reduce((sum, f) => sum + f.size, 0), [files]);

  const setToast = useCallback(() => {
    setToastVisible(true);
    setTimeout(() => setToastVisible(false), 1200);
  }, []);

  const handleFilesAdded = useCallback(
    (incoming: File[]) => {
      const next = [...files, ...incoming];
      const total = next.reduce((sum, f) => sum + f.size, 0);
      if (total / (1024 * 1024) > MAX_TOTAL_MB) {
        setError(`Total size exceeds ${MAX_TOTAL_MB}MB.`);
        return;
      }
      setError(null);
      setFiles(next);
    },
    [files]
  );

  const removeFile = useCallback(
    (index: number) => {
      const next = files.filter((_, i) => i !== index);
      setFiles(next);
    },
    [files]
  );

  const pollJob = useCallback(
    async (id: string, attempt = 0) => {
      const wait = Math.min(750 + attempt * 150, 1500);
      await new Promise((res) => setTimeout(res, wait));
      const res = await fetch(`${BACKEND_URL}/api/jobs/${id}`);
      if (!res.ok) {
        setStatus("error");
        setError("Failed to fetch job status.");
        return;
      }
      const data = (await res.json()) as JobStatus;
      setStage(data.stage);
      if (data.status === "done") {
        setStatus("done");
        setResult(data.result);
        setRawText(data.raw_text_preview || "");
        return;
      }
      if (data.status === "error") {
        setStatus("error");
        setError(data.error?.message || "Pipeline error.");
        setDetail(data.error?.detail || "");
        return;
      }
      setStatus("processing");
      pollJob(id, attempt + 1);
    },
    []
  );

  const runExtraction = useCallback(async () => {
    setError(null);
    setDetail(null);
    setStatus("uploading");
    setStage("UPLOAD");
    setResult(null);
    setRawText("");

    const form = new FormData();
    files.forEach((f) => form.append("files", f));

    const endpoint = SYNC_MODE ? "/api/extract_sync" : "/api/extract";
    const res = await fetch(`${BACKEND_URL}${endpoint}`, { method: "POST", body: form });
    if (!res.ok) {
      const message = await res.text();
      setStatus("error");
      setError("Upload failed.");
      setDetail(message);
      return;
    }

    const data = await res.json();
    if (SYNC_MODE) {
      setStatus(data.status === "done" ? "done" : "processing");
      setStage(data.stage || "OUTPUT");
      setResult(data.result || null);
      setRawText(data.raw_text_preview || "");
      return;
    }

    setStatus("processing");
    setStage("EXTRACT");
    setJobId(data.job_id);
    pollJob(data.job_id);
  }, [files, pollJob]);

  const copyJson = useCallback(async () => {
    if (!result) return;
    const text = JSON.stringify(result, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      setToast();
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      document.body.removeChild(area);
      setToast();
    }
  }, [result, setToast]);

  const copyRaw = useCallback(async () => {
    if (!rawText) return;
    try {
      await navigator.clipboard.writeText(rawText);
      setToast();
    } catch {
      const area = document.createElement("textarea");
      area.value = rawText;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      document.body.removeChild(area);
      setToast();
    }
  }, [rawText, setToast]);

  useEffect(() => {
    if (!files.length) {
      setStatus("idle");
      setStage("UPLOAD");
      setResult(null);
      setRawText("");
      setError(null);
      setDetail(null);
    }
  }, [files.length]);

  return (
    <main className="min-h-screen relative scanlines">
      <div className="max-w-[1400px] mx-auto px-10 py-10 space-y-10">
        <header className="grid-12 items-end">
          <div className="col-span-6 space-y-4">
            <div className="text-xs uppercase tracking-[0.3em] font-crest text-muted">AxiomESG</div>
            <div className="font-hero text-6xl uppercase">INGEST</div>
            <div className="font-heading text-xl max-w-[32ch]">
              A deterministic intelligence layer for ESG extraction.
            </div>
          </div>
          <div className="col-span-6 flex justify-end">
            <div className="text-xs uppercase tracking-[0.2em] text-muted">
              Status: {status.toUpperCase()}
            </div>
          </div>
        </header>

        <div className="border-t border-hairline pt-4">
          <Stepper status={status} stage={stage} />
        </div>

        {error ? (
          <div className="border border-hairline p-4 space-y-2">
            <div className="text-xs uppercase tracking-[0.2em]">Error</div>
            <div className="text-sm">{error}</div>
            {detail ? (
              <details className="text-xs text-muted">
                <summary className="cursor-pointer">Raw details</summary>
                <pre className="mono mt-2 whitespace-pre-wrap">{detail}</pre>
              </details>
            ) : null}
          </div>
        ) : null}

        <div className="grid-12 gap-8">
          <section className="col-span-4 space-y-6">
            <div className="text-xs uppercase tracking-[0.3em] font-crest">INGEST</div>
            <Dropzone
              onFilesAdded={handleFilesAdded}
              maxTotalMB={MAX_TOTAL_MB}
              currentTotalBytes={totalBytes}
              error={error}
            />
            {files.length ? <FileList files={files} onRemove={removeFile} /> : null}
            <button
              type="button"
              className="w-full border border-hairline px-4 py-3 text-xs uppercase tracking-[0.3em] focus-ring transition-all"
              onClick={runExtraction}
              disabled={!files.length || status === "uploading" || status === "processing"}
              aria-label="Run extraction"
            >
              Run Extraction
            </button>
          </section>

          <section className="col-span-4 space-y-6">
            <div className="text-xs uppercase tracking-[0.3em] font-crest">PREVIEW</div>
            <PreviewPane files={files} rawText={rawText} extractionDone={status === "done"} onCopyRaw={copyRaw} />
          </section>

          <section className="col-span-4 space-y-6">
            <div className="text-xs uppercase tracking-[0.3em] font-crest">ESG JSON</div>
            <JsonPane data={result} onCopy={copyJson} />
          </section>
        </div>
      </div>

      <Toast message="COPIED" visible={toastVisible} />
    </main>
  );
}
