type StepperProps = {
  status: "idle" | "uploading" | "processing" | "done" | "error";
  stage?: string;
};

const STEPS = ["UPLOAD", "EXTRACT", "FILTER", "WEIGHT", "INTELLIGENCE", "VALIDATE", "OUTPUT"];

export default function Stepper({ status, stage }: StepperProps) {
  const currentIndex = stage ? STEPS.indexOf(stage) : -1;

  return (
    <div className="flex items-center gap-4">
      {STEPS.map((step, idx) => {
        const isDone = status === "done" || (currentIndex >= 0 && idx < currentIndex);
        const isActive = currentIndex === idx && status !== "done" && status !== "error";
        return (
          <div key={step} className="flex items-center gap-2 text-xs uppercase tracking-[0.2em]">
            <div
              className={`h-3 w-3 border border-hairline ${isDone ? "bg-black" : "bg-transparent"} ${
                isActive ? "animate-pulse" : ""
              }`}
              aria-hidden
            />
            <span className={isActive ? "" : "text-muted"}>{step}</span>
          </div>
        );
      })}
      {status === "error" ? <span className="text-xs uppercase tracking-[0.2em]">FAILED</span> : null}
    </div>
  );
}
