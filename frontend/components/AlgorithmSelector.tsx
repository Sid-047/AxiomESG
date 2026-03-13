"use client";

import { useEffect, useState } from "react";

type Algorithm = {
  key: string;
  label: string;
  description: string;
};

type AlgorithmSelectorProps = {
  selected: string;
  onSelect: (key: string) => void;
  disabled?: boolean;
};

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function AlgorithmSelector({ selected, onSelect, disabled }: AlgorithmSelectorProps) {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);

  useEffect(() => {
    fetch(`${BACKEND_URL}/api/algorithms`)
      .then((res) => res.json())
      .then((data: Algorithm[]) => setAlgorithms(data))
      .catch(() => {
        // Fallback if backend is not running yet
        setAlgorithms([
          { key: "heuristic", label: "Heuristic AWFA", description: "Keyword-length heuristic weighting." },
          { key: "bert_mean", label: "BERT + Mean Fusion", description: "BERT with empirical-prior weighted mean." },
          { key: "bert_static", label: "BERT + Static Fusion", description: "BERT with fixed E/S/G weights." },
          { key: "bert_awfa_v1", label: "BERT + AWFA v1", description: "BERT with attention context network." },
          { key: "bert_awfa_v2", label: "BERT + AWFA v2", description: "BERT with multi-head attention." },
        ]);
      });
  }, []);

  if (!algorithms.length) return null;

  return (
    <div className="space-y-3">
      <div className="text-xs uppercase tracking-[0.2em] text-muted font-body">Algorithm</div>
      <div className="flex flex-wrap gap-2">
        {algorithms.map((algo) => {
          const isActive = selected === algo.key;
          return (
            <button
              key={algo.key}
              type="button"
              className={`border border-hairline px-3 py-2 text-xs uppercase tracking-[0.15em] focus-ring transition-all ${
                isActive ? "bg-black text-white border-black" : ""
              }`}
              onClick={() => onSelect(algo.key)}
              disabled={disabled}
              aria-label={`Select ${algo.label}`}
              aria-pressed={isActive}
              title={algo.description}
            >
              {algo.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
