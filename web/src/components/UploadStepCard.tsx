"use client";

import { ImageUp } from "lucide-react";
import { useCallback, useRef, useState } from "react";

// Cropped-PlantDoc crops used for real-photo checks. Values are names the
// classifier was trained on (PlantVillage); PlantDoc spells soybean "Soyabean".
const CROPS = [
  { value: "Apple", label: "Apple" },
  { value: "Bell pepper", label: "Bell pepper" },
  { value: "Blueberry", label: "Blueberry" },
  { value: "Cherry", label: "Cherry" },
  { value: "Corn", label: "Corn" },
  { value: "Grape", label: "Grape" },
  { value: "Peach", label: "Peach" },
  { value: "Potato", label: "Potato" },
  { value: "Raspberry", label: "Raspberry" },
  { value: "Soybean", label: "Soybean" },
  { value: "Squash", label: "Squash" },
  { value: "Strawberry", label: "Strawberry" },
  { value: "Tomato", label: "Tomato" },
];
const OTHER = "__other__";

export function UploadStepCard({
  onAnalyze,
  loading,
}: {
  onAnalyze: (file: File, crop: string, symptoms: string) => void;
  loading: boolean;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [crop, setCrop] = useState("");
  const [otherCrop, setOtherCrop] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [demoError, setDemoError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const setFileWithPreview = useCallback((f: File) => {
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
  }, []);

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith("image/")) setFileWithPreview(f);
  }

  return (
    <div className="mx-auto max-w-2xl rounded-card bg-white p-6 shadow-card">
      <div className="mb-6 flex items-center justify-between">
        <h3 className="font-semibold text-forest">Start with a crop photo</h3>
        <span className="rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
          Step 1 of 2
        </span>
      </div>

      <div
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center rounded-xl border-2 border-dashed px-6 py-12 transition-colors ${
          dragOver
            ? "border-forest bg-sage/50"
            : "border-[var(--color-border)] bg-[var(--color-cream)]/50"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) setFileWithPreview(f);
          }}
        />
        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preview}
            alt="Preview"
            className="mb-4 max-h-48 rounded-lg object-contain"
          />
        ) : (
          <span className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-sage">
            <ImageUp className="h-7 w-7 text-forest" />
          </span>
        )}
        <p className="font-semibold text-forest">
          {preview ? "Change photo" : "Drop your photo here"}
        </p>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          or click to browse · JPEG, PNG, WebP
        </p>
      </div>

      <label className="mt-6 block text-sm font-medium text-forest">
        Crop (optional hint)
        <select
          value={crop}
          onChange={(e) => setCrop(e.target.value)}
          className="mt-1 w-full rounded-xl border border-[var(--color-border)] bg-white px-3 py-2.5 text-sm"
        >
          <option value="">Select crop…</option>
          {CROPS.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
          <option value={OTHER}>Other crop (not listed)</option>
        </select>
        <span className="mt-1 block text-xs font-normal text-[var(--color-text-muted)]">
          Real-photo tests work best on this PlantDoc list. The model was
          trained on PlantVillage photos of these same plants. PlantDoc spells
          soybean as Soyabean.
        </span>
      </label>
      {crop === OTHER && (
        <label className="mt-4 block text-sm font-medium text-forest">
          Which crop?
          <input
            value={otherCrop}
            onChange={(e) => setOtherCrop(e.target.value)}
            placeholder="e.g. hibiscus"
            className="mt-1 w-full rounded-xl border border-[var(--color-border)] px-3 py-2.5 text-sm"
          />
          <span className="mt-1 block text-xs font-normal text-[var(--color-text-muted)]">
            Plants outside this list were not in training. The result will say
            the model does not cover that crop instead of guessing a disease.
          </span>
        </label>
      )}

      <label className="mt-4 block text-sm font-medium text-forest">
        Extra symptoms (optional)
        <textarea
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          placeholder="e.g. spots spreading after rain"
          rows={2}
          className="mt-1 w-full resize-none rounded-xl border border-[var(--color-border)] px-3 py-2.5 text-sm"
        />
      </label>

      <button
        type="button"
        disabled={!file || loading}
        onClick={() =>
          file &&
          onAnalyze(file, crop === OTHER ? otherCrop.trim() : crop, symptoms)
        }
        className="mt-6 w-full rounded-xl bg-forest py-3 text-sm font-semibold text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
      >
        {loading ? "Analyzing…" : "Analyze leaf"}
      </button>

      <p className="mt-5 text-xs font-medium uppercase tracking-wide text-[var(--color-forest-muted)]">
        Demo these paths on purpose
      </p>
      <div className="mt-2 grid gap-2 sm:grid-cols-2">
        <button
          type="button"
          disabled={loading}
          onClick={async () => {
            setDemoError(null);
            try {
              onAnalyze(await makeDarkJpeg(), "", "");
            } catch (e) {
              setDemoError(e instanceof Error ? e.message : "Could not start the retake demo");
            }
          }}
          className="rounded-xl border border-[var(--color-border)] px-3 py-2.5 text-sm font-medium text-forest disabled:opacity-40"
        >
          Show a retake
        </button>
        <button
          type="button"
          disabled={loading}
          onClick={async () => {
            setDemoError(null);
            try {
              onAnalyze(await loadNotSureDemo(), "", "");
            } catch (e) {
              setDemoError(e instanceof Error ? e.message : "Could not start the not-sure demo");
            }
          }}
          className="rounded-xl border border-[var(--color-border)] px-3 py-2.5 text-sm font-medium text-forest disabled:opacity-40"
        >
          Show “I&apos;m not sure”
        </button>
      </div>
      {demoError && (
        <p className="mt-2 text-center text-sm text-red-700">{demoError}</p>
      )}
    </div>
  );
}

function makeDarkJpeg(): Promise<File> {
  const canvas = document.createElement("canvas");
  canvas.width = 400;
  canvas.height = 400;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return Promise.reject(new Error("Could not draw a demo photo"));
  }
  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, 400, 400);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Could not build the retake demo photo"));
        return;
      }
      resolve(new File([blob], "too-dark.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.9);
  });
}

async function loadNotSureDemo(): Promise<File> {
  const res = await fetch("/demo/not-sure.jpg");
  if (!res.ok) {
    throw new Error("Demo photo is missing");
  }
  const blob = await res.blob();
  return new File([blob], "not-sure.jpg", { type: blob.type || "image/jpeg" });
}
