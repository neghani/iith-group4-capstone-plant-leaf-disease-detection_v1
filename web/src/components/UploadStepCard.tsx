"use client";

import { ImageUp } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { useLocale } from "@/i18n/LocaleProvider";
import { cropName } from "@/i18n";

// Cropped-PlantDoc crops used for real-photo checks. Values are names the
// classifier was trained on (PlantVillage); PlantDoc spells soybean "Soyabean".
const CROPS = [
  "Apple",
  "Bell pepper",
  "Blueberry",
  "Cherry",
  "Corn",
  "Grape",
  "Peach",
  "Potato",
  "Raspberry",
  "Soybean",
  "Squash",
  "Strawberry",
  "Tomato",
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
  const { locale, t } = useLocale();

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
        <h3 className="font-semibold text-forest">{t("upload.title")}</h3>
        <span className="rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
          {t("upload.step")}
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
            alt={t("upload.previewAlt")}
            className="mb-4 max-h-48 rounded-lg object-contain"
          />
        ) : (
          <span className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-sage">
            <ImageUp className="h-7 w-7 text-forest" />
          </span>
        )}
        <p className="font-semibold text-forest">
          {preview ? t("upload.changePhoto") : t("upload.drop")}
        </p>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          {t("upload.browse")}
        </p>
      </div>

      <label className="mt-6 block text-sm font-medium text-forest">
        {t("upload.cropLabel")}
        <select
          value={crop}
          onChange={(e) => setCrop(e.target.value)}
          className="mt-1 w-full rounded-xl border border-[var(--color-border)] bg-white px-3 py-2.5 text-sm"
        >
          <option value="">{t("upload.selectCrop")}</option>
          {CROPS.map((value) => (
            <option key={value} value={value}>
              {cropName(locale, value)}
            </option>
          ))}
          <option value={OTHER}>{t("upload.otherCrop")}</option>
        </select>
        <span className="mt-1 block text-xs font-normal text-[var(--color-text-muted)]">
          {t("upload.cropHint")}
        </span>
      </label>
      {crop === OTHER && (
        <label className="mt-4 block text-sm font-medium text-forest">
          {t("upload.whichCrop")}
          <input
            value={otherCrop}
            onChange={(e) => setOtherCrop(e.target.value)}
            placeholder={t("upload.whichPlaceholder")}
            className="mt-1 w-full rounded-xl border border-[var(--color-border)] px-3 py-2.5 text-sm"
          />
          <span className="mt-1 block text-xs font-normal text-[var(--color-text-muted)]">
            {t("upload.otherHint")}
          </span>
        </label>
      )}

      <label className="mt-4 block text-sm font-medium text-forest">
        {t("upload.symptomsLabel")}
        <textarea
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          placeholder={t("upload.symptomsPlaceholder")}
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
        {loading ? t("upload.analyzing") : t("upload.analyze")}
      </button>

      <p className="mt-5 text-xs font-medium uppercase tracking-wide text-[var(--color-forest-muted)]">
        {t("upload.demoEyebrow")}
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
              const key = e instanceof Error ? e.message : "";
              setDemoError(t(key.startsWith("upload.") ? key : "upload.retakeDemoError"));
            }
          }}
          className="rounded-xl border border-[var(--color-border)] px-3 py-2.5 text-sm font-medium text-forest disabled:opacity-40"
        >
          {t("upload.showRetake")}
        </button>
        <button
          type="button"
          disabled={loading}
          onClick={async () => {
            setDemoError(null);
            try {
              onAnalyze(await loadNotSureDemo(), "", "");
            } catch (e) {
              const key = e instanceof Error ? e.message : "";
              setDemoError(t(key.startsWith("upload.") ? key : "upload.unsureDemoError"));
            }
          }}
          className="rounded-xl border border-[var(--color-border)] px-3 py-2.5 text-sm font-medium text-forest disabled:opacity-40"
        >
          {t("upload.showUnsure")}
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
    return Promise.reject(new Error("upload.drawError"));
  }
  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, 400, 400);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("upload.buildError"));
        return;
      }
      resolve(new File([blob], "too-dark.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.9);
  });
}

async function loadNotSureDemo(): Promise<File> {
  const res = await fetch("/demo/not-sure.jpg");
  if (!res.ok) {
    throw new Error("upload.missingDemo");
  }
  const blob = await res.blob();
  return new File([blob], "not-sure.jpg", { type: blob.type || "image/jpeg" });
}
