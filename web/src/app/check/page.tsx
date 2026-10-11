"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { AppShell } from "@/components/AppShell";
import { UploadGuidelinesCard } from "@/components/UploadGuidelinesCard";
import { UploadStepCard } from "@/components/UploadStepCard";
import { checkQuality, predictLeaf } from "@/lib/api";
import { prepareUpload } from "@/lib/image";
import { saveCheckSession } from "@/lib/session";
import type { CheckSession } from "@/lib/types";

type Stage = "idle" | "checking" | "analyzing";

export default function CheckPage() {
  const router = useRouter();
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [retakeMessage, setRetakeMessage] = useState<string | null>(null);

  async function handleAnalyze(file: File, crop: string, symptoms: string) {
    setError(null);
    setRetakeMessage(null);
    try {
      setStage("checking");
      // Shrink big phone photos once; both API calls and the saved preview reuse the result.
      const { file: upload, previewDataUrl } = await prepareUpload(file);
      const quality = await checkQuality(upload);
      if (!quality.ok) {
        setRetakeMessage(
          quality.message || "Please retake the photo with the leaf in focus."
        );
        return;
      }

      setStage("analyzing");
      const data = await predictLeaf(upload, crop, symptoms);
      const cropLabel =
        data.report.crop || crop || "crop";

      const session: CheckSession = {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        cropLabel,
        previewDataUrl,
        report: data.report,
        sessionContext: data.session_context,
        topK: data.top_k,
        lowConfidence: data.low_confidence,
        qualityOk: data.quality.ok,
        inference: data.inference ?? undefined,
      };

      saveCheckSession(session);
      router.push("/results");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setStage("idle");
    }
  }

  return (
    <AppShell breadcrumb="Check a crop">
      <p className="text-xs font-semibold tracking-widest text-[var(--color-forest-muted)]">
        HEALTHIER CROPS, ONE PHOTO AT A TIME
      </p>
      <h1 className="mt-2 font-serif text-4xl font-semibold text-forest">
        Let&apos;s take a look at your crop
      </h1>
      <p className="mt-3 max-w-xl text-[var(--color-text-muted)]">
        Upload a leaf photo. Get possible causes and simple steps to help your
        plant.
      </p>

      <div className="mt-10 grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px] xl:items-start">
        <div>
          <UploadStepCard onAnalyze={handleAnalyze} loading={stage !== "idle"} />
          {stage !== "idle" && (
            <p className="mt-4 text-center text-sm text-[var(--color-text-muted)]">
              {stage === "checking"
                ? "Checking photo quality…"
                : "Photo looks good. Analyzing the leaf…"}
            </p>
          )}
          {retakeMessage && (
            <div className="mt-4 rounded-card border border-sage bg-sage/40 p-4 text-sm text-forest">
              <p className="font-semibold">Photo needs improvement</p>
              <p className="mt-1">{retakeMessage}</p>
              <p className="mt-1 text-[var(--color-text-muted)]">
                Retake the photo with the leaf centered, in daylight, and in focus.
              </p>
            </div>
          )}
          {error && (
            <p className="mt-4 text-center text-sm text-red-700">{error}</p>
          )}
        </div>
        <UploadGuidelinesCard />
      </div>
    </AppShell>
  );
}
