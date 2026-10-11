"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { AppShell } from "@/components/AppShell";
import { UploadGuidelinesCard } from "@/components/UploadGuidelinesCard";
import { UploadStepCard } from "@/components/UploadStepCard";
import { predictLeaf } from "@/lib/api";
import { saveCheckSession } from "@/lib/session";
import type { CheckSession } from "@/lib/types";

export default function CheckPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze(file: File, crop: string, symptoms: string) {
    setLoading(true);
    setError(null);
    try {
      const previewDataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      const data = await predictLeaf(file, crop, symptoms);
      const cropLabel =
        data.report.crop || crop || "crop";

      const session: CheckSession = {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        cropLabel,
        previewDataUrl,
        report: data.report,
        sessionContext: data.session_context,
        lowConfidence: data.low_confidence,
        qualityOk: data.quality.ok,
      };

      saveCheckSession(session);
      router.push("/results");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
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
          <UploadStepCard onAnalyze={handleAnalyze} loading={loading} />
          {error && (
            <p className="mt-4 text-center text-sm text-red-700">{error}</p>
          )}
        </div>
        <UploadGuidelinesCard />
      </div>
    </AppShell>
  );
}
