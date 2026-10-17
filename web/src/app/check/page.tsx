"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { AppShell } from "@/components/AppShell";
import { UploadGuidelinesCard } from "@/components/UploadGuidelinesCard";
import { UploadStepCard } from "@/components/UploadStepCard";
import { useLocale } from "@/i18n/LocaleProvider";
import { checkQuality, predictLeaf } from "@/lib/api";
import { translateError } from "@/lib/errors";
import { prepareUpload } from "@/lib/image";
import { qualityReportFromCheck } from "@/lib/qualityReport";
import { saveCheckSession } from "@/lib/session";
import type { CheckSession } from "@/lib/types";

type Stage = "idle" | "checking" | "analyzing";

export default function CheckPage() {
  const router = useRouter();
  const { locale, t } = useLocale();
  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze(file: File, crop: string, symptoms: string) {
    setError(null);
    try {
      setStage("checking");
      // Shrink big phone photos once; both API calls and the saved preview reuse the result.
      const { file: upload, previewDataUrl } = await prepareUpload(file);
      const quality = await checkQuality(upload);
      if (!quality.ok) {
        const report = qualityReportFromCheck(quality);
        const session: CheckSession = {
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          cropLabel: crop || "crop",
          previewDataUrl,
          report,
          sessionContext: {},
          topK: [],
          lowConfidence: true,
          qualityOk: false,
          qualityReason: quality.reason,
          language: locale,
        };
        saveCheckSession(session);
        router.push("/results");
        return;
      }

      setStage("analyzing");
      const data = await predictLeaf(upload, crop, symptoms, locale);
      const cropLabel = data.report.crop || crop || "crop";

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
        qualityReason: data.quality.reason,
        language: data.language === "te" ? "te" : "en",
        translationApplied: data.translation_applied === true,
      };

      saveCheckSession(session);
      router.push("/results");
    } catch (e) {
      setError(
        translateError(e instanceof Error ? e.message : "", t)
      );
    } finally {
      setStage("idle");
    }
  }

  return (
    <AppShell breadcrumb={t("check.breadcrumb")}>
      <p className="text-xs font-semibold tracking-widest text-[var(--color-forest-muted)]">
        {t("check.eyebrow")}
      </p>
      <h1 className="mt-2 font-serif text-4xl font-semibold text-forest">
        {t("check.title")}
      </h1>
      <p className="mt-3 max-w-xl text-[var(--color-text-muted)]">
        {t("check.subtitle")}
      </p>

      <div className="mt-10 grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px] xl:items-start">
        <div>
          <UploadStepCard onAnalyze={handleAnalyze} loading={stage !== "idle"} />
          {stage !== "idle" && (
            <p className="mt-4 text-center text-sm text-[var(--color-text-muted)]">
              {stage === "checking"
                ? t("check.checking")
                : t("check.analyzing")}
            </p>
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
