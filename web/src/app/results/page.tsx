"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ChatPanel } from "@/components/ChatPanel";
import { CropPhotoCard } from "@/components/CropPhotoCard";
import { DiagnosisCard } from "@/components/DiagnosisCard";
import { DisclaimerBanner } from "@/components/DisclaimerBanner";
import { predictionChatContext } from "@/lib/chatContext";
import { getCurrentCheck } from "@/lib/session";
import { cropName } from "@/i18n";
import { useLocale } from "@/i18n/LocaleProvider";
import type { CheckSession } from "@/lib/types";

/** The quality gate records the leaf box it analyzed; numbers come from /predict. */
function leafOverlay(session: CheckSession): {
  bbox?: number[] | null;
  bboxSpace?: number[] | null;
  imageSize?: number[] | null;
} {
  const prep = session.inference?.preprocessing as Record<string, unknown> | undefined;
  if (!prep) return {};
  const pick = (key: string): number[] | null => {
    const value = prep[key];
    return Array.isArray(value) && value.every((n) => typeof n === "number")
      ? (value as number[])
      : null;
  };
  return {
    bbox: pick("chosen_bbox"),
    bboxSpace: pick("work_size"),
    imageSize: pick("original_size"),
  };
}

export default function ResultsPage() {
  const router = useRouter();
  const { locale, t } = useLocale();
  const [session, setSession] = useState<CheckSession | null>(null);

  useEffect(() => {
    const current = getCurrentCheck();
    if (!current) {
      router.replace("/check");
      return;
    }
    setSession(current);
  }, [router]);

  if (!session) {
    return (
      <AppShell breadcrumb={t("results.breadcrumb")}>
        <p className="text-[var(--color-text-muted)]">{t("results.loading")}</p>
      </AppShell>
    );
  }

  // Prefer the model's predicted crop over the farmer hint when they differ.
  const cropForHeading = session.report.crop || session.cropLabel;
  const cropShown =
    locale === "en"
      ? cropName(locale, cropForHeading).toLowerCase()
      : cropName(locale, cropForHeading);
  const status = session.report.status;
  const heading =
    status === "quality_issue"
      ? t("results.headingQuality")
      : status === "not_sure"
        ? t("results.headingUnsure")
        : t("results.heading", { crop: cropShown });
  const sub =
    status === "quality_issue"
      ? t("results.subQuality")
      : status === "not_sure"
        ? t("results.subUnsure")
        : t("results.subOk");
  const showChat =
    session.qualityOk &&
    status !== "quality_issue" &&
    status !== "error" &&
    status !== "not_sure";

  return (
    <AppShell breadcrumb={t("results.breadcrumb")}>
      <Link
        href="/check"
        className="text-sm font-medium text-[var(--color-forest-muted)] hover:text-forest"
      >
        {t("results.back")}
      </Link>

      <h1 className="mt-4 font-serif text-4xl font-semibold text-forest">
        {heading}
      </h1>
      <p className="mt-2 text-[var(--color-text-muted)]">{sub}</p>

      <div className="mt-6">
        <DisclaimerBanner
          extra={
            status === "quality_issue"
              ? t("quality.limitations")
              : session.report.diagnostic_limitations
          }
        />
      </div>

      {session.language === "te" && session.translationApplied === false && (
        <p className="mt-4 text-sm text-[var(--color-text-muted)]">
          {t("results.translationUnavailable")}
        </p>
      )}

      <div className="mt-8 flex flex-col gap-6 lg:flex-row lg:items-start">
        <div className="w-full max-w-md shrink-0 lg:max-w-[22rem] xl:max-w-md">
          <CropPhotoCard src={session.previewDataUrl} {...leafOverlay(session)} />
        </div>
        <div className="min-w-0 flex-1">
          <DiagnosisCard
            report={session.report}
            lowConfidence={session.lowConfidence}
            topK={session.topK}
            inference={session.inference}
            qualityReason={session.qualityReason}
          />
        </div>
      </div>

      {showChat && (
        <ChatPanel
          key={session.id}
          sessionContext={predictionChatContext(session)}
        />
      )}

      <p className="mt-8 text-xs text-[var(--color-text-muted)]">
        {t("results.footer")}
      </p>
    </AppShell>
  );
}
