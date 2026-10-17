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

  const cropShown =
    locale === "en"
      ? cropName(locale, session.cropLabel).toLowerCase()
      : cropName(locale, session.cropLabel);
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

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <CropPhotoCard src={session.previewDataUrl} />
        <DiagnosisCard
          report={session.report}
          lowConfidence={session.lowConfidence}
          topK={session.topK}
          inference={session.inference}
          qualityReason={session.qualityReason}
        />
      </div>

      {showChat && (
        <div className="mt-8">
          <ChatPanel
            key={session.id}
            sessionContext={predictionChatContext(session)}
          />
        </div>
      )}

      <p className="mt-8 text-xs text-[var(--color-text-muted)]">
        {t("results.footer")}
      </p>
    </AppShell>
  );
}
