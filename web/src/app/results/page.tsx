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
import type { CheckSession } from "@/lib/types";

export default function ResultsPage() {
  const router = useRouter();
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
      <AppShell breadcrumb="Check a crop / Your results">
        <p className="text-[var(--color-text-muted)]">Loading…</p>
      </AppShell>
    );
  }

  const cropWord = session.cropLabel.toLowerCase().includes("tomato")
    ? "tomato"
    : session.cropLabel.toLowerCase();
  const status = session.report.status;
  const heading =
    status === "quality_issue"
      ? "This photo needs a retake"
      : status === "not_sure"
        ? "I'm not confident yet"
        : `Your ${cropWord} leaf check`;
  const sub =
    status === "quality_issue"
      ? "The quality gate stopped before the disease model. Retake so the leaf is clear."
      : status === "not_sure"
        ? "The model saw the leaf but the score is too low to name a disease."
        : "Understand the signs. Start with safer, practical care.";
  const showChat =
    session.qualityOk &&
    status !== "quality_issue" &&
    status !== "error" &&
    status !== "not_sure";

  return (
    <AppShell breadcrumb="Check a crop / Your results">
      <Link
        href="/check"
        className="text-sm font-medium text-[var(--color-forest-muted)] hover:text-forest"
      >
        ← Back to crop upload
      </Link>

      <h1 className="mt-4 font-serif text-4xl font-semibold text-forest">
        {heading}
      </h1>
      <p className="mt-2 text-[var(--color-text-muted)]">{sub}</p>

      <div className="mt-6">
        <DisclaimerBanner extra={session.report.diagnostic_limitations} />
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <CropPhotoCard src={session.previewDataUrl} />
        <DiagnosisCard
          report={session.report}
          lowConfidence={session.lowConfidence}
          topK={session.topK}
          inference={session.inference}
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
        Demo capstone — photos stay in your browser only. Confirm important
        decisions with a local agricultural expert.
      </p>
    </AppShell>
  );
}
