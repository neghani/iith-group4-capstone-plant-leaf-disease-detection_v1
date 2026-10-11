"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ChatPanel } from "@/components/ChatPanel";
import { CropPhotoCard } from "@/components/CropPhotoCard";
import { DiagnosisCard } from "@/components/DiagnosisCard";
import { DisclaimerBanner } from "@/components/DisclaimerBanner";
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

  return (
    <AppShell breadcrumb="Check a crop / Your results">
      <Link
        href="/check"
        className="text-sm font-medium text-[var(--color-forest-muted)] hover:text-forest"
      >
        ← Back to crop upload
      </Link>

      <h1 className="mt-4 font-serif text-4xl font-semibold text-forest">
        Your {cropWord} leaf check
      </h1>
      <p className="mt-2 text-[var(--color-text-muted)]">
        Understand the signs. Start with safer, practical care.
      </p>

      <div className="mt-6">
        <DisclaimerBanner extra={session.report.diagnostic_limitations} />
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <CropPhotoCard src={session.previewDataUrl} />
        <DiagnosisCard
          report={session.report}
          lowConfidence={session.lowConfidence}
        />
      </div>

      {session.qualityOk && session.report.status !== "quality_issue" && (
        <div className="mt-8">
          <ChatPanel sessionContext={session.sessionContext} />
        </div>
      )}

      <p className="mt-8 text-xs text-[var(--color-text-muted)]">
        Demo capstone — photos stay in your browser only. Confirm important
        decisions with a local agricultural expert.
      </p>
    </AppShell>
  );
}
