"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { cropName } from "@/i18n";
import { useLocale } from "@/i18n/LocaleProvider";
import {
  getCheckHistory,
  setCurrentFromHistory,
} from "@/lib/session";
import type { CheckSession } from "@/lib/types";

export default function HistoryPage() {
  const router = useRouter();
  const { locale, t } = useLocale();
  const [items, setItems] = useState<CheckSession[]>([]);

  useEffect(() => {
    setItems(getCheckHistory());
  }, []);

  function openCheck(session: CheckSession) {
    setCurrentFromHistory(session);
    router.push("/results");
  }

  return (
    <AppShell breadcrumb={t("history.breadcrumb")}>
      <h1 className="font-serif text-3xl font-semibold text-forest">
        {t("history.title")}
      </h1>
      <p className="mt-2 text-sm text-[var(--color-text-muted)]">
        {t("history.subtitle")}
      </p>

      {items.length === 0 ? (
        <div className="mt-10 rounded-card bg-white p-8 text-center shadow-card">
          <p className="text-[var(--color-text-muted)]">{t("history.empty")}</p>
          <Link
            href="/check"
            className="mt-4 inline-block rounded-xl bg-forest px-5 py-2.5 text-sm font-semibold text-white"
          >
            {t("history.cta")}
          </Link>
        </div>
      ) : (
        <ul className="mt-8 space-y-3">
          {items.map((item) => (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => openCheck(item)}
                className="flex w-full items-center gap-4 rounded-card bg-white p-4 text-left shadow-card transition hover:ring-2 hover:ring-sage"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={item.previewDataUrl}
                  alt=""
                  className="h-16 w-16 rounded-lg object-cover"
                />
                <div>
                  <p className="font-semibold text-forest">
                    {item.report.diagnosis_name}
                  </p>
                  <p className="text-xs text-[var(--color-text-muted)]">
                    {new Date(item.timestamp).toLocaleString(
                      locale === "te" ? "te-IN" : undefined
                    )}{" "}
                    · {cropName(locale, item.cropLabel)}
                  </p>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </AppShell>
  );
}
