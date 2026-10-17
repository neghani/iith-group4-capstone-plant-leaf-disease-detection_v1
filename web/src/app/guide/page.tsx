"use client";

import { AppShell } from "@/components/AppShell";
import { useT } from "@/i18n/LocaleProvider";

const tips = [
  { title: "guide.tip1Title", body: "guide.tip1Body" },
  { title: "guide.tip2Title", body: "guide.tip2Body" },
  { title: "guide.tip3Title", body: "guide.tip3Body" },
  { title: "guide.tip4Title", body: "guide.tip4Body" },
];

export default function GuidePage() {
  const t = useT();
  return (
    <AppShell breadcrumb={t("guide.breadcrumb")}>
      <h1 className="font-serif text-3xl font-semibold text-forest">
        {t("guide.title")}
      </h1>
      <p className="mt-2 max-w-2xl text-[var(--color-text-muted)]">
        {t("guide.intro")}
      </p>

      <div className="mt-10 grid gap-4 md:grid-cols-2">
        {tips.map((tip) => (
          <div
            key={tip.title}
            className="rounded-card bg-white p-6 shadow-card"
          >
            <h3 className="font-semibold text-forest">{t(tip.title)}</h3>
            <p className="mt-2 text-sm text-[var(--color-text-muted)]">
              {t(tip.body)}
            </p>
          </div>
        ))}
      </div>
    </AppShell>
  );
}
