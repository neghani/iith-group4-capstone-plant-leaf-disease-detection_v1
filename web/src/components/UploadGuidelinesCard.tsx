"use client";

import { Camera, CheckCircle2, Droplets, Sun } from "lucide-react";
import { useT } from "@/i18n/LocaleProvider";

const tips = [
  { icon: Camera, title: "guidelines.focusTitle", body: "guidelines.focusBody" },
  { icon: Sun, title: "guidelines.lightTitle", body: "guidelines.lightBody" },
  { icon: Droplets, title: "guidelines.wetTitle", body: "guidelines.wetBody" },
  { icon: CheckCircle2, title: "guidelines.oneTitle", body: "guidelines.oneBody" },
];

export function UploadGuidelinesCard() {
  const t = useT();

  return (
    <aside className="rounded-card bg-white p-6 shadow-card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-semibold text-forest">{t("guidelines.title")}</h3>
        <span className="rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
          {t("guidelines.badge")}
        </span>
      </div>

      <p className="text-sm text-[var(--color-text-muted)]">{t("guidelines.intro")}</p>

      <ul className="mt-5 space-y-4">
        {tips.map((tip) => (
          <li key={tip.title} className="flex gap-3">
            <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-sage text-forest">
              <tip.icon className="h-4 w-4" />
            </span>
            <div>
              <p className="text-sm font-medium text-forest">{t(tip.title)}</p>
              <p className="text-xs leading-relaxed text-[var(--color-text-muted)]">
                {t(tip.body)}
              </p>
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}
