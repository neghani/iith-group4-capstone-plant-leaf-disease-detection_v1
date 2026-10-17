"use client";

import { useLocale } from "@/i18n/LocaleProvider";

export function LanguageToggle() {
  const { locale, setLocale, t } = useLocale();

  return (
    <div
      role="group"
      aria-label={t("lang.group")}
      className="flex shrink-0 rounded-full border border-[var(--color-border)] bg-white p-0.5 text-sm font-medium"
    >
      <button
        type="button"
        aria-pressed={locale === "en"}
        onClick={() => setLocale("en")}
        className={`rounded-full px-3 py-1 ${
          locale === "en" ? "bg-forest text-white" : "text-forest"
        }`}
      >
        English
      </button>
      <button
        type="button"
        aria-pressed={locale === "te"}
        onClick={() => setLocale("te")}
        className={`rounded-full px-3 py-1 ${
          locale === "te" ? "bg-forest text-white" : "text-forest"
        }`}
      >
        తెలుగు
      </button>
    </div>
  );
}
