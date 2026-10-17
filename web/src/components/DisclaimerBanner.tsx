"use client";

import { Info } from "lucide-react";
import { useT } from "@/i18n/LocaleProvider";

export function DisclaimerBanner({ extra }: { extra?: string }) {
  const t = useT();
  return (
    <div className="flex gap-3 rounded-card border border-[#f0e0c8] bg-disclaimer px-4 py-3 text-sm text-disclaimer-text">
      <Info className="mt-0.5 h-4 w-4 shrink-0" />
      <p>
        {t("disclaimer")}
        {extra ? ` ${extra}` : null}
      </p>
    </div>
  );
}
