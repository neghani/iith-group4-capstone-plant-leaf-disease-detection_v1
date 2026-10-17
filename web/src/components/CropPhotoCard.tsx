"use client";

import { useT } from "@/i18n/LocaleProvider";

export function CropPhotoCard({
  src,
  sample,
}: {
  src: string;
  sample?: boolean;
}) {
  const t = useT();
  return (
    <div className="rounded-card bg-white p-5 shadow-card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-semibold text-forest">{t("photo.title")}</h3>
        {sample && (
          <span className="rounded-full bg-sage px-3 py-0.5 text-xs font-medium text-forest">
            {t("photo.sample")}
          </span>
        )}
      </div>
      <div className="relative aspect-[4/3] overflow-hidden rounded-xl bg-[var(--color-sage)]/30">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={t("photo.alt")}
          className="h-full w-full object-cover"
        />
      </div>
    </div>
  );
}
