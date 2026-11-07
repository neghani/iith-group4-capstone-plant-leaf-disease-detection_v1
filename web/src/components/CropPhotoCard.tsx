"use client";

import { Crop } from "lucide-react";
import { useT } from "@/i18n/LocaleProvider";

function scaleBbox(
  bbox: number[],
  workSize: number[],
  displaySize: number[]
): number[] | null {
  if (bbox.length !== 4 || workSize.length !== 2 || displaySize.length !== 2) {
    return null;
  }
  const [w, h] = workSize;
  const [dw, dh] = displaySize;
  if (!w || !h || !dw || !dh) return null;
  const sx = dw / w;
  const sy = dh / h;
  return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy];
}

export function CropPhotoCard({
  src,
  sample,
  bbox,
  bboxSpace,
  imageSize,
}: {
  src: string;
  sample?: boolean;
  /** Leaf box chosen by the quality gate, in bboxSpace coordinates. */
  bbox?: number[] | null;
  /** Coordinate space of bbox (the size the gate measured). */
  bboxSpace?: number[] | null;
  /** Displayed image size (the original photo), so the overlay maps correctly. */
  imageSize?: number[] | null;
}) {
  const t = useT();
  const display = imageSize && imageSize[0] > 0 && imageSize[1] > 0 ? imageSize : null;
  const box = bbox && bboxSpace && display ? scaleBbox(bbox, bboxSpace, display) : null;
  const aspect = display ? `${display[0]} / ${display[1]}` : "4 / 3";

  let overlay: React.ReactNode = null;
  if (box) {
    const [x1, y1, x2, y2] = box;
    const left = Math.max(0, (x1 / display![0]) * 100);
    const top = Math.max(0, (y1 / display![1]) * 100);
    const width = Math.min(100 - left, ((x2 - x1) / display![0]) * 100);
    const height = Math.min(100 - top, ((y2 - y1) / display![1]) * 100);
    // Amber + dark outline — forest green vanishes on leaf photos.
    overlay = (
      <>
        <div
          aria-hidden
          className="absolute rounded-md border-[3px] border-dashed border-[#f5c518] shadow-[0_0_0_1px_rgba(0,0,0,0.75),0_0_0_9999px_rgba(0,0,0,0.12)]"
          style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
        />
        <span
          className="absolute inline-flex items-center gap-1 rounded-full bg-[#f5c518] px-2 py-0.5 text-[10px] font-semibold text-[#1a1a1a] shadow-[0_1px_2px_rgba(0,0,0,0.45)]"
          style={{ left: `calc(${left}% + 6px)`, top: `calc(${top}% + 6px)` }}
        >
          <Crop className="h-3 w-3" />
          {t("sources.analyzedRegion")}
        </span>
      </>
    );
  }

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
      <div
        className="relative w-full overflow-hidden rounded-xl bg-[var(--color-sage)]/30"
        style={{ aspectRatio: aspect }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={t("photo.alt")}
          className="h-full w-full object-cover"
        />
        {overlay}
      </div>
    </div>
  );
}