export function CropPhotoCard({
  src,
  sample,
}: {
  src: string;
  sample?: boolean;
}) {
  return (
    <div className="rounded-card bg-white p-5 shadow-card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-semibold text-forest">Your crop photo</h3>
        {sample && (
          <span className="rounded-full bg-sage px-3 py-0.5 text-xs font-medium text-forest">
            Sample
          </span>
        )}
      </div>
      <div className="relative aspect-[4/3] overflow-hidden rounded-xl bg-[var(--color-sage)]/30">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt="Uploaded crop leaf"
          className="h-full w-full object-cover"
        />
      </div>
    </div>
  );
}
