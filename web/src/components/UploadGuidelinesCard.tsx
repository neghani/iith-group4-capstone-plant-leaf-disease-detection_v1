import { Camera, CheckCircle2, Droplets, Sun } from "lucide-react";

const tips = [
  {
    icon: Camera,
    title: "Keep the leaf in focus",
    body: "Hold the camera steady and let the leaf fill most of the frame.",
  },
  {
    icon: Sun,
    title: "Use natural light",
    body: "Avoid harsh shadows and bright glare on the leaf surface.",
  },
  {
    icon: Droplets,
    title: "Avoid wet leaves",
    body: "Water drops can hide spots and reduce diagnosis quality.",
  },
  {
    icon: CheckCircle2,
    title: "Upload one clear leaf",
    body: "JPEG, PNG, or WebP up to 25 MB. Large phone photos are shrunk automatically.",
  },
];

export function UploadGuidelinesCard() {
  return (
    <aside className="rounded-card bg-white p-6 shadow-card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="font-semibold text-forest">Photo guidelines</h3>
        <span className="rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
          Before upload
        </span>
      </div>

      <p className="text-sm text-[var(--color-text-muted)]">
        Follow these quick checks so the model can detect visible symptoms more
        reliably.
      </p>

      <ul className="mt-5 space-y-4">
        {tips.map((tip) => (
          <li key={tip.title} className="flex gap-3">
            <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-sage text-forest">
              <tip.icon className="h-4 w-4" />
            </span>
            <div>
              <p className="text-sm font-medium text-forest">{tip.title}</p>
              <p className="text-xs leading-relaxed text-[var(--color-text-muted)]">
                {tip.body}
              </p>
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}
