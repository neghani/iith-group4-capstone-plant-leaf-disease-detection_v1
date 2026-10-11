import { AppShell } from "@/components/AppShell";

const tips = [
  {
    title: "Take clear leaf photos",
    body: "Use daylight, hold steady, and fill the frame with the affected leaf.",
  },
  {
    title: "Check lower leaves first",
    body: "Many fungal issues start on older foliage before spreading upward.",
  },
  {
    title: "Confirm before spraying",
    body: "Match symptoms over a few days and ask your local extension office before costly treatment.",
  },
  {
    title: "Rotate and sanitize",
    body: "Remove heavily infected plant material and avoid working in wet fields to limit spread.",
  },
];

export default function GuidePage() {
  return (
    <AppShell breadcrumb="Growing guide">
      <h1 className="font-serif text-3xl font-semibold text-forest">
        Growing guide
      </h1>
      <p className="mt-2 max-w-2xl text-[var(--color-text-muted)]">
        Practical habits for healthier crops. This demo guide supports the
        CropCare AI capstone — not a substitute for regional agronomy advice.
      </p>

      <div className="mt-10 grid gap-4 md:grid-cols-2">
        {tips.map((tip) => (
          <div
            key={tip.title}
            className="rounded-card bg-white p-6 shadow-card"
          >
            <h3 className="font-semibold text-forest">{tip.title}</h3>
            <p className="mt-2 text-sm text-[var(--color-text-muted)]">
              {tip.body}
            </p>
          </div>
        ))}
      </div>
    </AppShell>
  );
}
