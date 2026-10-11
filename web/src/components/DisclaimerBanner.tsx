import { Info } from "lucide-react";

const DEFAULT_COPY =
  "Illustrative example only — this is not a verified diagnosis. Similar leaf spots can have different causes.";

export function DisclaimerBanner({ extra }: { extra?: string }) {
  return (
    <div className="flex gap-3 rounded-card border border-[#f0e0c8] bg-disclaimer px-4 py-3 text-sm text-disclaimer-text">
      <Info className="mt-0.5 h-4 w-4 shrink-0" />
      <p>
        {DEFAULT_COPY}
        {extra ? ` ${extra}` : null}
      </p>
    </div>
  );
}
