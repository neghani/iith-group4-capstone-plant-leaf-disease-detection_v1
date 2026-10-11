import type { CheckSession } from "./types";

/** Compact leaf-check facts sent with every follow-up, including short ones. */
export function predictionChatContext(
  session: CheckSession
): Record<string, unknown> {
  const stored = session.sessionContext ?? {};
  const storedTop = stored.top_k;
  const topK =
    Array.isArray(storedTop) && storedTop.length > 0
      ? storedTop
      : (session.topK ?? []);

  return {
    disease_id: stored.disease_id ?? session.report.disease_id ?? null,
    diagnosis_name: stored.diagnosis_name ?? session.report.diagnosis_name,
    crop: stored.crop ?? session.report.crop ?? session.cropLabel,
    confidence: stored.confidence ?? session.report.confidence,
    status: stored.status ?? session.report.status,
    low_confidence: stored.low_confidence ?? session.lowConfidence,
    top_k: topK.slice(0, 3),
    summary: stored.summary ?? session.report.initial_message,
  };
}

export function predictionContextLabel(ctx: Record<string, unknown>): string {
  const crop = typeof ctx.crop === "string" && ctx.crop ? ctx.crop : "";
  const name =
    typeof ctx.diagnosis_name === "string" && ctx.diagnosis_name
      ? ctx.diagnosis_name
      : "";
  const conf =
    typeof ctx.confidence === "number"
      ? `${Math.round(ctx.confidence * 100)}%`
      : "";
  return [crop, name, conf].filter(Boolean).join(" · ");
}
