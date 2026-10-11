import { ScanLine } from "lucide-react";
import type { DiagnosisReport, InferenceInfo, TopKPrediction } from "@/lib/types";

export function DiagnosisCard({
  report,
  lowConfidence,
  topK,
  inference,
}: {
  report: DiagnosisReport;
  lowConfidence?: boolean;
  topK?: TopKPrediction[];
  inference?: InferenceInfo;
}) {
  if (report.status === "quality_issue" || report.status === "error") {
    return (
      <div className="rounded-card bg-white p-6 shadow-card">
        <h3 className="font-serif text-xl font-semibold text-forest">
          {report.status_label}
        </h3>
        <p className="mt-3 text-[var(--color-text-muted)]">{report.description}</p>
        {report.actions?.length > 0 && (
          <ul className="mt-4 list-inside list-decimal text-sm text-forest">
            {report.actions.map((a, i) => (
              <li key={i}>{a.step}</li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-card bg-white p-6 shadow-card">
        <span className="inline-flex items-center gap-2 rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
          <ScanLine className="h-3.5 w-3.5" />
          {report.status_label}
        </span>

        <h2 className="mt-4 font-serif text-3xl font-semibold text-forest">
          {report.diagnosis_name}
        </h2>
        <p className="mt-2 text-[var(--color-text-muted)]">{report.description}</p>

        {lowConfidence && (
          <p className="mt-3 rounded-lg bg-sage/60 px-3 py-2 text-sm text-forest">
            {report.confidence_explanation}
          </p>
        )}

        {report.expert_confirmation_required && (
          <p className="mt-3 text-sm font-medium text-disclaimer-text">
            Needs confirmation — check with a local expert before major treatment.
          </p>
        )}

        {report.symptoms.length > 0 && (
          <div className="mt-6 border-t border-[var(--color-border)] pt-6">
            <h4 className="font-semibold text-forest">Signs that may fit</h4>
            <ul className="mt-3 space-y-2">
              {report.symptoms.map((s) => (
                <li key={s.name} className="text-sm">
                  <span className="font-medium text-forest">{s.name}</span>
                  <span className="text-[var(--color-text-muted)]">
                    {" "}
                    — {s.explanation}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* No disease_id = the crop is outside the model, so its matches would mislead. */}
        {report.disease_id && topK && topK.length > 0 && (
          <div className="mt-6 border-t border-[var(--color-border)] pt-6">
            <h4 className="font-semibold text-forest">Model&apos;s top matches</h4>
            <ul className="mt-3 space-y-2">
              {topK.map((t) => (
                <li key={t.label} className="text-sm">
                  <div className="flex justify-between text-forest">
                    <span>{t.display_name}</span>
                    <span>{Math.round(t.score * 100)}%</span>
                  </div>
                  <div className="mt-1 h-1.5 rounded-full bg-sage/60">
                    <div
                      className="h-1.5 rounded-full bg-forest"
                      style={{ width: `${Math.round(t.score * 100)}%` }}
                    />
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        {(!report.disease_id || !topK || topK.length === 0) &&
          report.alternatives.length > 0 && (
          <div className="mt-6">
            <h4 className="font-semibold text-forest">Other possibilities</h4>
            <p className="mt-2 text-sm text-[var(--color-text-muted)]">
              {report.alternatives.join(", ")}
            </p>
          </div>
        )}
      </div>

      {report.actions.length > 0 && (
        <div className="rounded-card bg-white p-6 shadow-card">
          <h4 className="font-semibold text-forest">What you can do now</h4>
          <ol className="mt-4 list-decimal space-y-3 pl-5 text-sm text-forest">
            {report.actions.map((a, i) => (
              <li key={i}>
                {a.step}
                {a.safety_note && (
                  <span className="mt-1 block text-xs text-[var(--color-text-muted)]">
                    Safety: {a.safety_note}
                  </span>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}

      {report.monitoring_guidance && (
        <div className="rounded-card bg-white p-6 shadow-card">
          <h4 className="font-semibold text-forest">Follow-up monitoring</h4>
          <p className="mt-2 text-sm text-[var(--color-text-muted)]">
            {report.monitoring_guidance}
          </p>
        </div>
      )}

      {report.initial_message && (
        <div className="rounded-card border border-sage bg-sage/40 p-5 text-sm text-forest">
          {report.initial_message}
        </div>
      )}

      {inference && (
        <p className="text-xs text-[var(--color-text-muted)]">
          Model: {inference.model_id} ({inference.model_revision})
          {inference.model_verified ? " · verified by server" : ""} ·{" "}
          {inference.latency_ms} ms
        </p>
      )}
    </div>
  );
}
