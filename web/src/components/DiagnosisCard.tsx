"use client";

import Link from "next/link";
import { ScanLine } from "lucide-react";
import { MarkdownMessage } from "@/components/MarkdownMessage";
import { useT } from "@/i18n/LocaleProvider";
import { en } from "@/i18n";
import type { DiagnosisReport, InferenceInfo, TopKPrediction } from "@/lib/types";

function qualityDetail(
  reason: string | null | undefined,
  fallback: string,
  t: (key: string) => string
): string {
  if (reason) {
    const key = `quality.${reason}`;
    const text = t(key);
    if (text !== key) return text;
  }
  const known = Object.entries(en.quality).find(([, value]) => value === fallback);
  if (known) return t(`quality.${known[0]}`);
  return fallback || t("quality.fallback");
}

function RetakeButton() {
  const t = useT();
  return (
    <Link
      href="/check"
      className="mt-5 inline-flex rounded-xl bg-forest px-4 py-2.5 text-sm font-semibold text-white"
    >
      {t("diagnosis.retakeButton")}
    </Link>
  );
}

function Top1Confidence({
  top,
  unsure,
}: {
  top: TopKPrediction;
  unsure?: boolean;
}) {
  const t = useT();
  const percent = Math.round(top.score * 100);
  return (
    <div className="mt-6 border-t border-[var(--color-border)] pt-6">
      {unsure && (
        <p className="text-xs text-[var(--color-text-muted)]">
          {t("diagnosis.possibilitiesOnly")}
        </p>
      )}
      <p className={`text-sm font-medium text-forest ${unsure ? "mt-2" : ""}`}>
        {t("diagnosis.confidence", {
          label: top.display_name,
          percent: String(percent),
        })}
      </p>
      <div className="mt-2 h-1.5 max-w-xs rounded-full bg-sage/60">
        <div
          className="h-1.5 rounded-full bg-forest"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

export function DiagnosisCard({
  report,
  lowConfidence,
  topK,
  inference,
  qualityReason,
}: {
  report: DiagnosisReport;
  lowConfidence?: boolean;
  topK?: TopKPrediction[];
  inference?: InferenceInfo;
  qualityReason?: string | null;
}) {
  const t = useT();
  const top1 = topK?.[0];
  const badge =
    report.status === "quality_issue"
      ? t("diagnosis.qualityBadge")
      : report.status === "not_sure"
        ? t("diagnosis.unsureBadge")
        : report.status === "error"
          ? t("diagnosis.errorBadge")
          : report.status_label;
  const modelLine = inference
    ? t("diagnosis.model", {
        id: inference.model_id,
        revision: inference.model_revision,
        verified: inference.model_verified ? t("diagnosis.verified") : "",
        ms: inference.latency_ms,
      })
    : null;

  if (report.status === "quality_issue" || report.status === "error") {
    return (
      <div className="rounded-card bg-white p-6 shadow-card">
        <span className="inline-flex items-center gap-2 rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
          <ScanLine className="h-3.5 w-3.5" />
          {badge}
        </span>
        <h2 className="mt-4 font-serif text-3xl font-semibold text-forest">
          {report.status === "quality_issue"
            ? t("diagnosis.retakeTitle")
            : report.diagnosis_name}
        </h2>
        <p className="mt-3 text-[var(--color-text-muted)]">
          {report.status === "quality_issue"
            ? qualityDetail(qualityReason, report.description, t)
            : report.description}
        </p>
        {report.status === "quality_issue" ? (
          <ul className="mt-4 list-inside list-decimal text-sm text-forest">
            <li>{t("quality.retakeStep")}</li>
          </ul>
        ) : (
          report.actions?.length > 0 && (
            <ul className="mt-4 list-inside list-decimal text-sm text-forest">
              {report.actions.map((a, i) => (
                <li key={i}>{a.step}</li>
              ))}
            </ul>
          )
        )}
        {report.status === "quality_issue" && <RetakeButton />}
      </div>
    );
  }

  if (report.status === "not_sure") {
    return (
      <div className="space-y-6">
        <div className="rounded-card bg-white p-6 shadow-card">
          <span className="inline-flex items-center gap-2 rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
            <ScanLine className="h-3.5 w-3.5" />
            {badge}
          </span>
          <h2 className="mt-4 font-serif text-3xl font-semibold text-forest">
            {t("diagnosis.unsureTitle")}
          </h2>
          <p className="mt-3 text-[var(--color-text-muted)]">{report.description}</p>
          <p className="mt-3 rounded-lg bg-sage/60 px-3 py-2 text-sm text-forest">
            {report.confidence_explanation}
          </p>
          {top1 && <Top1Confidence top={top1} unsure />}
          <RetakeButton />
        </div>
        {report.actions.length > 0 && (
          <div className="rounded-card bg-white p-6 shadow-card">
            <h4 className="font-semibold text-forest">{t("diagnosis.actions")}</h4>
            <ol className="mt-4 list-decimal space-y-3 pl-5 text-sm text-forest">
              {report.actions.map((a, i) => (
                <li key={i}>{a.step}</li>
              ))}
            </ol>
          </div>
        )}
        {modelLine && (
          <p className="text-xs text-[var(--color-text-muted)]">{modelLine}</p>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-card bg-white p-6 shadow-card">
        <span className="inline-flex items-center gap-2 rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
          <ScanLine className="h-3.5 w-3.5" />
          {badge}
        </span>

        <h2 className="mt-4 font-serif text-3xl font-semibold text-forest">
          {report.diagnosis_name}
        </h2>
        <p className="mt-2 text-[var(--color-text-muted)]">{report.description}</p>

        {top1 && <Top1Confidence top={top1} />}

        {lowConfidence && (
          <p className="mt-3 rounded-lg bg-sage/60 px-3 py-2 text-sm text-forest">
            {report.confidence_explanation}
          </p>
        )}

        {report.expert_confirmation_required && (
          <p className="mt-3 text-sm font-medium text-disclaimer-text">
            {t("diagnosis.confirm")}
          </p>
        )}

        {report.symptoms.length > 0 && (
          <div className="mt-6 border-t border-[var(--color-border)] pt-6">
            <h4 className="font-semibold text-forest">{t("diagnosis.signs")}</h4>
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
      </div>

      {report.actions.length > 0 && (
        <div className="rounded-card bg-white p-6 shadow-card">
          <h4 className="font-semibold text-forest">{t("diagnosis.actions")}</h4>
          <ol className="mt-4 list-decimal space-y-3 pl-5 text-sm text-forest">
            {report.actions.map((a, i) => (
              <li key={i}>
                {a.step}
                {a.safety_note && (
                  <span className="mt-1 block text-xs text-[var(--color-text-muted)]">
                    {t("diagnosis.safety")} {a.safety_note}
                  </span>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}

      {report.monitoring_guidance && (
        <div className="rounded-card bg-white p-6 shadow-card">
          <h4 className="font-semibold text-forest">{t("diagnosis.monitoring")}</h4>
          <p className="mt-2 text-sm text-[var(--color-text-muted)]">
            {report.monitoring_guidance}
          </p>
        </div>
      )}

      {report.initial_message && (
        <div className="rounded-card border border-sage bg-sage/40 p-5">
          <MarkdownMessage content={report.initial_message} />
        </div>
      )}

      {modelLine && (
        <p className="text-xs text-[var(--color-text-muted)]">{modelLine}</p>
      )}
    </div>
  );
}
