"use client";

import Link from "next/link";
import { BookOpen, Cpu, ScanLine, Sparkles } from "lucide-react";
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

function SourceChip({
  icon,
  label,
}: {
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-[var(--color-border)] bg-[var(--color-sage)]/40 px-2.5 py-1 text-[11px] font-medium text-forest">
      {icon}
      {label}
    </span>
  );
}

function ZoneTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
      {children}
    </h3>
  );
}

function modelChipText(
  inference: InferenceInfo | undefined,
  t: (key: string, vars?: Record<string, string | number>) => string
): string | null {
  if (!inference) return null;
  return t("diagnosis.model", {
    id: inference.model_id,
    revision: inference.model_revision,
    verified: inference.model_verified ? t("diagnosis.verified") : "",
    ms: inference.latency_ms,
  });
}

function ModelChip({ inference }: { inference?: InferenceInfo }) {
  const t = useT();
  const text = modelChipText(inference, t);
  if (!text) return null;
  return <SourceChip icon={<Cpu className="h-3.5 w-3.5" />} label={text} />;
}

/** Single top-1 bar — enough when the model named a disease. */
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
    <div className="mt-6">
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
      <div className="mt-2 h-2 max-w-sm rounded-full bg-sage/60">
        <div
          className="h-2 rounded-full bg-forest"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Full top-k list — only for the unsure path, where no disease was chosen and
 * the farmer needs to see every close guess.
 */
function TopKBars({ topK }: { topK: TopKPrediction[] }) {
  const t = useT();
  if (topK.length === 0) return null;
  const [top1, ...rest] = topK;
  return (
    <div className="mt-6 border-t border-[var(--color-border)] pt-5">
      <p className="text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
        {t("diagnosis.topMatches")}
      </p>
      <Top1Confidence top={top1} unsure />
      {rest.length > 0 && (
        <ul className="mt-4 space-y-2.5">
          {rest.map((p) => {
            const percent = Math.round(p.score * 100);
            return (
              <li key={p.label} className="max-w-sm">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-sm text-[var(--color-text-muted)]">
                    {p.display_name}
                  </span>
                  <span className="text-xs text-[var(--color-text-muted)]">
                    {percent}%
                  </span>
                </div>
                <div className="mt-1 h-1.5 rounded-full bg-sage/60">
                  <div
                    className="h-1.5 rounded-full bg-[var(--color-forest-muted)]"
                    style={{ width: `${percent}%` }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      )}
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
  const badge =
    report.status === "quality_issue"
      ? t("diagnosis.qualityBadge")
      : report.status === "not_sure"
        ? t("diagnosis.unsureBadge")
        : report.status === "error"
          ? t("diagnosis.errorBadge")
          : report.status_label;

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
          <div className="flex flex-wrap items-center justify-between gap-2">
            <ZoneTitle>{t("sources.modelZone")}</ZoneTitle>
            <ModelChip inference={inference} />
          </div>
          <span className="mt-4 inline-flex items-center gap-2 rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
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
          {topK && topK.length > 0 ? (
            <TopKBars topK={topK} />
          ) : report.alternatives.length > 0 ? (
            <div className="mt-4">
              <p className="text-xs text-[var(--color-text-muted)]">
                {t("diagnosis.possibilitiesOnly")}
              </p>
              <ul className="mt-2 list-inside list-disc text-sm text-forest">
                {report.alternatives.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </div>
          ) : null}
          <RetakeButton />
        </div>
        {report.actions.length > 0 && (
          <div className="rounded-card bg-white p-6 shadow-card">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <ZoneTitle>{t("sources.guidanceZone")}</ZoneTitle>
              <SourceChip
                icon={<BookOpen className="h-3.5 w-3.5" />}
                label={t("sources.appRulesChip")}
              />
            </div>
            <ol className="mt-4 list-decimal space-y-3 pl-5 text-sm text-forest">
              {report.actions.map((a, i) => (
                <li key={i}>{a.step}</li>
              ))}
            </ol>
          </div>
        )}
      </div>
    );
  }

  const hasGuidance =
    report.description ||
    report.symptoms.length > 0 ||
    report.actions.length > 0 ||
    report.monitoring_guidance;

  const topMatch = topK?.[0] ?? {
    label: report.disease_id ?? "",
    display_name: report.diagnosis_name,
    score: report.confidence,
  };

  return (
    <div className="space-y-6">
      {/* Zone 1 — what the image classifier produced */}
      <section className="rounded-card bg-white p-6 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <ZoneTitle>{t("sources.modelZone")}</ZoneTitle>
          <ModelChip inference={inference} />
        </div>
        <span className="mt-4 inline-flex items-center gap-2 rounded-full bg-sage px-3 py-1 text-xs font-medium text-forest">
          <ScanLine className="h-3.5 w-3.5" />
          {badge}
        </span>

        <h2 className="mt-4 font-serif text-3xl font-semibold text-forest">
          {report.diagnosis_name}
        </h2>

        <Top1Confidence top={topMatch} />

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
      </section>

      {/* Zone 2 — curated reference content, not generated per photo */}
      {hasGuidance && (
        <section className="rounded-card bg-white p-6 shadow-card">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <ZoneTitle>{t("sources.guidanceZone")}</ZoneTitle>
            <SourceChip
              icon={<BookOpen className="h-3.5 w-3.5" />}
              label={t("sources.referenceChip")}
            />
          </div>

          {report.description && (
            <p className="mt-4 text-[var(--color-text-muted)]">{report.description}</p>
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

          {report.actions.length > 0 && (
            <div className="mt-6 border-t border-[var(--color-border)] pt-6">
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
            <div className="mt-6 border-t border-[var(--color-border)] pt-6">
              <h4 className="font-semibold text-forest">{t("diagnosis.monitoring")}</h4>
              <p className="mt-2 text-sm text-[var(--color-text-muted)]">
                {report.monitoring_guidance}
              </p>
            </div>
          )}
        </section>
      )}

      {/* Zone 3 — the only LLM-written text on this screen */}
      {report.initial_message && (
        <div className="rounded-card border border-sage bg-sage/40 p-5">
          <SourceChip
            icon={<Sparkles className="h-3.5 w-3.5" />}
            label={t("sources.llmChip")}
          />
          <div className="mt-3">
            <MarkdownMessage content={report.initial_message} />
          </div>
        </div>
      )}
    </div>
  );
}
