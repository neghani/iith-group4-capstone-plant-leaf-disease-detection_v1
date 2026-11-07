"use client";

import { useEffect, useRef, useState } from "react";
import { FolderOpen, Timer, Cpu } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useT } from "@/i18n/LocaleProvider";
import {
  fetchLocalTestMeta,
  runLocalTest,
  type LocalTestMetaResponse,
  type SmokeVerdict,
  type WebTestEvent,
} from "@/lib/api";

type Row = Extract<WebTestEvent, { status: "row" }>;

function modelBadge(modelId?: string | null, revision?: string | null) {
  if (!modelId) return "—";
  const short = modelId.includes("/") ? modelId.split("/").pop() : modelId;
  return revision ? `${short}@${revision}` : short;
}

function verdictStyle(verdict: SmokeVerdict | undefined, match: boolean) {
  switch (verdict) {
    case "yes":
      return "bg-emerald-100 text-emerald-900";
    case "abstain":
      return "bg-slate-100 text-slate-800";
    case "crop_mismatch":
      return "bg-orange-100 text-orange-950";
    case "no":
      return "bg-amber-100 text-amber-950";
    default:
      return match
        ? "bg-emerald-100 text-emerald-900"
        : "bg-amber-100 text-amber-950";
  }
}

export default function LocalTestPage() {
  const t = useT();
  const [meta, setMeta] = useState<LocalTestMetaResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [n, setN] = useState(10);
  const [useGtCropHint, setUseGtCropHint] = useState(true);
  const [split, setSplit] = useState<"" | "tune" | "final">("");
  const [running, setRunning] = useState(false);
  const [statusLine, setStatusLine] = useState("");
  const [scheduleHint, setScheduleHint] = useState<string[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [summary, setSummary] = useState<{
    correct: number;
    abstain: number;
    wrong: number;
    n: number;
    accuracy: number;
    answeredAccuracy: number | null;
    threshold: number;
  } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchLocalTestMeta()
      .then((data) => {
        if (!cancelled) setMeta(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setLoadError(err.message || t("localTest.loadError"));
      });
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, [t]);

  function verdictLabel(row: Row) {
    switch (row.verdict) {
      case "yes":
        return t("webTest.matchYes");
      case "abstain":
        return t("webTest.matchAbstain");
      case "crop_mismatch":
        return t("webTest.matchCropMismatch");
      case "no":
        return t("webTest.matchNo");
      default:
        return row.match ? t("webTest.matchYes") : t("webTest.matchNo");
    }
  }

  async function onRun() {
    if (running) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setRunning(true);
    setRows([]);
    setSummary(null);
    setScheduleHint([]);
    setStatusLine(t("localTest.searching"));
    try {
      await runLocalTest(
        n,
        (event) => {
          if (event.status === "searching") {
            setScheduleHint(event.schedule ?? []);
            setStatusLine(
              t("localTest.searchingFor", {
                model: `${event.model_id}@${event.model_revision}`,
                n: String(event.n),
              })
            );
          } else if (event.status === "found") {
            setStatusLine(
              t("localTest.found", { count: String(event.candidates) })
            );
          } else if (event.status === "row") {
            setRows((prev) => [...prev, event]);
            setStatusLine(
              t("localTest.rowProgress", {
                i: String(event.index + 1),
                n: String(n),
              })
            );
          } else if (event.status === "done") {
            setSummary({
              correct: event.correct,
              abstain: event.abstain ?? 0,
              wrong:
                event.wrong ??
                Math.max(0, event.n - event.correct - (event.abstain ?? 0)),
              n: event.n,
              accuracy: event.accuracy,
              answeredAccuracy: event.answered_accuracy ?? null,
              threshold:
                event.confidence_threshold ?? meta?.confidence_threshold ?? 0.5,
            });
            setStatusLine(
              t("webTest.doneBand", {
                correct: String(event.correct),
                abstain: String(event.abstain ?? 0),
                wrong: String(
                  event.wrong ??
                    Math.max(0, event.n - event.correct - (event.abstain ?? 0))
                ),
                n: String(event.n),
                pct: String(Math.round(event.accuracy * 100)),
              })
            );
          } else if (event.status === "error") {
            setStatusLine(event.detail);
          }
        },
        ac.signal,
        useGtCropHint,
        split || null
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setStatusLine((err as Error).message || t("localTest.runError"));
      }
    } finally {
      setRunning(false);
    }
  }

  const pinLabel = meta
    ? `${meta.model_id}@${meta.model_revision}`
    : t("webTest.pinLoading");
  const thrPct = Math.round((meta?.confidence_threshold ?? 0.5) * 100);

  return (
    <AppShell breadcrumb={t("localTest.breadcrumb")}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-forest-muted)]">
            {t("localTest.eyebrow")}
          </p>
          <h1 className="mt-1 font-serif text-3xl font-semibold text-forest">
            {t("localTest.title")}
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-[var(--color-text-muted)]">
            {t("localTest.subtitle")}
          </p>
        </div>
        {meta && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-forest/20 bg-sage/50 px-3 py-1.5 text-xs font-semibold text-forest">
              <Cpu className="h-3.5 w-3.5" aria-hidden />
              {pinLabel}
            </span>
            <span className="rounded-full border border-[var(--color-border)] bg-white px-3 py-1.5 text-xs font-semibold text-forest">
              {t("webTest.thresholdBadge", { pct: String(thrPct) })}
            </span>
            <span className="rounded-full border border-[var(--color-border)] bg-white px-3 py-1.5 text-xs font-semibold text-forest">
              {t("localTest.setBadge", {
                n: String(meta.n_images),
                csv: String(meta.n_labels_csv),
              })}
            </span>
            {!meta.hf_configured && (
              <span className="text-xs font-medium text-red-700">
                {t("webTest.noHf")}
              </span>
            )}
          </div>
        )}
      </div>

      {loadError && (
        <p className="mt-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {loadError}
        </p>
      )}

      {meta && !meta.available && (
        <p className="mt-6 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
          {t("localTest.missingImages", {
            csv: String(meta.n_labels_csv),
            dir: meta.set_dir || "model/data/real_world_set",
          })}
        </p>
      )}

      <div className="mt-8 flex flex-col gap-3 rounded-2xl border border-[var(--color-border)] bg-white p-4 sm:flex-row sm:items-end">
        <div className="min-w-0 flex-1 text-sm text-[var(--color-text-muted)]">
          <p className="font-medium text-forest">{t("localTest.poolLabel")}</p>
          <p className="mt-1">{t("localTest.poolHint")}</p>
          <label className="mt-3 flex cursor-pointer items-start gap-2 text-sm text-forest">
            <input
              type="checkbox"
              className="mt-1"
              checked={useGtCropHint}
              disabled={running}
              onChange={(e) => setUseGtCropHint(e.target.checked)}
            />
            <span>
              <span className="font-medium">{t("webTest.cropHintLabel")}</span>
              <span className="mt-0.5 block text-xs text-[var(--color-text-muted)]">
                {t("webTest.cropHintHint")}
              </span>
            </span>
          </label>
        </div>
        <label className="w-full text-sm sm:w-32">
          <span className="mb-1 block font-medium text-forest">
            {t("localTest.splitLabel")}
          </span>
          <select
            value={split}
            disabled={running}
            onChange={(e) =>
              setSplit(e.target.value as "" | "tune" | "final")
            }
            className="w-full rounded-xl border border-[var(--color-border)] bg-[var(--color-cream)] px-3 py-2.5 text-sm text-forest outline-none focus:border-forest/40"
          >
            <option value="">{t("localTest.splitAny")}</option>
            <option value="tune">tune</option>
            <option value="final">final</option>
          </select>
        </label>
        <label className="w-full text-sm sm:w-28">
          <span className="mb-1 block font-medium text-forest">
            {t("webTest.nLabel")}
          </span>
          <input
            type="number"
            min={1}
            max={15}
            value={n}
            disabled={running}
            onChange={(e) => setN(Number(e.target.value) || 10)}
            className="w-full rounded-xl border border-[var(--color-border)] bg-[var(--color-cream)] px-3 py-2.5 text-sm text-forest outline-none focus:border-forest/40"
          />
        </label>
        <button
          type="button"
          disabled={
            running || meta?.hf_configured === false || meta?.available === false
          }
          onClick={() => void onRun()}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-forest px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--color-forest-muted)] disabled:opacity-40"
        >
          <FolderOpen className="h-4 w-4" aria-hidden />
          {running ? t("localTest.running") : t("localTest.run")}
        </button>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-muted)]">
        {t("localTest.gtNote")}
      </p>
      {scheduleHint.length > 0 && (
        <p className="mt-2 text-xs text-[var(--color-forest-muted)]">
          {t("webTest.schedule")}: {scheduleHint.join(" · ")}
        </p>
      )}
      {statusLine && (
        <p className="mt-4 text-sm font-medium text-forest" aria-live="polite">
          {statusLine}
        </p>
      )}
      {summary && (
        <p className="mt-2 text-sm text-[var(--color-forest-muted)]">
          {t("webTest.scorelineBand", {
            correct: String(summary.correct),
            abstain: String(summary.abstain),
            wrong: String(summary.wrong),
            n: String(summary.n),
            pct: String(Math.round(summary.accuracy * 100)),
            answered:
              summary.answeredAccuracy != null
                ? String(Math.round(summary.answeredAccuracy * 100))
                : "—",
            thr: String(Math.round(summary.threshold * 100)),
          })}
        </p>
      )}

      <div className="mt-6 overflow-x-auto rounded-2xl border border-[var(--color-border)] bg-white">
        <table className="min-w-full text-left text-sm">
          <thead className="border-b border-[var(--color-border)] bg-[var(--color-cream)] text-xs uppercase tracking-wide text-[var(--color-text-muted)]">
            <tr>
              <th className="px-3 py-3 font-semibold">#</th>
              <th className="px-3 py-3 font-semibold">{t("webTest.colImage")}</th>
              <th className="px-3 py-3 font-semibold">{t("webTest.colGt")}</th>
              <th className="px-3 py-3 font-semibold">{t("webTest.colPred")}</th>
              <th className="px-3 py-3 font-semibold">{t("webTest.colScore")}</th>
              <th className="px-3 py-3 font-semibold">{t("webTest.colModel")}</th>
              <th className="px-3 py-3 font-semibold">{t("webTest.colLatency")}</th>
              <th className="px-3 py-3 font-semibold">{t("webTest.colMatch")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td
                  colSpan={8}
                  className="px-3 py-10 text-center text-[var(--color-text-muted)]"
                >
                  {running ? t("webTest.waitingRows") : t("localTest.empty")}
                </td>
              </tr>
            )}
            {rows.map((row) => (
              <tr
                key={`${row.index}-${row.source_url}`}
                className="border-t border-[var(--color-border)] align-middle"
              >
                <td className="px-3 py-3 text-[var(--color-text-muted)]">
                  {row.index + 1}
                </td>
                <td className="px-3 py-3">
                  <div className="flex flex-col items-start gap-1">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={row.image}
                      alt=""
                      className="h-28 w-28 rounded-xl object-cover ring-1 ring-[var(--color-border)]"
                    />
                    <span className="text-[10px] font-medium uppercase tracking-wide text-[var(--color-text-muted)]">
                      {row.leaf_found
                        ? t("webTest.leafCrop")
                        : t("webTest.wholeFrame")}
                      {row.crop_strategy ? ` · ${row.crop_strategy}` : ""}
                      {row.split ? ` · ${row.split}` : ""}
                    </span>
                  </div>
                </td>
                <td className="px-3 py-3 text-forest">
                  {row.gt_display}
                  {row.use_gt_crop_hint && row.gt_crop && (
                    <span className="mt-1 block text-[11px] text-[var(--color-text-muted)]">
                      {t("webTest.hintCrop", { crop: row.gt_crop })}
                    </span>
                  )}
                </td>
                <td className="px-3 py-3 text-forest">
                  {row.pred_display ?? "—"}
                  {row.crop_mismatch && row.pred_crop && (
                    <span className="mt-1 block text-[11px] text-orange-800">
                      {t("webTest.predCrop", { crop: row.pred_crop })}
                    </span>
                  )}
                  {row.error && (
                    <span className="mt-1 block text-xs text-red-700">
                      {row.error}
                    </span>
                  )}
                </td>
                <td className="px-3 py-3 tabular-nums text-forest">
                  {row.score != null ? `${Math.round(row.score * 100)}%` : "—"}
                </td>
                <td className="px-3 py-3">
                  <span className="inline-flex items-center gap-1 rounded-full bg-forest/10 px-2.5 py-0.5 text-[11px] font-semibold text-forest">
                    <Cpu className="h-3 w-3 shrink-0" aria-hidden />
                    {modelBadge(row.model_id, row.model_revision)}
                  </span>
                </td>
                <td className="px-3 py-3">
                  <span className="inline-flex items-center gap-1 rounded-full bg-[var(--color-cream)] px-2.5 py-0.5 text-[11px] font-semibold tabular-nums text-forest">
                    <Timer className="h-3 w-3 shrink-0" aria-hidden />
                    {row.latency_ms != null ? `${row.latency_ms} ms` : "—"}
                  </span>
                </td>
                <td className="px-3 py-3">
                  <span
                    className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold ${verdictStyle(
                      row.verdict,
                      row.match
                    )}`}
                  >
                    {verdictLabel(row)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </AppShell>
  );
}
