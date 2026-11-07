"use client";

import { useEffect, useRef, useState } from "react";
import { FlaskConical, Timer, Cpu } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { useT } from "@/i18n/LocaleProvider";
import {
  fetchWebTestClasses,
  runWebTest,
  type WebTestClassesResponse,
  type WebTestEvent,
} from "@/lib/api";

type Row = Extract<WebTestEvent, { status: "row" }>;

function modelBadge(modelId?: string | null, revision?: string | null) {
  if (!modelId) return "—";
  const short = modelId.includes("/") ? modelId.split("/").pop() : modelId;
  return revision ? `${short}@${revision}` : short;
}

export default function WebTestPage() {
  const t = useT();
  const [meta, setMeta] = useState<WebTestClassesResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [n, setN] = useState(10);
  const [running, setRunning] = useState(false);
  const [statusLine, setStatusLine] = useState("");
  const [scheduleHint, setScheduleHint] = useState<string[]>([]);
  const [rows, setRows] = useState<Row[]>([]);
  const [summary, setSummary] = useState<{
    correct: number;
    n: number;
    accuracy: number;
  } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchWebTestClasses()
      .then((data) => {
        if (!cancelled) setMeta(data);
      })
      .catch((err: Error) => {
        if (!cancelled) setLoadError(err.message || t("webTest.loadError"));
      });
    return () => {
      cancelled = true;
      abortRef.current?.abort();
    };
  }, [t]);

  async function onRun() {
    if (running) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setRunning(true);
    setRows([]);
    setSummary(null);
    setScheduleHint([]);
    setStatusLine(t("webTest.searching"));
    try {
      await runWebTest(
        n,
        (event) => {
          if (event.status === "searching") {
            setScheduleHint(event.schedule ?? []);
            setStatusLine(
              t("webTest.searchingFor", {
                name: event.gt_display,
                model: `${event.model_id}@${event.model_revision}`,
              })
            );
          } else if (event.status === "found") {
            setStatusLine(t("webTest.found", { count: String(event.candidates) }));
          } else if (event.status === "row") {
            setRows((prev) => [...prev, event]);
            setStatusLine(
              t("webTest.rowProgress", {
                i: String(event.index + 1),
                n: String(n),
              })
            );
          } else if (event.status === "done") {
            setSummary({
              correct: event.correct,
              n: event.n,
              accuracy: event.accuracy,
            });
            setStatusLine(
              t("webTest.done", {
                correct: String(event.correct),
                n: String(event.n),
                pct: String(Math.round(event.accuracy * 100)),
              })
            );
          } else if (event.status === "error") {
            setStatusLine(event.detail);
          }
        },
        ac.signal
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setStatusLine((err as Error).message || t("webTest.runError"));
      }
    } finally {
      setRunning(false);
    }
  }

  const pinLabel = meta
    ? `${meta.model_id}@${meta.model_revision}`
    : t("webTest.pinLoading");

  return (
    <AppShell breadcrumb={t("webTest.breadcrumb")}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-forest-muted)]">
            {t("webTest.eyebrow")}
          </p>
          <h1 className="mt-1 font-serif text-3xl font-semibold text-forest">
            {t("webTest.title")}
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-[var(--color-text-muted)]">
            {t("webTest.subtitle")}
          </p>
        </div>
        {meta && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-forest/20 bg-sage/50 px-3 py-1.5 text-xs font-semibold text-forest">
              <Cpu className="h-3.5 w-3.5" aria-hidden />
              {pinLabel}
            </span>
            {!meta.hf_configured && (
              <span className="text-xs font-medium text-red-700">{t("webTest.noHf")}</span>
            )}
          </div>
        )}
      </div>

      {loadError && (
        <p className="mt-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {loadError}
        </p>
      )}

      <div className="mt-8 flex flex-col gap-3 rounded-2xl border border-[var(--color-border)] bg-white p-4 sm:flex-row sm:items-end">
        <div className="min-w-0 flex-1 text-sm text-[var(--color-text-muted)]">
          <p className="font-medium text-forest">{t("webTest.poolLabel")}</p>
          <p className="mt-1">
            {t("webTest.poolHint", {
              count: String(meta?.demo_pool?.length ?? 0),
            })}
          </p>
        </div>
        <label className="w-full text-sm sm:w-28">
          <span className="mb-1 block font-medium text-forest">{t("webTest.nLabel")}</span>
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
          disabled={running || meta?.hf_configured === false}
          onClick={() => void onRun()}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-forest px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--color-forest-muted)] disabled:opacity-40"
        >
          <FlaskConical className="h-4 w-4" aria-hidden />
          {running ? t("webTest.running") : t("webTest.run")}
        </button>
      </div>

      <p className="mt-3 text-xs text-[var(--color-text-muted)]">{t("webTest.gtNote")}</p>
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
          {t("webTest.scoreline", {
            correct: String(summary.correct),
            n: String(summary.n),
            pct: String(Math.round(summary.accuracy * 100)),
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
                  {running ? t("webTest.waitingRows") : t("webTest.empty")}
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
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={row.image}
                    alt=""
                    className="h-16 w-16 rounded-lg object-cover"
                  />
                </td>
                <td className="px-3 py-3 text-forest">{row.gt_display}</td>
                <td className="px-3 py-3 text-forest">
                  {row.pred_display ?? "—"}
                  {row.error && (
                    <span className="mt-1 block text-xs text-red-700">{row.error}</span>
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
                    className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                      row.match
                        ? "bg-emerald-100 text-emerald-900"
                        : "bg-amber-100 text-amber-950"
                    }`}
                  >
                    {row.match ? t("webTest.matchYes") : t("webTest.matchNo")}
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
