"use client";

import { useEffect, useRef, useState } from "react";
import { MessageCircle, Send, Sparkles, X } from "lucide-react";
import { MarkdownMessage } from "@/components/MarkdownMessage";
import { useLocale } from "@/i18n/LocaleProvider";
import { sendChat } from "@/lib/api";
import { predictionContextLabel } from "@/lib/chatContext";

type Message = { role: "user" | "assistant"; content: string };

const SUGGESTION_KEYS = [
  "chat.suggestion1",
  "chat.suggestion2",
  "chat.suggestion3",
] as const;

export function ChatPanel({
  sessionContext,
}: {
  sessionContext: Record<string, unknown>;
}) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const { locale, t } = useLocale();
  const contextLabel = predictionContextLabel(sessionContext);
  const inputRef = useRef<HTMLInputElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    const el = threadRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, loading, open]);

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const history = messages;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: trimmed }]);
    setLoading(true);
    try {
      const { reply } = await sendChat(trimmed, sessionContext, history, locale);
      setMessages((m) => [...m, { role: "assistant", content: reply }]);
    } catch {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: t("chat.unreachable"),
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    void ask(input);
  }

  return (
    <div className="pointer-events-none fixed bottom-5 right-5 z-50 flex flex-col items-end gap-3 sm:bottom-6 sm:right-6">
      {open && (
        <section
          role="dialog"
          aria-modal="false"
          aria-labelledby="chat-heading"
          className="pointer-events-auto flex h-[min(34rem,calc(100vh-6.5rem))] w-[min(24rem,calc(100vw-1.5rem))] flex-col overflow-hidden rounded-2xl border border-forest/15 bg-white shadow-[0_18px_50px_rgba(27,48,34,0.22)]"
        >
          <header className="flex shrink-0 items-start gap-3 border-b border-forest/10 bg-[linear-gradient(135deg,var(--color-sage)_0%,#f3f7f1_60%,var(--color-cream)_100%)] px-4 py-3.5">
            <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-forest text-white">
              <MessageCircle className="h-4 w-4" aria-hidden />
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-forest-muted)]">
                {t("chat.eyebrow")}
              </p>
              <h2
                id="chat-heading"
                className="truncate font-serif text-lg font-semibold text-forest"
              >
                {t("chat.titleShort")}
              </h2>
              <p className="mt-0.5 flex items-center gap-1 text-[11px] text-[var(--color-text-muted)]">
                <Sparkles className="h-3 w-3 shrink-0" aria-hidden />
                {t("chat.llmChip")}
              </p>
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="rounded-lg p-1.5 text-forest transition hover:bg-white/70"
              aria-label={t("chat.close")}
            >
              <X className="h-4 w-4" />
            </button>
          </header>

          {contextLabel && (
            <p className="shrink-0 border-b border-[var(--color-border)] bg-[var(--color-cream)] px-4 py-2 text-xs font-medium text-forest">
              {t("chat.thisCheck", { label: contextLabel })}
            </p>
          )}

          <div
            ref={threadRef}
            className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-4"
          >
            {messages.length === 0 && (
              <div>
                <p className="text-sm text-[var(--color-text-muted)]">
                  {t("chat.hintShort")}
                </p>
                <p className="mt-4 text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
                  {t("chat.tryAsking")}
                </p>
                <div className="mt-2 flex flex-col gap-2">
                  {SUGGESTION_KEYS.map((key) => (
                    <button
                      key={key}
                      type="button"
                      disabled={loading}
                      onClick={() => void ask(t(key))}
                      className="rounded-xl border border-forest/15 bg-sage/40 px-3 py-2.5 text-left text-sm text-forest transition hover:border-forest/35 hover:bg-sage disabled:opacity-50"
                    >
                      {t(key)}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div
                key={i}
                className={`rounded-2xl px-3 py-2 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "ml-8 bg-forest text-white"
                    : "mr-6 border border-[var(--color-border)] bg-[var(--color-cream)] text-forest"
                }`}
              >
                {m.role === "assistant" ? (
                  <MarkdownMessage content={m.content} />
                ) : (
                  m.content
                )}
              </div>
            ))}
            {loading && (
              <p className="mr-6 text-sm text-[var(--color-text-muted)]">
                {t("chat.thinking")}
              </p>
            )}
          </div>

          <form
            onSubmit={onSubmit}
            className="flex shrink-0 items-end gap-2 border-t border-[var(--color-border)] bg-white p-3"
          >
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={t("chat.placeholder")}
              className="min-h-11 flex-1 rounded-xl border border-[var(--color-border)] bg-[var(--color-cream)] px-3 py-2.5 text-sm text-forest outline-none placeholder:text-[var(--color-text-muted)] focus:border-forest/40"
              disabled={loading}
              aria-label={t("chat.placeholder")}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-forest text-white transition hover:bg-[var(--color-forest-muted)] disabled:opacity-40"
              aria-label={t("chat.send")}
            >
              <Send className="h-4 w-4" />
            </button>
          </form>
        </section>
      )}

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="pointer-events-auto flex items-center gap-2 rounded-full bg-forest px-4 py-3.5 text-sm font-semibold text-white shadow-[0_10px_28px_rgba(27,48,34,0.35)] transition hover:bg-[var(--color-forest-muted)]"
        aria-expanded={open}
        aria-controls={open ? "chat-heading" : undefined}
        aria-label={open ? t("chat.close") : t("chat.open")}
      >
        {open ? (
          <X className="h-5 w-5" aria-hidden />
        ) : (
          <>
            <MessageCircle className="h-5 w-5" aria-hidden />
            <span className="pr-0.5">{t("chat.fab")}</span>
            {messages.length > 0 && (
              <span className="ml-0.5 flex h-5 min-w-5 items-center justify-center rounded-full bg-[#f5c518] px-1.5 text-[11px] font-bold text-[#1a1a1a]">
                {messages.length}
              </span>
            )}
          </>
        )}
      </button>
    </div>
  );
}
