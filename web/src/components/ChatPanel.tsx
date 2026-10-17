"use client";

import { useState } from "react";
import { Send } from "lucide-react";
import { useLocale } from "@/i18n/LocaleProvider";
import { sendChat } from "@/lib/api";
import { predictionContextLabel } from "@/lib/chatContext";

type Message = { role: "user" | "assistant"; content: string };

export function ChatPanel({
  sessionContext,
}: {
  sessionContext: Record<string, unknown>;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const { locale, t } = useLocale();
  const contextLabel = predictionContextLabel(sessionContext);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const history = messages;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    try {
      const { reply } = await sendChat(text, sessionContext, history, locale);
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

  return (
    <div className="rounded-card bg-white p-6 shadow-card">
      <h4 className="font-semibold text-forest">{t("chat.title")}</h4>
      <p className="mt-1 text-xs text-[var(--color-text-muted)]">
        {t("chat.hint")}
      </p>
      {contextLabel && (
        <p className="mt-3 rounded-lg bg-sage/50 px-3 py-2 text-sm text-forest">
          {t("chat.thisCheck", { label: contextLabel })}
        </p>
      )}

      <div className="mt-4 max-h-64 space-y-3 overflow-y-auto">
        {messages.length === 0 && (
          <p className="text-sm text-[var(--color-text-muted)]">
            {t("chat.example")}
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`rounded-xl px-3 py-2 text-sm ${
              m.role === "user"
                ? "ml-8 bg-sage text-forest"
                : "mr-8 bg-[var(--color-cream)] text-forest"
            }`}
          >
            {m.content}
          </div>
        ))}
      </div>

      <form onSubmit={onSubmit} className="mt-4 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={t("chat.placeholder")}
          className="flex-1 rounded-xl border border-[var(--color-border)] px-4 py-2.5 text-sm outline-none focus:border-forest/40"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="flex items-center justify-center rounded-xl bg-forest px-4 py-2.5 text-white disabled:opacity-50"
        >
          <Send className="h-4 w-4" />
        </button>
      </form>
    </div>
  );
}
