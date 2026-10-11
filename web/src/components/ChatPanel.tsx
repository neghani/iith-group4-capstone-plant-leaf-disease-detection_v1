"use client";

import { useState } from "react";
import { Send } from "lucide-react";
import { sendChat } from "@/lib/api";

type Message = { role: "user" | "assistant"; content: string };

export function ChatPanel({
  sessionContext,
}: {
  sessionContext: Record<string, unknown>;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    try {
      const { reply } = await sendChat(text, sessionContext);
      setMessages((m) => [...m, { role: "assistant", content: reply }]);
    } catch {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: "Could not reach the assistant. Is the API running?",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-card bg-white p-6 shadow-card">
      <h4 className="font-semibold text-forest">Ask a follow-up</h4>
      <p className="mt-1 text-xs text-[var(--color-text-muted)]">
        Agriculture questions only · tied to this check in your browser session
      </p>

      <div className="mt-4 max-h-64 space-y-3 overflow-y-auto">
        {messages.length === 0 && (
          <p className="text-sm text-[var(--color-text-muted)]">
            e.g. Should I remove the lower leaves?
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
          placeholder="Ask about symptoms, care, or monitoring…"
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
