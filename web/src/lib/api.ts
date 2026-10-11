import type { PredictResponse } from "./types";

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export async function fetchHealth(): Promise<{
  status: string;
  llm_configured: boolean;
} | null> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function predictLeaf(
  file: File,
  crop: string,
  additionalSymptoms: string
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  if (crop) form.append("crop", crop);
  if (additionalSymptoms) form.append("additional_symptoms", additionalSymptoms);

  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(
      typeof err.detail === "string" ? err.detail : "Upload failed"
    );
  }

  return res.json();
}

export async function sendChat(
  message: string,
  sessionContext: Record<string, unknown>
): Promise<{ reply: string; declined: boolean }> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_context: sessionContext }),
  });

  if (!res.ok) throw new Error("Chat request failed");
  return res.json();
}
