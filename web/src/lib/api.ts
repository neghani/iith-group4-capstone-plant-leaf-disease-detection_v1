import type { ImageQualityResult, PredictResponse } from "./types";

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export async function fetchHealth(): Promise<{
  status: string;
  llm_configured: boolean;
  hf_model_configured: boolean;
  hf_model_id: string;
  hf_model_revision: string;
  hf_endpoint_configured: boolean;
  hf_reachable?: boolean | null;
  hf_served_model?: string | null;
} | null> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function errorDetail(res: Response, fallback: string): Promise<string> {
  const err = await res.json().catch(() => ({}));
  return typeof err.detail === "string" ? err.detail : fallback;
}

export async function checkQuality(file: File): Promise<ImageQualityResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/quality-check`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) throw new Error(await errorDetail(res, "Quality check failed"));
  return res.json();
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

  if (!res.ok) throw new Error(await errorDetail(res, "Upload failed"));

  return res.json();
}

export async function sendChat(
  message: string,
  sessionContext: Record<string, unknown>,
  history: { role: "user" | "assistant"; content: string }[] = []
): Promise<{ reply: string; declined: boolean }> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_context: sessionContext,
      history: history.slice(-6).map((turn) => ({
        role: turn.role,
        content: turn.content.slice(0, 400),
      })),
    }),
  });

  if (!res.ok) throw new Error("Chat request failed");
  return res.json();
}
