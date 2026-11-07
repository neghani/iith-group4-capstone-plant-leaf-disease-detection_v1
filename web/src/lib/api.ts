import type { ImageQualityResult, PredictResponse } from "./types";

const rawApi = process.env.NEXT_PUBLIC_API_URL;
const API_URL = (
  rawApi === "" || rawApi === "same-origin"
    ? ""
    : (rawApi ?? "http://localhost:8000")
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
  additionalSymptoms: string,
  language: "en" | "te" = "en"
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("language", language);
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
  history: { role: "user" | "assistant"; content: string }[] = [],
  language: "en" | "te" = "en"
): Promise<{ reply: string; declined: boolean }> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      language,
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

export type WebTestClass = {
  class_id: string;
  display_name: string;
  query: string;
  curated: boolean;
};

export type WebTestClassesResponse = {
  model_id: string;
  model_revision: string;
  hf_configured: boolean;
  confidence_threshold?: number;
  demo_pool: WebTestClass[];
  curated: WebTestClass[];
  all: WebTestClass[];
  note: string;
};

export type LocalTestMetaResponse = {
  model_id: string;
  model_revision: string;
  hf_configured: boolean;
  confidence_threshold?: number;
  set_dir: string | null;
  available: boolean;
  n_images: number;
  n_labels_csv: number;
  splits: Record<string, number>;
  note: string;
};

export type SmokeVerdict = "yes" | "no" | "abstain" | "crop_mismatch";

export type WebTestEvent =
  | {
      status: "searching";
      mode?: "mixed" | "single" | "local";
      source?: string;
      class_id?: string | null;
      gt_display: string;
      query?: string | null;
      schedule?: string[];
      n: number;
      model_id: string;
      model_revision: string;
      confidence_threshold?: number;
      use_gt_crop_hint?: boolean;
      split?: string | null;
      set_dir?: string;
    }
  | { status: "found"; candidates: number }
  | {
      status: "row";
      index: number;
      image: string;
      source_url: string;
      gt_id: string;
      gt_display: string;
      pred_id: string | null;
      pred_display: string | null;
      score: number | null;
      match: boolean;
      verdict?: SmokeVerdict;
      crop_mismatch?: boolean;
      gt_crop?: string;
      pred_crop?: string | null;
      confidence_threshold?: number;
      use_gt_crop_hint?: boolean;
      error?: string;
      top_k?: { label: string; display_name: string; score: number }[];
      model_id?: string;
      model_revision?: string;
      latency_ms?: number | null;
      gate_ok?: boolean;
      leaf_found?: boolean | null;
      crop_strategy?: string;
      model_status?: string;
      split?: string;
      plantdoc_class?: string;
    }
  | {
      status: "done";
      mode?: "mixed" | "single" | "local";
      source?: string;
      n: number;
      correct: number;
      abstain?: number;
      wrong?: number;
      accuracy: number;
      answered_accuracy?: number | null;
      confidence_threshold?: number;
      use_gt_crop_hint?: boolean;
      class_id?: string | null;
      gt_display: string;
      model_id?: string;
      model_revision?: string;
    }
  | { status: "error"; detail: string };

async function readNdjsonStream(
  res: Response,
  onEvent: (event: WebTestEvent) => void
): Promise<void> {
  if (!res.body) throw new Error("No response body");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      onEvent(JSON.parse(trimmed) as WebTestEvent);
    }
  }
  const tail = buffer.trim();
  if (tail) onEvent(JSON.parse(tail) as WebTestEvent);
}

export async function fetchWebTestClasses(): Promise<WebTestClassesResponse> {
  const res = await fetch(`${API_URL}/web-test/classes`, { cache: "no-store" });
  if (!res.ok) throw new Error(await errorDetail(res, "Could not load web-test classes"));
  return res.json();
}

export async function fetchLocalTestMeta(): Promise<LocalTestMetaResponse> {
  const res = await fetch(`${API_URL}/local-test/meta`, { cache: "no-store" });
  if (!res.ok) throw new Error(await errorDetail(res, "Could not load local-test meta"));
  return res.json();
}

/** Stream NDJSON events from search → classify (rows appear one by one). */
export async function runWebTest(
  n: number,
  onEvent: (event: WebTestEvent) => void,
  signal?: AbortSignal,
  classId?: string | null,
  useGtCropHint: boolean = true
): Promise<void> {
  const body: { n: number; class_id?: string; use_gt_crop_hint: boolean } = {
    n,
    use_gt_crop_hint: useGtCropHint,
  };
  if (classId) body.class_id = classId;
  const res = await fetch(`${API_URL}/web-test/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(await errorDetail(res, "Web test failed"));
  await readNdjsonStream(res, onEvent);
}

/** Stream NDJSON from labelled real_world_set sample → classify. */
export async function runLocalTest(
  n: number,
  onEvent: (event: WebTestEvent) => void,
  signal?: AbortSignal,
  useGtCropHint: boolean = true,
  split?: string | null
): Promise<void> {
  const body: {
    n: number;
    use_gt_crop_hint: boolean;
    split?: string;
  } = { n, use_gt_crop_hint: useGtCropHint };
  if (split) body.split = split;
  const res = await fetch(`${API_URL}/local-test/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/x-ndjson" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(await errorDetail(res, "Local test failed"));
  await readNdjsonStream(res, onEvent);
}
