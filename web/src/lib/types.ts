export type DiagnosisReport = {
  status: string;
  status_label: string;
  disease_id?: string | null;
  diagnosis_name: string;
  crop?: string | null;
  confidence: number;
  confidence_category: string;
  confidence_explanation: string;
  description: string;
  symptoms: { name: string; explanation: string; evidence_source?: string }[];
  alternatives: string[];
  expert_confirmation_required: boolean;
  actions: { step: string; safety_note?: string | null }[];
  monitoring_guidance: string;
  diagnostic_limitations: string;
  initial_message: string;
};

export type ImageQualityResult = {
  ok: boolean;
  reason?: string | null;
  message?: string | null;
  brightness?: number | null;
  blur_score?: number | null;
  leaf_found?: boolean | null;
  leaf_coverage?: number | null;
  original_size?: number[] | null;
};

export type InferenceInfo = {
  model_id: string;
  model_revision: string;
  model_verified: boolean;
  source: string;
  endpoint_host: string;
  latency_ms: number;
  input_bytes: number;
  input_size?: number[] | null;
  preprocessing: Record<string, unknown>;
};

export type TopKPrediction = {
  label: string;
  display_name: string;
  score: number;
};

export type ProvenanceEntry = {
  /** Dotted path into the response, e.g. "report.description". */
  field: string;
  source:
    | "image_quality_checks"
    | "image_classifier"
    | "decision_rules"
    | "curated_reference"
    | "farmer_input"
    | "llm_summary"
    | "template_text";
  /** Human-readable component name, e.g. "Image classifier (org/model v1.1)". */
  label: string;
  /** What exactly that component contributed to the field. */
  detail: string;
};

export type PredictResponse = {
  quality: ImageQualityResult;
  low_confidence: boolean;
  report: DiagnosisReport;
  top_k: TopKPrediction[];
  session_context: Record<string, unknown>;
  inference?: InferenceInfo | null;
  provenance?: ProvenanceEntry[];
  language?: "en" | "te" | string;
  translation_applied?: boolean;
};

export type CheckSession = {
  id: string;
  timestamp: number;
  cropLabel: string;
  previewDataUrl: string;
  report: DiagnosisReport;
  sessionContext: Record<string, unknown>;
  topK?: TopKPrediction[];
  lowConfidence: boolean;
  qualityOk: boolean;
  inference?: InferenceInfo;
  qualityReason?: string | null;
  language?: "en" | "te";
  translationApplied?: boolean;
};
