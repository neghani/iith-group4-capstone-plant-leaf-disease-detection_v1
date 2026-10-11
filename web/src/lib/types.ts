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

export type PredictResponse = {
  quality: { ok: boolean; message?: string | null };
  low_confidence: boolean;
  report: DiagnosisReport;
  session_context: Record<string, unknown>;
};

export type CheckSession = {
  id: string;
  timestamp: number;
  cropLabel: string;
  previewDataUrl: string;
  report: DiagnosisReport;
  sessionContext: Record<string, unknown>;
  lowConfidence: boolean;
  qualityOk: boolean;
};
