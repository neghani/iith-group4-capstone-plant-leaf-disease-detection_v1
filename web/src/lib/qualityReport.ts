import type { DiagnosisReport, ImageQualityResult } from "./types";

/** Matches api/app/services/report.py build_quality_report so a failed gate can open results without /predict. */
export function qualityReportFromCheck(quality: ImageQualityResult): DiagnosisReport {
  const message =
    quality.message || "Please retake the photo with the leaf in focus.";
  return {
    status: "quality_issue",
    status_label: "Photo needs improvement",
    diagnosis_name: "Unable to analyze",
    confidence: 0,
    confidence_category: "inconclusive",
    confidence_explanation: "Image quality checks failed before prediction.",
    description: message,
    symptoms: [],
    alternatives: [],
    expert_confirmation_required: true,
    actions: [
      {
        step: "Retake the photo with the leaf centered, in daylight, and in focus.",
      },
    ],
    monitoring_guidance: "After retaking, upload again for a new assessment.",
    diagnostic_limitations: "No disease prediction was run on this image.",
    initial_message: message,
  };
}
