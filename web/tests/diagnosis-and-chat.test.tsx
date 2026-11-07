import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatPanel } from "../src/components/ChatPanel";
import { DiagnosisCard } from "../src/components/DiagnosisCard";
import { MarkdownMessage } from "../src/components/MarkdownMessage";
import { LocaleProvider } from "../src/i18n/LocaleProvider";
import type { DiagnosisReport, TopKPrediction } from "../src/lib/types";

vi.mock("next/link", () => ({
  default: ({
    href,
    children,
    className,
  }: {
    href: string;
    children: React.ReactNode;
    className?: string;
  }) => (
    <a href={href} className={className}>
      {children}
    </a>
  ),
}));

vi.mock("../src/lib/api", () => ({
  sendChat: vi.fn(async () => ({ reply: "Use **neem oil** sparingly." })),
}));

afterEach(() => {
  cleanup();
});

const baseReport: DiagnosisReport = {
  status: "confirmed_likely",
  status_label: "Likely disease",
  disease_id: "Tomato___Early_blight",
  diagnosis_name: "Tomato - Early blight",
  crop: "Tomato",
  confidence: 0.91,
  confidence_category: "high",
  confidence_explanation: "Looks good.",
  description: "A common fungal disease.",
  symptoms: [],
  alternatives: ["Tomato - Target Spot (6%)"],
  expert_confirmation_required: false,
  actions: [],
  monitoring_guidance: "",
  diagnostic_limitations: "",
  initial_message: "",
};

const topK: TopKPrediction[] = [
  {
    label: "Tomato___Early_blight",
    display_name: "Tomato - Early blight",
    score: 0.91,
  },
  {
    label: "Tomato___Target_Spot",
    display_name: "Tomato - Target Spot",
    score: 0.06,
  },
  {
    label: "Tomato___healthy",
    display_name: "Tomato - healthy",
    score: 0.02,
  },
];

describe("DiagnosisCard provenance", () => {
  it("shows only the top-1 classifier match on a firm diagnosis", () => {
    render(
      <LocaleProvider>
        <DiagnosisCard report={baseReport} topK={topK} />
      </LocaleProvider>
    );

    expect(
      screen.getByText("Tomato - Early blight, 91% confidence")
    ).toBeTruthy();
    expect(screen.getByText("What the model detected")).toBeTruthy();
    expect(screen.queryByText("Tomato - Target Spot")).toBeNull();
    expect(screen.queryByText("Tomato - healthy")).toBeNull();
  });

  it("shows all classifier guesses when the model is not sure", () => {
    render(
      <LocaleProvider>
        <DiagnosisCard
          report={{
            ...baseReport,
            status: "not_sure",
            status_label: "Not sure",
            diagnosis_name: "I'm not confident — please retake the photo.",
            disease_id: null,
            initial_message: "",
          }}
          topK={topK}
        />
      </LocaleProvider>
    );

    expect(screen.getByText("Tomato - Early blight, 91% confidence")).toBeTruthy();
    expect(screen.getByText("Tomato - Target Spot")).toBeTruthy();
    expect(screen.getByText("Tomato - healthy")).toBeTruthy();
  });

  it("labels the curated reference and the LLM summary separately", () => {
    render(
      <LocaleProvider>
        <DiagnosisCard
          report={{ ...baseReport, initial_message: "Here is what to do." }}
          topK={topK}
        />
      </LocaleProvider>
    );

    expect(screen.getByText("From knowledge base")).toBeTruthy();
    expect(
      screen.getByText("AI-written summary (text only — your photo is never sent)")
    ).toBeTruthy();
    expect(screen.getByText("Here is what to do.")).toBeTruthy();
  });

  it("shows the model chip when inference info is present", () => {
    render(
      <LocaleProvider>
        <DiagnosisCard
          report={baseReport}
          topK={topK}
          inference={{
            model_id: "neghani/plantvillage-resnet18",
            model_revision: "v1.1",
            model_verified: true,
            source: "endpoint",
            endpoint_host: "inference-service:7860",
            latency_ms: 160,
            input_bytes: 33853,
            input_size: [256, 256],
            preprocessing: {},
          }}
        />
      </LocaleProvider>
    );

    expect(
      screen.getByText(
        "Model: neghani/plantvillage-resnet18 (v1.1) · verified by server · 160 ms"
      )
    ).toBeTruthy();
  });
});

describe("MarkdownMessage", () => {
  it("renders **bold** as strong text", () => {
    render(<MarkdownMessage content="Try **neem oil** today." />);
    const strong = screen.getByText("neem oil");
    expect(strong.tagName).toBe("STRONG");
  });
});

describe("ChatPanel markdown", () => {
  it("opens as a popup and renders assistant replies with markdown", async () => {
    const user = userEvent.setup();
    render(
      <LocaleProvider>
        <ChatPanel sessionContext={{ diagnosis_name: "Tomato - Early blight" }} />
      </LocaleProvider>
    );

    expect(screen.getByRole("button", { name: "Open CropCare assistant" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Open CropCare assistant" }));

    expect(screen.getByText("Ask CropCare")).toBeTruthy();
    expect(screen.getByText("Should I remove the lower leaves?")).toBeTruthy();

    await user.type(
      screen.getByRole("textbox"),
      "What should I spray?"
    );
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => {
      const strong = screen.getByText("neem oil");
      expect(strong.tagName).toBe("STRONG");
    });
  });
});
