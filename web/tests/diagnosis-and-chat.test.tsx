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

describe("DiagnosisCard top-1 display", () => {
  it("shows only the best match with confidence, not the full top-3 list", () => {
    render(
      <LocaleProvider>
        <DiagnosisCard report={baseReport} topK={topK} />
      </LocaleProvider>
    );

    expect(
      screen.getByText("Tomato - Early blight, 91% confidence")
    ).toBeTruthy();
    expect(screen.queryByText("Tomato - Target Spot")).toBeNull();
    expect(screen.queryByText("6%")).toBeNull();
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
  it("renders assistant replies with markdown formatting", async () => {
    const user = userEvent.setup();
    render(
      <LocaleProvider>
        <ChatPanel sessionContext={{ diagnosis_name: "Tomato - Early blight" }} />
      </LocaleProvider>
    );

    await user.type(
      screen.getByRole("textbox"),
      "What should I spray?"
    );
    await user.click(screen.getByRole("button"));

    await waitFor(() => {
      const strong = screen.getByText("neem oil");
      expect(strong.tagName).toBe("STRONG");
    });
  });
});
