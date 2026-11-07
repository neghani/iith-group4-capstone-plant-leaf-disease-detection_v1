#!/usr/bin/env python3
"""Generate the CropCare AI demo-day deck (PPTX).

Slide order and content follow docs/10-slide-visuals.md and
docs/demo-day-prep-guide.md (sections 0, 2, 5, 9.1, 11.4).

Usage: python3 make_deck.py   ->  writes CropCare-AI-Demo-Day.pptx at repo root
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "docs", "assets")

# ---------- theme ----------
GREEN_DARK = RGBColor(0x1B, 0x5E, 0x20)   # deep green
GREEN_MID = RGBColor(0x2E, 0x7D, 0x32)
GREEN_ACCENT = RGBColor(0x66, 0xBB, 0x6A)
LEAF_LIGHT = RGBColor(0xF1, 0xF8, 0xE9)
TEXT_DARK = RGBColor(0x21, 0x21, 0x21)
TEXT_GREY = RGBColor(0x55, 0x5B, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
AMBER = RGBColor(0xB2, 0x6A, 0x00)
RED = RGBColor(0xB7, 0x1C, 0x1C)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def fill_bg(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, color, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
    sh.shadow.inherit = False
    return sh


def add_text(slide, x, y, w, h, text, size=18, color=TEXT_DARK, bold=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.name = font
    return tb


def add_bullets(slide, x, y, w, h, items, size=18, color=TEXT_DARK, gap=6):
    """items: list of (text, bold, color) or plain strings."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if isinstance(item, tuple):
            text, bold = item[0], item[1]
            col = item[2] if len(item) > 2 else color
        else:
            text, bold, col = item, False, color
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = text
        p.space_after = Pt(gap)
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = col
            r.font.name = "Calibri"
    return tb


def add_header(slide, title, kicker=None):
    """Standard content-slide header: green band + title."""
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.0), GREEN_DARK)
    add_text(slide, Inches(0.55), Inches(0.12), Inches(11.5), Inches(0.5),
             title, size=28, color=WHITE, bold=True)
    if kicker:
        add_text(slide, Inches(0.55), Inches(0.62), Inches(12.3), Inches(0.35),
                 kicker, size=13, color=GREEN_ACCENT)


def add_footer(slide, text="CropCare AI — IIIT-H Capstone Project 7, Group 4"):
    add_text(slide, Inches(0.55), Inches(7.12), Inches(9), Inches(0.3),
             text, size=10, color=TEXT_GREY)


def add_image_fit(slide, path, x, y, w, h):
    """Place image centered inside box (x,y,w,h), preserving aspect ratio."""
    with Image.open(path) as im:
        iw, ih = im.size
    box_ratio = w / h
    img_ratio = iw / ih
    if img_ratio > box_ratio:
        new_w = w
        new_h = int(w / img_ratio)
    else:
        new_h = h
        new_w = int(h * img_ratio)
    left = x + (w - new_w) // 2
    top = y + (h - new_h) // 2
    return slide.shapes.add_picture(path, left, top, width=new_w, height=new_h)


def img_card(slide, path, x, y, w, h, caption=None):
    add_rect(slide, x - Emu(28575), y - Emu(28575), w + Emu(57150), h + Emu(57150),
             LEAF_LIGHT, line=GREEN_MID)
    add_image_fit(slide, path, x, y, w, h)
    if caption:
        add_text(slide, x, y + h + Emu(63500), w, Inches(0.35), caption,
                 size=11, color=TEXT_GREY, align=PP_ALIGN.CENTER)


def chart_path(name):
    return os.path.join(ASSETS, "charts", name)


def shot_path(name):
    return os.path.join(ASSETS, "screenshots", name)


# =====================================================================
# 1. Title slide
# =====================================================================
s = add_slide()
fill_bg(s, GREEN_DARK)
add_rect(s, 0, Inches(6.9), SLIDE_W, Inches(0.6), GREEN_MID)
add_text(s, Inches(0.9), Inches(1.7), Inches(11.5), Inches(1.4),
         "AI-Powered Plant Disease Detection\nand Farmer Assistance",
         size=40, color=WHITE, bold=True)
add_text(s, Inches(0.9), Inches(3.4), Inches(11), Inches(0.6),
         "CropCare AI — photograph a leaf, get a diagnosis and safe next steps",
         size=20, color=GREEN_ACCENT)
add_text(s, Inches(0.9), Inches(5.3), Inches(11), Inches(1.2),
         "IIIT-H Capstone Project 7  ·  Group 4\nMobileNetV2 on Hugging Face  ·  LangGraph  ·  FastAPI  ·  Next.js",
         size=16, color=WHITE)
add_text(s, Inches(0.9), Inches(7.02), Inches(12), Inches(0.4),
         "Demo day deck · numbers quoted from model/final_report.md and shared/release.json",
         size=10, color=LEAF_LIGHT)

# =====================================================================
# 2. Problem + pitch
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "The problem — and what we built", "60-second pitch")
add_bullets(s, Inches(0.55), Inches(1.35), Inches(6.6), Inches(4.6), [
    ("Farmers lack fast, safe, understandable crop-disease guidance.", True),
    ("A farmer photographs a leaf; a quality gate checks the photo (dark, blurry, tiny, no leaf) and asks for a retake instead of guessing.", False),
    ("A fine-tuned MobileNetV2 (38 classes, 14 crops) predicts the disease.", False),
    ("A LangGraph pipeline looks the class up in a curated knowledge base; an optional LLM writes a plain-language summary.", False),
    ("The LLM never decides the disease and never sees the image.", True, GREEN_MID),
    ("Low confidence, wrong crop, and off-topic chat are handled explicitly.", False),
], size=17)
add_rect(s, Inches(7.5), Inches(1.5), Inches(5.3), Inches(4.9), LEAF_LIGHT, line=GREEN_MID)
add_text(s, Inches(7.8), Inches(1.75), Inches(4.8), Inches(0.5),
         "Numbers to quote", size=18, color=GREEN_DARK, bold=True)
add_bullets(s, Inches(7.8), Inches(2.35), Inches(4.8), Inches(3.9), [
    ("Lab (PlantVillage held-out): ~98–99%", False),
    ("Field-style locked set: 137/284 = 48.2%  (@v1.1)", True, RED),
    ("Bake-off: MobileNet 48.2% vs ResNet-18 ~34% on the same 284 protocol", False),
    ("Live pin: neghani/plantvillage-mobilenetv2@v1.2  (thr 0.50, T=0.885, crop strategy B)", False),
], size=15)
add_footer(s)

# =====================================================================
# 3. Product in one shot
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Product in one shot", "What the farmer actually sees")
img_card(s, shot_path("03-results-ok.png"), Inches(0.8), Inches(1.35), Inches(3.1), Inches(5.4),
         caption="Results — Squash powdery mildew ~98%, pin @v1.2")
add_bullets(s, Inches(4.6), Inches(1.6), Inches(8.1), Inches(5.0), [
    ("One photo → disease name, confidence, top-3 alternatives", True),
    ("Curated guidance: what to do now, what to watch, monitoring cadence", False),
    ("Follow-up chat stays in the agriculture domain (EN / Telugu toggle)", False),
    ("Vision model decides; knowledge base + LLM explain", False),
    ("History stays in browser sessionStorage — no accounts, no stored images", False),
], size=17)

# =====================================================================
# 4. Architecture
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Architecture", "Vision decides, language explains")
steps = [
    ("1. Next.js UI (:3000)", "Browser shrinks large photos, then POST /quality-check"),
    ("2. Quality gate", "Leaf finder + brightness + blur; fail → retake, no model call"),
    ("3. POST /predict", "Gate re-runs server-side; crop strategy B → 256 px square"),
    ("4. Inference service (:7860)", "Pinned MobileNetV2 @v1.2 from HF Hub; top-3 + scores"),
    ("5. LangGraph agent", "predict → knowledge lookup → report → safety; side exits for error / unsupported crop / not sure"),
    ("6. LLM (text only)", "Farmer summary + chat from KB facts; never the image"),
]
y = Inches(1.3)
for title, desc in steps:
    add_rect(s, Inches(0.55), y, Inches(7.3), Inches(0.82), LEAF_LIGHT, line=GREEN_MID)
    add_text(s, Inches(0.75), y + Inches(0.06), Inches(3.1), Inches(0.7),
             title, size=14, color=GREEN_DARK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, Inches(3.9), y + Inches(0.06), Inches(3.8), Inches(0.7),
             desc, size=11.5, color=TEXT_DARK, anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(0.92)
add_rect(s, Inches(8.2), Inches(1.3), Inches(4.6), Inches(5.5), GREEN_DARK)
add_text(s, Inches(8.5), Inches(1.55), Inches(4.0), Inches(0.5),
         "Design rules worth saying out loud", size=16, color=GREEN_ACCENT, bold=True)
add_bullets(s, Inches(8.5), Inches(2.2), Inches(4.0), Inches(4.4), [
    ("The image never goes to the LLM", True, WHITE),
    ("Two-step UX: cheap gate first; /predict re-runs it server-side (never trust the client)", False, WHITE),
    ("Provider-independent LLM: OpenAI-compatible client, swap via LLM_BASE_URL", False, WHITE),
    ("Pinned model revision in shared/release.json — /health?probe=true must match", False, WHITE),
    ("Fail soft: model down → honest report; LLM down → KB template text", False, WHITE),
], size=12.5, gap=8)
add_footer(s)

# =====================================================================
# 5. Lab vs field
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Lab vs field — keep the two numbers apart",
           "Do not jump from 99% to 48% in one breath")
img_card(s, chart_path("lab-vs-field.png"), Inches(0.7), Inches(1.35), Inches(5.6), Inches(5.4),
         caption="Lab PlantVillage held-out vs locked 284-image field-style set")
add_bullets(s, Inches(6.9), Inches(1.7), Inches(5.9), Inches(4.8), [
    ("Lab ~98–99% is the in-distribution ceiling — not the demo headline", True),
    ("PlantVillage has near-duplicate leaves and a uniform background → leakage + shortcut learning", False),
    ("Field-style locked set: 137/284 (48.2%) from MobileNet @v1.1, one-shot, never re-tuned after seeing the number", False),
    ("The 284-image set is burned — we refuse to re-run final_test on it", True, GREEN_MID),
    ("Quote both numbers, always together", False),
], size=16)

# =====================================================================
# 6. Training story
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Training story — ImageNet → PlantVillage → PlantDoc",
           "18.7% → 48.2% field pass rate, stage by stage")
img_card(s, chart_path("training-stages.png"), Inches(0.6), Inches(1.35), Inches(6.0), Inches(5.3),
         caption="Stages: v1.0 PV → v1.1 PlantDoc stage 2 → v1.2 sampler fix")
img_card(s, chart_path("field-pass-journey.png"), Inches(7.0), Inches(1.35), Inches(5.6), Inches(4.3),
         caption="Field pass-rate journey (locked 284 protocol)")
add_text(s, Inches(7.0), Inches(5.9), Inches(5.9), Inches(1.1),
         "Stage 2: lower LR from v1.0, label smoothing 0.1, PlantDoc mix in train,\nsampler bug fixed in v1.2 (live pin). Threshold / temperature from PlantDoc val.",
         size=12.5, color=TEXT_DARK)

# =====================================================================
# 7. Field failures
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Where the field set still fails", "Per-class zeros on the locked 284")
img_card(s, chart_path("per-class-field-failures.png"), Inches(0.7), Inches(1.35), Inches(5.6), Inches(5.4),
         caption="Classes the field model gets wrong (tomato bacterial spot, mosaic, cherry healthy …)")
add_bullets(s, Inches(6.9), Inches(1.7), Inches(5.9), Inches(4.6), [
    ("Look-alike foliar diseases confuse the model in the lab too (early blight vs Septoria vs target spot)", False),
    ("Wrong-crop confidence is still open: strawberry → tomato style failures remain", True, RED),
    ("OOD margin (0.05) + crop-hint check + quality gate reduce — but do not eliminate — confident nonsense", False),
    ("Next step is a crop-level head / negatives — not another backbone flip", True, GREEN_MID),
], size=16)

# =====================================================================
# 8. Lab confusion matrix
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Lab confusion matrix (in-distribution)", "PlantVillage held-out split")
img_card(s, chart_path("confusion-matrix-lab.png"), Inches(0.6), Inches(1.3), Inches(5.9), Inches(5.6),
         caption="38 classes; confusion concentrated in look-alike blights and corn leaf spots")
add_bullets(s, Inches(7.0), Inches(1.7), Inches(5.8), Inches(4.6), [
    ("Top-3 accuracy 99.94% — the right answer is almost always among the three shown", True),
    ("Weak spots: Potato healthy recall 73% but support only 15 images; Tomato early blight precision 90%", False),
    ("Confidence threshold 0.5 chosen by rule: accuracy ≥ 97% on covered images, coverage ≥ 50%", False),
    ("Honest caveat: threshold tuned on the easy lab split, before the field number existed", False, AMBER),
], size=16)

# =====================================================================
# 9. Quality gate
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Quality gate — a bad photo never reaches the model",
           "Retake instead of a wrong guess")
img_card(s, shot_path("04-retake.png"), Inches(0.8), Inches(1.35), Inches(3.1), Inches(5.4),
         caption="Too dark → retake guidance, no model call")
add_bullets(s, Inches(4.6), Inches(1.6), Inches(8.2), Inches(5.2), [
    ("v0 gate failed in the field: whole-frame Laplacian variance depends on resolution and background — sharp phone photos were rejected", False),
    ("v1 (live): HSV vegetation mask finds the leaf, drops out-of-focus vegetation, picks the best leaf by size / sharpness / centrality", False),
    ("Sharpness measured leaf-only at fixed scale via re-blur ratio + absolute-detail floor (ratio < 0.78 or variance < 25 → reject)", True),
    ("Handles EXIF rotation, alpha channels, 48 MP photos, decompression bombs; every rejection is a farmer-friendly instruction", False),
    ("Caveat: thresholds hand-calibrated on a blur sweep + ~40 PV images — no labelled good/bad phone set yet", False, AMBER),
], size=15.5)

# =====================================================================
# 10. Not-sure path
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "“Not sure” is a feature, not a failure",
           "Softmax is always confident — we abstain on purpose")
img_card(s, shot_path("02-not-sure.png"), Inches(0.8), Inches(1.35), Inches(5.4), Inches(5.4),
         caption="Results — “I'm not confident yet” path")
add_bullets(s, Inches(6.7), Inches(1.7), Inches(6.1), Inches(4.8), [
    ("Softmax over 38 classes always produces a confident-looking answer — even for a shoe", True),
    ("Below threshold (0.5) or flat distribution (OOD margin 0.05) → “Possible match” / not-sure path", False),
    ("Top-3 shown to the farmer so ambiguity is visible rather than hidden", False),
    ("Crop-hint mismatch or unsupported crop downgrades the report and forces an expert-confirmation flag", False),
    ("Better next step: label ~100 phone photos good/bad and pick gate thresholds from an ROC curve", False),
], size=16)

# =====================================================================
# 11. Agentic safety
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Agentic AI — with guardrails", "Orchestrated workflow, deliberate safety")
add_bullets(s, Inches(0.55), Inches(1.4), Inches(6.7), Inches(5.0), [
    ("LangGraph: fixed graph with conditional exits — error, unsupported crop, retake, not sure", False),
    ("predict → knowledge → generate_report → safety; safety node can downgrade “confirmed” to “possible”", False),
    ("Grounding: structured fields come from a curated 38-entry JSON knowledge base — no free-form LLM facts", False),
    ("Prompt forbids naming chemicals or doses; curated actions carry the real guidance", False),
    ("Chat topic-guarded: keyword pre-filter + system-prompt refusal (classifier guard is future work)", False),
    ("No key / outage → KB template text; the demo works without an LLM", False),
], size=16)
add_rect(s, Inches(7.6), Inches(1.5), Inches(5.2), Inches(5.0), GREEN_DARK)
add_text(s, Inches(7.9), Inches(1.8), Inches(4.7), Inches(0.6),
         "The LLM never sees the image", size=20, color=GREEN_ACCENT, bold=True)
add_bullets(s, Inches(7.9), Inches(2.6), Inches(4.7), Inches(3.6), [
    ("generate_text() raises on non-string input — enforced in code, not just policy", False, WHITE),
    ("Vision decides; language explains", True, WHITE),
    ("Avoids vision-LLM hallucination and cost", False, WHITE),
    ("Honest answer to “is it agentic?”: orchestrated workflow with deterministic tools and one generative step — chosen for auditability in a medical-advice-like domain", False, WHITE),
], size=14, gap=8)
add_footer(s)

# =====================================================================
# 12. Bake-off
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Model bake-off — MobileNet vs ResNet-18",
           "Same 284 images, same protocol (crop strategy B)")
# simple bar comparison
bar_x, bar_y, bar_max_w, bar_h = Inches(1.0), Inches(2.2), Inches(10.5), Inches(0.9)
add_text(s, Inches(1.0), Inches(1.6), Inches(4), Inches(0.4), "Field one-shot pass rate",
         size=14, color=TEXT_GREY)
add_rect(s, bar_x, bar_y, Emu(int(bar_max_w * 0.482)), bar_h, GREEN_MID)
add_text(s, bar_x + Emu(int(bar_max_w * 0.482)) + Inches(0.15), bar_y + Inches(0.2),
         Inches(3), Inches(0.5), "MobileNetV2  137/284 = 48.2%", size=16, color=GREEN_DARK, bold=True)
add_rect(s, bar_x, bar_y + Inches(1.3), Emu(int(bar_max_w * 0.345)), bar_h, TEXT_GREY)
add_text(s, bar_x + Emu(int(bar_max_w * 0.345)) + Inches(0.15), bar_y + Inches(1.5),
         Inches(4), Inches(0.5), "ResNet-18  ~98/284 ≈ 34%", size=16, color=TEXT_GREY, bold=True)
add_bullets(s, Inches(1.0), Inches(5.0), Inches(11.8), Inches(1.9), [
    ("ResNet was stronger on clean lab photos but did not transfer to the field set — more “not sure” answers", False),
    ("MobileNetV2 stays on the happy path: ~9 MB, fast on CPU, matches the brief; ResNet artefacts kept as comparison only", False),
    ("Details: docs/model-bakeoff.md", False, TEXT_GREY),
], size=15)
add_footer(s)

# =====================================================================
# 13. Challenges & decisions
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Top engineering challenges", "What bit us, and what we did")
rows = [
    ("Phone photos are big and rotated", "JPEG draft decode, EXIF transpose, working copy capped at 2048 px"),
    ("Blur metrics lie", "Leaf-only, fixed-scale re-blur ratio with absolute-detail floor"),
    ("Backgrounds confuse a lab-trained model", "Leaf localisation + padded square crop, resize once to 256"),
    ("Softmax is overconfident", "Threshold sweep, top-3 display, crop-hint cross-check, “possible match” downgrade"),
    ("LLMs invent things", "Curated KB grounding, no chemicals/doses in prompt, text-only interface"),
    ("HF serverless can't host custom fine-tunes", "Serve pinned revision from our own container (hf_space/), provenance headers"),
    ("Train/serve skew", "Same processor config: resize 256 → center-crop 224; API sizes the crop to exactly 256"),
]
y = Inches(1.3)
for challenge, fix in rows:
    add_rect(s, Inches(0.55), y, Inches(12.25), Inches(0.72), LEAF_LIGHT, line=GREEN_MID)
    add_text(s, Inches(0.75), y + Inches(0.04), Inches(4.6), Inches(0.64),
             challenge, size=13, color=GREEN_DARK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, Inches(5.5), y + Inches(0.04), Inches(7.2), Inches(0.64),
             fix, size=12, color=TEXT_DARK, anchor=MSO_ANCHOR.MIDDLE)
    y += Inches(0.8)
add_footer(s)

# =====================================================================
# 14. Limitations
# =====================================================================
s = add_slide()
fill_bg(s)
add_header(s, "Honest limitations", "We own the gaps before you find them")
add_bullets(s, Inches(0.55), Inches(1.45), Inches(6.4), Inches(5.2), [
    ("Field one-shot is 48.2% — below the proposal's 50% success criterion (reported as-is)", True, RED),
    ("Lab 99% is optimistic: near-duplicate leakage + uniform background in PlantVillage", False),
    ("Wrong-crop confidence still open despite OOD margin and crop-hint checks", False),
    ("Quality-gate thresholds hand-calibrated, not learned from labelled good/bad photos", False),
    ("“Agentic” is an orchestrated workflow, not an autonomous agent — deliberate safety choice", False),
    ("Chat topic guard is a keyword list — brittle; classifier guard is future work", False),
], size=16)
add_bullets(s, Inches(7.4), Inches(1.45), Inches(5.4), Inches(5.2), [
    ("KB entries carry thin source citations", False),
    ("Hindi was promised — we shipped a Telugu MVP (EN/TE toggle + glossary-backed translation); IndicTrans is the heavier follow-up", False),
    ("Proposal drift, owned: Streamlit → Next.js; Grok → OpenAI-compatible client; two services → three in Compose", False),
    ("No auth, no persistent image storage, no server-side history — explicit demo scope", False),
], size=16)
add_footer(s)

# =====================================================================
# 15. Next steps + close
# =====================================================================
s = add_slide()
fill_bg(s, GREEN_DARK)
add_rect(s, 0, 0, SLIDE_W, Inches(1.0), GREEN_MID)
add_text(s, Inches(0.55), Inches(0.2), Inches(12), Inches(0.6),
         "Next steps — and thank you", size=30, color=WHITE, bold=True)
add_bullets(s, Inches(0.7), Inches(1.5), Inches(6.6), Inches(4.6), [
    ("Cluster-based splits to kill near-duplicate leakage", True, WHITE),
    ("Phone-like training data (test set stays locked)", False, WHITE),
    ("Crop-level head / negatives for wrong-crop OOD", False, WHITE),
    ("KB source citations (ICAR / state agri universities)", False, WHITE),
    ("Learned quality-gate thresholds from a labelled set", False, WHITE),
    ("Real agentic step: tool-calling chat (lookup_disease, get_prediction)", False, WHITE),
], size=17, gap=10)
add_rect(s, Inches(7.7), Inches(1.5), Inches(5.1), Inches(4.6), GREEN_MID)
add_text(s, Inches(8.0), Inches(1.8), Inches(4.6), Inches(0.5),
         "Try it", size=18, color=GREEN_ACCENT, bold=True)
add_bullets(s, Inches(8.0), Inches(2.4), Inches(4.6), Inches(3.4), [
    ("UI  →  localhost:3000/check", False, WHITE),
    ("API  →  localhost:8000/docs", False, WHITE),
    ("Health + pin  →  /health?probe=true", False, WHITE),
    ("Model  →  huggingface.co/neghani/plantvillage-mobilenetv2 @v1.2", False, WHITE),
], size=14, gap=8)
add_text(s, Inches(0.7), Inches(6.4), Inches(12), Inches(0.8),
         "Treatment suggestions are general guidance only and must be validated with local agricultural experts before critical action.",
         size=12, color=LEAF_LIGHT)

OUT = os.path.join(ROOT, "CropCare-AI-Demo-Day.pptx")
prs.save(OUT)
print("wrote", OUT, f"({len(prs.slides.__iter__.__self__._sldIdLst)} slides)" if False else "")
print("slides:", len(prs.slides._sldIdLst))
