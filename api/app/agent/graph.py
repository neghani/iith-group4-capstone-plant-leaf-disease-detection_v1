from langgraph.graph import END, START, StateGraph

from app.demo_log import demo
from app.agent.nodes import (
    generate_report_node,
    knowledge_node,
    not_sure_node,
    predict_error_node,
    predict_node,
    retake_node,
    safety_node,
    unsupported_crop_node,
)
from app.agent.state import DiagnosisState


def _route_after_predict(state: DiagnosisState) -> str:
    if state.get("error"):
        demo("AGENT", "→", "route: predict_error — no fabricated write-up will be produced")
        return "predict_error"
    if state.get("crop_flag") == "unsupported":
        demo("AGENT", "→", f"route: unsupported_crop — crop '{state.get('crop_hint', '')}' was never trained on")
        return "unsupported_crop"
    if state.get("model_status") == "retake_photo":
        demo("AGENT", "→", "route: retake — photo quality too low to classify")
        return "retake"
    if state.get("model_status") == "not_sure":
        demo("AGENT", "→", "route: not_sure — confidence/margin below threshold")
        return "not_sure"
    demo("AGENT", "→", "route: knowledge — confident prediction, continuing the graph")
    return "knowledge"


def build_diagnosis_graph():
    """Runs after the API-layer quality gate.

    predict -> knowledge -> generate_report -> safety, with side exits that skip the
    disease write-up: classifier failure, unsupported crop, quality retake, or not sure.
    """
    graph = StateGraph(DiagnosisState)
    graph.add_node("predict", predict_node)
    graph.add_node("predict_error", predict_error_node)
    graph.add_node("unsupported_crop", unsupported_crop_node)
    graph.add_node("retake", retake_node)
    graph.add_node("not_sure", not_sure_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("safety", safety_node)

    graph.add_edge(START, "predict")
    graph.add_conditional_edges(
        "predict",
        _route_after_predict,
        {
            "predict_error": "predict_error",
            "unsupported_crop": "unsupported_crop",
            "retake": "retake",
            "not_sure": "not_sure",
            "knowledge": "knowledge",
        },
    )
    graph.add_edge("predict_error", "safety")
    graph.add_edge("unsupported_crop", "safety")
    graph.add_edge("retake", "safety")
    graph.add_edge("not_sure", "safety")
    graph.add_edge("knowledge", "generate_report")
    graph.add_edge("generate_report", "safety")
    graph.add_edge("safety", END)

    return graph.compile()


diagnosis_graph = build_diagnosis_graph()
