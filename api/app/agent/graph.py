from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    generate_report_node,
    knowledge_node,
    predict_error_node,
    predict_node,
    safety_node,
    unsupported_crop_node,
)
from app.agent.state import DiagnosisState


def _route_after_predict(state: DiagnosisState) -> str:
    if state.get("error"):
        return "predict_error"
    if state.get("crop_flag") == "unsupported":
        return "unsupported_crop"
    return "knowledge"


def build_diagnosis_graph():
    """Runs after the API-layer quality gate.

    predict -> knowledge -> generate_report -> safety, with two side exits that skip the
    disease write-up: the classifier failed, or the farmer's crop is one it was never trained on.
    """
    graph = StateGraph(DiagnosisState)
    graph.add_node("predict", predict_node)
    graph.add_node("predict_error", predict_error_node)
    graph.add_node("unsupported_crop", unsupported_crop_node)
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
            "knowledge": "knowledge",
        },
    )
    graph.add_edge("predict_error", "safety")
    graph.add_edge("unsupported_crop", "safety")
    graph.add_edge("knowledge", "generate_report")
    graph.add_edge("generate_report", "safety")
    graph.add_edge("safety", END)

    return graph.compile()


diagnosis_graph = build_diagnosis_graph()
