from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    check_quality_node,
    generate_report_node,
    knowledge_node,
    predict_node,
    quality_fail_node,
    safety_node,
)
from app.agent.state import DiagnosisState


def _route_after_quality(state: DiagnosisState) -> str:
    if not state.get("quality_ok", False):
        return "quality_fail"
    return "predict"


def build_diagnosis_graph():
    graph = StateGraph(DiagnosisState)
    graph.add_node("check_quality", check_quality_node)
    graph.add_node("quality_fail", quality_fail_node)
    graph.add_node("predict", predict_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("safety", safety_node)

    graph.add_edge(START, "check_quality")
    graph.add_conditional_edges(
        "check_quality",
        _route_after_quality,
        {"quality_fail": "quality_fail", "predict": "predict"},
    )
    graph.add_edge("quality_fail", "safety")
    graph.add_edge("predict", "knowledge")
    graph.add_edge("knowledge", "generate_report")
    graph.add_edge("generate_report", "safety")
    graph.add_edge("safety", END)

    return graph.compile()


diagnosis_graph = build_diagnosis_graph()
