"""
Graph Builder - LangGraph Workflow Construction

Features:
    Orchestrate Multi-Agent workflow using LangGraph
    Implement conditional routing for Self-Correction mechanism
"""
from typing import Literal
from langgraph.graph import StateGraph, END

from .state import DataCleaningState
from ..nodes.profiler import profiler_node
from ..nodes.coder import coder_node
from ..nodes.executor import executor_node
from ..nodes.qa import qa_node
from ..nodes.eda import eda_node


# ========== Routing Functions ==========
def route_after_executor(state: DataCleaningState) -> Literal["coder", "qa", "end"]:
    """
    Conditional routing after Executor Node

    Logic:
        - If execution failed (execution_success=False):
            - retry_count < 3: return "coder" (Self-Correction)
            - retry_count >= 3: return "end" (retry limit exceeded)
        - If execution succeeded: return "qa" (proceed to QA)

    Args:
        state: Global state

    Returns:
        Name of the next node
    """
    execution_success = state.get("execution_success", False)
    retry_count = state.get("retry_count", 0)

    if not execution_success:
        if retry_count < 3:
            print(f"\n[RETRY] Execution failed, entering Self-Correction (attempt {retry_count + 1})")
            return "coder"
        else:
            print(f"\n[ABORT] Retry limit reached ({retry_count} attempts), terminating workflow")
            return "end"
    else:
        print(f"\n[SUCCESS] Execution succeeded, proceeding to QA stage")
        return "qa"


def route_after_qa(state: DataCleaningState) -> Literal["eda", "end"]:
    """
    Conditional routing after QA Node

    Logic:
        - If QA passed (qa_result["passed"]=True): return "eda" (proceed to EDA analysis)
        - If QA failed: return "end" (terminate workflow)

    Args:
        state: Global state

    Returns:
        Name of the next node
    """
    qa_result = state.get("qa_result") or {}
    passed = qa_result.get("passed", False)

    if passed:
        print(f"\n[SUCCESS] QA passed, proceeding to EDA analysis stage")
        return "eda"
    else:
        print(f"\n[SKIP] QA failed, skipping EDA analysis")
        return "end"


# ========== Build Graph ==========
def build_graph() -> StateGraph:
    """
    Build LangGraph workflow

    Flow:
        START → Profiler → Coder → Executor → [conditional routing]
                                       ↑         ↓
                                       └─────────┤
                                    (Self-Correction)
                                                 ↓
                                                QA → [conditional routing]
                                                 ↓         ↓
                                                EDA       END
                                                 ↓
                                                END

    Returns:
        Compiled LangGraph application
    """
    # 1. Create StateGraph
    workflow = StateGraph(DataCleaningState)

    # 2. Add nodes
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("eda", eda_node)

    # 3. Set entry point
    workflow.set_entry_point("profiler")

    # 4. Add fixed edges
    workflow.add_edge("profiler", "coder")      # Profiler → Coder
    workflow.add_edge("coder", "executor")      # Coder → Executor
    workflow.add_edge("eda", END)               # EDA → END

    # 5. Add conditional edges (core of Self-Correction)
    workflow.add_conditional_edges(
        "executor",                              # From Executor node
        route_after_executor,                    # Use routing function
        {
            "coder": "coder",                    # Retry: back to Coder
            "qa": "qa",                          # Success: proceed to QA
            "end": END                           # Limit exceeded: force end
        }
    )

    # 6. Add conditional edges after QA (EDA routing)
    workflow.add_conditional_edges(
        "qa",                                    # From QA node
        route_after_qa,                          # Use routing function
        {
            "eda": "eda",                        # QA passed: proceed to EDA
            "end": END                           # QA failed: end
        }
    )

    return workflow


# ========== Compile Graph ==========
# Build and compile graph
graph = build_graph()
app = graph.compile()


# ========== Visualization Tool (Optional) ==========
def visualize_graph(save_path: str = "docs/graph.png") -> None:
    """
    Visualize workflow graph (requires pygraphviz)

    Args:
        save_path: Save path
    """
    try:
        from pathlib import Path

        # Get Mermaid diagram
        mermaid_graph = app.get_graph().draw_mermaid()

        # Save as text file
        output_path = Path(save_path).with_suffix(".mmd")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(mermaid_graph)

        print(f"[SUCCESS] Workflow graph saved: {output_path}")
        print(f"   View with Mermaid tool: https://mermaid.live/")

    except Exception as e:
        print(f"[WARNING] Visualization failed: {str(e)}")
