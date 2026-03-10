import operator
from typing import Annotated, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from src.rag import get_retriever
from src.utils import get_llm


class AgentState(TypedDict):
    input_text: str
    parsed_data: dict
    retrieved_context: str
    solution_plan: str
    final_answer: str
    verification_status: str
    critique: str
    messages: Annotated[List[str], operator.add]


llm = get_llm()


def parser_node(state: AgentState):
    """Agent 1: Parser - clean input and create structured problem object."""
    print("--- 1. PARSER AGENT ---")

    parsed_data = {
        "problem": state["input_text"],
        "topic": "Math",
        "needs_clarification": False,
    }
    return {"parsed_data": parsed_data, "messages": ["Parser: Processed input."]}


def solver_node(state: AgentState):
    """Agent 2: Solver - retrieve RAG context and draft solution."""
    print("--- 2. SOLVER AGENT ---")

    retriever = get_retriever()

    docs = retriever.invoke(state["parsed_data"]["problem"])
    context = "\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a JEE Math Tutor. Solve the problem using the provided "
                "context. Show all steps clearly.",
            ),
            (
                "user",
                "Problem: {problem}\n\n"
                "Use this context if helpful:\n{context}",
            ),
        ]
    )
    response = llm.invoke(
        prompt.format(
            problem=state["parsed_data"]["problem"],
            context=context,
        )
    )

    return {"solution_plan": response.content, "retrieved_context": context}


def verifier_node(state: AgentState):
    """Agent 3: Verifier - check solution correctness."""
    print("--- 3. VERIFIER AGENT ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict math solution verifier. "
                "Check the solution for correctness. "
                "Return ONLY 'APPROVED' if fully correct, or 'REJECTED' "
                "followed by a brief critique if there are issues.",
            ),
            (
                "user",
                "Problem: {problem}\n\nProposed solution:\n{solution}",
            ),
        ]
    )
    response = llm.invoke(
        prompt.format(
            problem=state["parsed_data"]["problem"],
            solution=state["solution_plan"],
        )
    )
    content_upper = response.content.upper()
    status = "rejected" if "REJECTED" in content_upper and "APPROVED" not in content_upper else "approved"
    return {"verification_status": status, "critique": response.content}


def explainer_node(state: AgentState):
    """Agent 4: Explainer - format final explanation."""
    print("--- 4. EXPLAINER AGENT ---")

    base_solution = state["solution_plan"]
    critique = state.get("critique", "")

    if state.get("verification_status") == "rejected":
        combined = (
            "The following solution was critiqued:\n\n"
            f"{base_solution}\n\n"
            "Verifier critique:\n"
            f"{critique}\n\n"
            "Please correct any mistakes and present a final, fully correct "
            "step-by-step solution for the student."
        )
    else:
        combined = base_solution

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a patient JEE math mentor. "
                "Explain the solution clearly with steps and key insights.",
            ),
            ("user", "Solution (possibly with verifier feedback):\n{solution}"),
        ]
    )
    response = llm.invoke(prompt.format(solution=combined))
    return {"final_answer": response.content}


workflow = StateGraph(AgentState)
workflow.add_node("parser", parser_node)
workflow.add_node("solver", solver_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("explainer", explainer_node)

workflow.set_entry_point("parser")
workflow.add_edge("parser", "solver")
workflow.add_edge("solver", "verifier")
workflow.add_edge("verifier", "explainer")
workflow.add_edge("explainer", END)

app_graph = workflow.compile()

