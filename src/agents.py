import json
import operator
from typing import Annotated, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from src.rag import get_retriever
from src.utils import get_llm


class AgentState(TypedDict):
    input_text: str
    input_type: str  # "text" | "image" | "audio"
    parsed_data: dict  # problem_text, topic, variables, constraints, needs_clarification
    retrieved_context: str
    solution_plan: str
    final_answer: str
    verification_status: str  # "approved" | "rejected" | "uncertain"
    verification_confidence: str  # for UI
    critique: str
    messages: Annotated[List[str], operator.add]


llm = get_llm()

# Topics we support (JEE scope)
ALLOWED_TOPICS = ["algebra", "probability", "calculus", "linear_algebra"]


def _normalize_parsed_data(data: dict, raw: str) -> dict:
    """Ensure parsed_data has required keys; accept 'problem' or 'problem_text'."""
    if not isinstance(data, dict):
        return {
            "problem_text": raw,
            "topic": "algebra",
            "variables": [],
            "constraints": [],
            "needs_clarification": False,
        }
    # LLM may return "problem" instead of "problem_text"
    problem_text = data.get("problem_text") or data.get("problem") or raw
    if not isinstance(problem_text, str):
        problem_text = str(problem_text) if problem_text else raw
    topic = data.get("topic") or "algebra"
    if topic not in ALLOWED_TOPICS:
        topic = "algebra"
    return {
        "problem_text": problem_text,
        "topic": topic,
        "variables": data.get("variables") if isinstance(data.get("variables"), list) else [],
        "constraints": data.get("constraints") if isinstance(data.get("constraints"), list) else [],
        "needs_clarification": bool(data.get("needs_clarification", False)),
    }


def parser_node(state: AgentState):
    """Agent 1: Parser - clean input, structured problem, detect ambiguity."""
    print("--- 1. PARSER AGENT ---")
    raw = (state.get("input_text") or "").strip()
    if not raw:
        data = _normalize_parsed_data({}, raw)
        return {"parsed_data": data, "messages": ["Parser: No input."]}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a math problem parser. Clean OCR/ASR or typed input and output a structured JSON.\n"
                "Rules: Extract problem_text (cleaned), topic (one of: algebra, probability, calculus, linear_algebra), "
                "variables (list of symbols e.g. [\"x\", \"y\"] or []), constraints (e.g. [\"x > 0\"] or []). "
                "Set needs_clarification to true ONLY if the problem is ambiguous, has missing info, or is unreadable. "
                "Reply with ONLY valid JSON in this exact shape:\n"
                '{"problem_text": "...", "topic": "...", "variables": [], "constraints": [], "needs_clarification": false}',
            ),
            ("user", "Raw input:\n{raw}"),
        ]
    )
    try:
        response = llm.invoke(prompt.format(raw=raw))
        text = (response.content or "").strip()
    except Exception:
        data = _normalize_parsed_data({}, raw)
        return {"parsed_data": data, "messages": ["Parser: Fallback (LLM error)."]}

    # Extract JSON (handle markdown code blocks)
    if "```" in text:
        parts = text.split("```")
        for p in parts[1:]:
            p = p.replace("json", "").strip()
            if p.startswith("{"):
                text = p
                break
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
    data = _normalize_parsed_data(data, raw)
    return {"parsed_data": data, "messages": ["Parser: Structured problem."]}


def intent_router_node(state: AgentState):
    """Agent 2: Intent Router - classify problem type and route workflow."""
    print("--- 2. INTENT ROUTER AGENT ---")
    parsed = state.get("parsed_data") or {}
    topic = parsed.get("topic", "algebra")
    # Route: all go to solver; topic is used by solver for RAG/context
    return {"messages": [f"Intent Router: Classified as {topic}."]}


def solver_node(state: AgentState):
    """Agent 3: Solver - retrieve RAG context and draft solution."""
    print("--- 3. SOLVER AGENT ---")
    parsed = state.get("parsed_data") or {}
    problem = (parsed.get("problem_text") or parsed.get("problem") or
               state.get("input_text") or "")
    if not isinstance(problem, str):
        problem = str(problem) if problem else ""

    retriever = get_retriever()
    docs = retriever.invoke(problem)
    context = "\n".join([d.page_content for d in docs]) if docs else "(No RAG context available.)"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a JEE Math Tutor. Solve the problem using the provided context when relevant. Show all steps clearly.",
            ),
            ("user", "Problem: {problem}\n\nContext (use if helpful):\n{context}"),
        ]
    )
    response = llm.invoke(prompt.format(problem=problem, context=context))
    return {"solution_plan": response.content, "retrieved_context": context}


def verifier_node(state: AgentState):
    """Agent 4: Verifier/Critic - check correctness, units, edge cases; can trigger HITL if uncertain."""
    print("--- 4. VERIFIER AGENT ---")
    parsed = state.get("parsed_data") or {}
    problem = (parsed.get("problem_text") or parsed.get("problem") or
               state.get("input_text") or "")
    if not isinstance(problem, str):
        problem = str(problem) if problem else ""
    solution = state.get("solution_plan", "")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict math solution verifier. Check: correctness, units/domain, edge cases. "
                "Reply with exactly one of: APPROVED | REJECTED | UNCERTAIN. "
                "If APPROVED, add one short line why. If REJECTED or UNCERTAIN, add a brief critique. "
                "Use UNCERTAIN when you are not confident (e.g. ambiguous problem or borderline solution).",
            ),
            ("user", "Problem: {problem}\n\nSolution:\n{solution}"),
        ]
    )
    response = llm.invoke(prompt.format(problem=problem, solution=solution))
    content = response.content.upper()
    if "REJECTED" in content and "APPROVED" not in content.split("REJECTED")[0]:
        status = "rejected"
    elif "UNCERTAIN" in content:
        status = "uncertain"
    else:
        status = "approved"
    return {
        "verification_status": status,
        "verification_confidence": response.content,
        "critique": response.content,
    }


def explainer_node(state: AgentState):
    """Agent 5: Explainer - step-by-step, student-friendly explanation."""
    print("--- 5. EXPLAINER AGENT ---")
    base_solution = state.get("solution_plan", "")
    critique = state.get("critique", "")
    status = state.get("verification_status", "approved")

    if status == "rejected":
        combined = (
            "The following solution was critiqued:\n\n"
            f"{base_solution}\n\nVerifier: {critique}\n\n"
            "Provide a corrected, full step-by-step solution."
        )
    elif status == "uncertain":
        combined = (
            f"{base_solution}\n\n(Verifier was uncertain: {critique}. "
            "Present the solution clearly and note that human verification is recommended.)"
        )
    else:
        combined = base_solution

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a patient JEE math mentor. Explain the solution clearly with steps and key insights.",
            ),
            ("user", "{solution}"),
        ]
    )
    response = llm.invoke(prompt.format(solution=combined))
    return {"final_answer": response.content}


workflow = StateGraph(AgentState)
workflow.add_node("parser", parser_node)
workflow.add_node("intent_router", intent_router_node)
workflow.add_node("solver", solver_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("explainer", explainer_node)

workflow.set_entry_point("parser")
workflow.add_edge("parser", "intent_router")
workflow.add_edge("intent_router", "solver")
workflow.add_edge("solver", "verifier")
workflow.add_edge("verifier", "explainer")
workflow.add_edge("explainer", END)

app_graph = workflow.compile()
