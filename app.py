import streamlit as st
import os
import json

from src.agents import app_graph
from src.utils import perform_ocr, transcribe_audio

# Vector DB is built lazily when first needed (see src.rag.get_retriever).
# No startup init so the app runs on Spaces even without a pre-built chroma_db.

st.set_page_config(page_title="Math Mentor AI", layout="wide")
st.title("🧮 Reliable Multimodal Math Mentor")


MEMORY_FILE = "memory.json"


def load_memory():
    """Load memory entries. Each can have: question, answer, input_type, parsed_question, retrieved_context_snippet, verifier_outcome, user_feedback, topic."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_memory(question, answer, meta=None):
    """Save entry with full schema: original input (question), answer, input_type, parsed_question, retrieved_context_snippet, verifier_outcome, user_feedback, topic."""
    history = load_memory()
    entry = {"question": question, "answer": answer}
    if meta:
        for k, v in meta.items():
            if v is None:
                continue
            if k == "retrieved_context" and len(str(v)) > 500:
                entry["retrieved_context_snippet"] = str(v)[:500] + "..."
            elif k == "parsed_data" and isinstance(v, dict):
                entry["parsed_question"] = v.get("problem_text") or question
                entry["topic"] = v.get("topic")
            else:
                entry[k] = v
    history.append(entry)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def find_similar_solution(user_input):
    """Memory reuse: return (answer, matched_entry) if we have a similar solved problem, else (None, None)."""
    history = load_memory()
    user_lower = user_input.strip().lower()
    for entry in history:
        q = (entry.get("question") or entry.get("parsed_question") or "").strip().lower()
        if not q:
            continue
        if user_lower in q or q in user_lower:
            return entry.get("answer"), entry
    return None, None


with st.sidebar:
    st.header("⚙️ Debug & Options")
    show_trace = st.checkbox(
        "Show Agent Trace",
        value=True,
        help="See what the agents are thinking",
    )
    memory_entries = load_memory()
    st.caption(f"🧠 **Memory:** {len(memory_entries)} entries")
    with st.expander("How to test memory"):
        st.markdown(
            "1. **Solve** a problem (e.g. *integrate 7x^7+8x+4*).  \n"
            "2. Click **✅** on the answer to **save to memory**.  \n"
            "3. Ask the **same or very similar** question again.  \n"
            "4. You should see **\"Answer from memory (pattern reuse)\"** and the stored solution."
        )
    st.markdown("### Retrieved Context")
    context_placeholder = st.empty()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_final_state" not in st.session_state:
    st.session_state.last_final_state = None


input_method = st.radio(
    "Input Method:", ["Text", "Image", "Audio"], horizontal=True
)

if input_method == "Text":
    st.session_state.input_text = st.text_area(
        "Type problem:",
        value=st.session_state.input_text,
        height=100,
    )

elif input_method == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded", width=300)
        if st.button("Extract Text"):
            with st.spinner("Extracting..."):
                text = perform_ocr(uploaded_file)
                st.session_state.input_text = text
                st.rerun()

elif input_method == "Audio":
    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])
    if audio_file:
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                text = transcribe_audio(audio_file)
                st.session_state.input_text = text
                st.rerun()


if st.session_state.input_text:
    user_input = st.text_area(
        "Confirm Question (Edit if needed):",
        value=st.session_state.input_text,
        key="final_input",
        height=120,
    )
    st.caption("Edit the text above if OCR/transcript is wrong (HITL: extraction preview).")

    if st.button("🚀 Solve"):
        cached_answer, matched_entry = find_similar_solution(user_input)

        if cached_answer is not None:
            st.success("🧠 Using memory — similar problem found (pattern reuse).")
            input_src = (matched_entry or {}).get("input_type") or "text"
            memory_note = (
                "📌 **Answer from memory (pattern reuse)** — I've seen a similar problem before. "
                f"Reusing stored solution _(originally from {input_src} input)_.\n\n---\n\n"
            )
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": memory_note + cached_answer,
                    "meta": {"confidence": "memory", "from_memory": True},
                }
            )
        else:
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            with st.status(
                "🤖 Agent Workflow Running...", expanded=show_trace
            ) as status:
                try:
                    inputs = {
                        "input_text": user_input,
                        "input_type": input_method.lower(),
                        "messages": [],
                    }
                    if show_trace:
                        st.write("1️⃣ **Parser Agent:** Cleaning & structuring problem...")
                        st.write("2️⃣ **Intent Router Agent:** Classifying problem type...")
                        st.write("3️⃣ **Solver Agent:** Retrieving RAG context & solving...")

                    final_state = app_graph.invoke(inputs)
                    st.session_state.last_final_state = final_state
                    st.session_state.last_input_type = input_method.lower()

                    if show_trace:
                        ver_status = final_state.get("verification_status", "approved")
                        if ver_status == "approved":
                            st.write("4️⃣ **Verifier Agent:** Checking logic... ✅ Approved")
                        elif ver_status == "rejected":
                            st.write("4️⃣ **Verifier Agent:** Checking logic... ❌ Rejected")
                        else:
                            st.write("4️⃣ **Verifier Agent:** Checking logic... ⚠️ Uncertain (HITL)")
                        st.write("5️⃣ **Explainer Agent:** Formatting final explanation...")

                    answer = final_state["final_answer"]
                    rag_context = final_state.get(
                        "retrieved_context", "No context found."
                    )
                    parsed = final_state.get("parsed_data") or {}
                    needs_clarification = parsed.get("needs_clarification", False)
                    verification_status = final_state.get("verification_status", "approved")

                    context_placeholder.text_area(
                        "RAG Source:", rag_context, height=200
                    )

                    # Confidence indicator
                    if verification_status == "approved":
                        confidence_label = "✅ **Verified**"
                    elif verification_status == "rejected":
                        confidence_label = "❌ **Rejected** — solution was critiqued; explanation includes corrections."
                    else:
                        confidence_label = "⚠️ **Uncertain** — please verify (HITL)."

                    # HITL messages
                    hitl_parts = []
                    if needs_clarification:
                        hitl_parts.append("Parser flagged ambiguous or incomplete input; please verify the solution.")
                    if verification_status == "uncertain":
                        hitl_parts.append("Verifier was not confident; human verification recommended.")

                    full_content = answer
                    if confidence_label:
                        full_content = f"{confidence_label}\n\n{full_content}"
                    if hitl_parts:
                        full_content += "\n\n---\n**HITL:** " + " ".join(hitl_parts)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_content,
                            "meta": {
                                "confidence": verification_status,
                                "needs_clarification": needs_clarification,
                                "parsed_data": parsed,
                                "verifier_outcome": final_state.get("verification_confidence"),
                                "input_type": input_method.lower(),
                                "retrieved_context": rag_context[:500] if rag_context else None,
                            },
                        }
                    )
                    status.update(
                        label="✅ Solved!", state="complete", expanded=False
                    )

                except Exception as e:
                    st.error(f"Error: {e}")


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("✅", key=f"good_{i}"):
                    if i > 0:
                        question = st.session_state.messages[i - 1]["content"]
                        ans = msg["content"]
                        meta = msg.get("meta") or {}
                        fs = st.session_state.get("last_final_state")
                        if fs and i == len(st.session_state.messages) - 1:
                            parsed = fs.get("parsed_data") or {}
                            meta = {
                                "input_type": meta.get("input_type"),
                                "topic": parsed.get("topic"),
                                "verifier_outcome": fs.get("verification_confidence"),
                                "user_feedback": "correct",
                                "retrieved_context": fs.get("retrieved_context"),
                                "parsed_data": parsed,
                            }
                        else:
                            meta = {
                                "verifier_outcome": meta.get("verifier_outcome"),
                                "user_feedback": "correct",
                            }
                        save_memory(question, ans, meta=meta)
                    st.toast("Saved to Memory! 🧠")
            with col2:
                if st.button("❌", key=f"bad_{i}"):
                    # Store correction feedback for learning
                    if i > 0 and st.session_state.get("last_final_state"):
                        fs = st.session_state.last_final_state
                        question = st.session_state.messages[i - 1]["content"]
                        save_memory(
                            question,
                            msg["content"],
                            meta={
                                "input_type": st.session_state.get("last_input_type"),
                                "topic": (fs.get("parsed_data") or {}).get("topic"),
                                "verifier_outcome": fs.get("verification_confidence"),
                                "user_feedback": "incorrect",
                            },
                        )
                    st.toast("Feedback recorded.")