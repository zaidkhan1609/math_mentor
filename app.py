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
    """Load memory entries (each may have question, answer, input_type, topic, verifier_outcome, user_feedback)."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_memory(question, answer, meta=None):
    """Save a verified/corrected entry. meta can include input_type, topic, verifier_outcome, user_feedback."""
    history = load_memory()
    entry = {"question": question, "answer": answer}
    if meta:
        entry.update({k: v for k, v in meta.items() if v is not None})
    history.append(entry)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def find_similar_solution(user_input):
    """Memory reuse: return answer if we have a similar solved problem."""
    history = load_memory()
    user_lower = user_input.strip().lower()
    for entry in history:
        q = (entry.get("question") or "").strip().lower()
        if user_lower in q or q in user_lower:
            return entry.get("answer")
    return None


with st.sidebar:
    st.header("⚙️ Debug & Options")
    show_trace = st.checkbox(
        "Show Agent Trace",
        value=True,
        help="See what the agents are thinking",
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
        cached_answer = find_similar_solution(user_input)

        if cached_answer:
            st.success("🧠 I remembered a similar problem!")
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"**[From Memory]**\n\n{cached_answer}",
                    "meta": {"confidence": "memory", "hitl": None},
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
                                "topic": parsed.get("topic"),
                                "verifier_outcome": fs.get("verification_confidence"),
                                "user_feedback": "correct",
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
                                "topic": (fs.get("parsed_data") or {}).get("topic"),
                                "verifier_outcome": fs.get("verification_confidence"),
                                "user_feedback": "incorrect",
                            },
                        )
                    st.toast("Feedback recorded.")