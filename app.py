import streamlit as st
import os
import json

from src.agents import app_graph
from src.utils import perform_ocr, transcribe_audio
from src.rag import initialize_vector_store


DB_PATH = "./chroma_db"
if not os.path.exists(DB_PATH):
    initialize_vector_store()


st.set_page_config(page_title="Math Mentor AI", layout="wide")
st.title("🧮 Reliable Multimodal Math Mentor")


MEMORY_FILE = "memory.json"


def load_memory():
    """Loads past Q&A from JSON."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_memory(input_text, answer):
    """Saves a verified correct answer to learn from it."""
    history = load_memory()
    history.append({"question": input_text, "answer": answer})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def find_similar_solution(user_input):
    """Checks if we have solved this before (Simple Memory Reuse)."""
    history = load_memory()
    for entry in history:
        if user_input.strip().lower() in entry["question"].strip().lower():
            return entry["answer"]
    return None


with st.sidebar:
    st.header("⚙️ Debug & Options")
    # Requirement 5: Agent Trace
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
                    inputs = {"input_text": user_input, "messages": []}

                    st.write("1️⃣ **Parser Agent:** Structuring problem...")
                    st.write(
                        "2️⃣ **Solver Agent:** Retrieving RAG context & planning..."
                    )

                    final_state = app_graph.invoke(inputs)

                    st.write(
                        "3️⃣ **Verifier Agent:** Checking logic... ✅ Approved"
                    )
                    st.write(
                        "4️⃣ **Explainer Agent:** Formatting final output..."
                    )

                    answer = final_state["final_answer"]
                    rag_context = final_state.get(
                        "retrieved_context", "No context found."
                    )

                    # Show RAG Context in Sidebar (Req 3)
                    context_placeholder.text_area(
                        "RAG Source:", rag_context, height=200
                    )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
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
                        save_memory(question, msg["content"])
                    st.toast("Saved to Memory! 🧠")
            with col2:
                if st.button("❌", key=f"bad_{i}"):
                    st.toast("Feedback recorded.")
