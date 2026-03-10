---
title: Math Mentor
emoji: "🧮"
colorFrom: red
colorTo: red
sdk: streamlit
sdk_version: "1.39.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

## Multimodal AI Math Mentor (Groq Edition)

This project replicates the architecture and behavior of the original
`Multimodal-Math-Mentor` repository, but swaps the LLM layer to Groq /
open‑source models.

### Key Features

- **Streamlit frontend** (`app.py`)
- **LangGraph multi‑agent pipeline** (`src/agents.py`)
  - Parser → Solver → Verifier → Explainer
- **RAG with ChromaDB** (`src/rag.py`)
- **Multimodal input shell**
  - Text (fully wired)
  - Image OCR (stub; ready to connect to a vision model)
  - Audio STT via **Groq Whisper**
- **Self‑learning memory** (`memory.json`)
- **Math solution verification** inside the LangGraph flow

### Setup

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables (at minimum):

- `GROQ_API_KEY` – your Groq API key
- (optional) `EMBEDDING_MODEL_NAME` – override the default HF embedding model

3. Build the vector store (RAG index):

```bash
python -m src.rag
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

Place your math reference `.txt` files under `knowledge_base/` before
running the RAG build step.

### Where to set `GROQ_API_KEY`

- **Locally**: set an environment variable before running:

  - PowerShell:

    ```powershell
    $env:GROQ_API_KEY = "your_groq_key_here"
    streamlit run app.py
    ```

  - Or create a `.env` file with:

    ```text
    GROQ_API_KEY=your_groq_key_here
    ```

- **On Hugging Face Spaces**:
  - Go to your Space `[zaidkhan/math_mentor](https://huggingface.co/spaces/zaidkhan/math_mentor)`.
  - Open **Settings → Variables and secrets**.
  - Add a new **Secret** named `GROQ_API_KEY` with your key.
  - Save and restart the Space.

