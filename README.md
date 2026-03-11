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

## Multimodal AI Math Mentor

JEE-style math mentor with **multimodal input** (text, image OCR, audio), **LangGraph** agents, **RAG** (ChromaDB), and **self-learning memory**. Uses **OpenAI GPT-4o** for solving and verification, **GPT-4o Vision** for image OCR, and **Whisper** for audio.

### Key Features

- **Streamlit frontend** (`app.py`)
- **LangGraph multi-agent pipeline** (`src/agents.py`)
  - Parser → Solver → Verifier → Explainer
- **RAG with ChromaDB** (`src/rag.py`) — optional; app runs without a pre-built DB (e.g. on Spaces)
- **Multimodal input**: Text, Image (OCR via GPT-4o Vision), Audio (Whisper)
- **Self-learning memory** (`memory.json`)
- **Math solution verification** in the agent flow

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set **OpenAI API key**:

- **Locally**: `$env:OPENAI_API_KEY = "sk-..."` (PowerShell) or add `OPENAI_API_KEY=sk-...` to a `.env` file.
- **Hugging Face Spaces**: **Settings → Variables and secrets** → add a **Secret** named `OPENAI_API_KEY` with your key, then restart the Space.

3. (Optional) Build the vector store for RAG: ensure `OPENAI_API_KEY` is set, then run:

```bash
python -m src.rag
```

The repo includes JEE-style reference files in `knowledge_base/` (algebra, probability, calculus, linear algebra). Add more `.txt` files there if needed.

4. Run the app:

```bash
streamlit run app.py
```

If no vector DB exists (e.g. on Spaces with no `knowledge_base`), the app still runs and the solver uses no RAG context.
