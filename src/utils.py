import base64
import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment or in "
            "Hugging Face Space Settings → Variables and secrets."
        )
    return OpenAI(api_key=api_key)


def get_embedding_model():
    """Return OpenAI embeddings for RAG (used by Chroma)."""
    from langchain_openai import OpenAIEmbeddings

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )


def get_llm(model_name: str = "gpt-4o") -> BaseChatModel:
    """Return a LangChain ChatOpenAI model (GPT-4o)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment or in "
            "Hugging Face Space Settings → Variables and secrets."
        )
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
    )


def encode_image(image_file) -> str:
    """Convert uploaded image file to base64 string."""
    return base64.b64encode(image_file.read()).decode("utf-8")


def perform_ocr(image_file) -> str:
    """Extract text from image using GPT-4o Vision."""
    base64_image = encode_image(image_file)
    llm_vision = get_llm("gpt-4o")
    content = [
        {
            "type": "text",
            "text": "Transcribe this math problem exactly into LaTeX/text. Do NOT solve it.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        },
    ]
    response = llm_vision.invoke([HumanMessage(content=content)])
    return response.content


def transcribe_audio(audio_file) -> str:
    """Transcribe audio using OpenAI Whisper."""
    client = _get_openai_client()
    # OpenAI API expects a file-like object with name; Streamlit uploads have it
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcription.text


def simple_chat_completion(
    prompt: str, model_name: Optional[str] = None
) -> str:
    """Utility for quick one-off completions, if ever needed."""
    llm = get_llm(model_name=model_name or "gpt-4o")
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content if hasattr(result, "content") else str(result)
