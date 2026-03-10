import base64
import os
from typing import Optional

from groq import Groq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings


def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please set it in your environment."
        )
    return Groq(api_key=api_key)


def get_llm(model_name: str = "llama-3.1-70b-versatile") -> BaseChatModel:
    """Return a LangChain-compatible chat model backed by Groq."""
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please set it in your environment."
        )

    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0,
    )


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return an open-source embedding model for ChromaDB."""
    # Small, fast general-purpose embedding model.
    model_name = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    return HuggingFaceEmbeddings(model_name=model_name)


def encode_image(image_file) -> str:
    """Convert uploaded image file to base64 string."""
    return base64.b64encode(image_file.read()).decode("utf-8")


def perform_ocr(image_file) -> str:
    """Extract text from an image.

    NOTE: Groq does not currently provide a vision model API.
    For now, this function simply returns a placeholder message
    with a suggestion to type the text manually.
    """
    _ = encode_image(image_file)
    return (
        "Image received. Vision OCR is not configured with Groq in this template.\n\n"
        "Please type the math problem text manually, or extend `perform_ocr` to call "
        "a vision-capable open-source or hosted model."
    )


def transcribe_audio(audio_file) -> str:
    """Transcribe audio using Groq Whisper model."""
    client = _get_groq_client()
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        response_format="json",
    )
    # The Groq SDK returns an object with 'text' for Whisper.
    return transcription.text  # type: ignore[attr-defined]


def simple_chat_completion(prompt: str, model_name: Optional[str] = None) -> str:
    """Utility for quick one-off completions, if ever needed."""
    llm = get_llm(model_name=model_name or "llama-3.1-70b-versatile")
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content if hasattr(result, "content") else str(result)

