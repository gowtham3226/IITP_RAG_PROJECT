"""Create the selected provider's chat model and embedding model."""

import os
from typing import Any

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config


def require_api_key(provider: str) -> str:
    """Return the selected provider's API key or raise a helpful error.

    Args:
        provider: Either ``openai`` or ``gemini``.

    Returns:
        The API key read from the environment.
    """
    variable_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    api_key = os.getenv(variable_name, "").strip()
    if not api_key:
        raise ValueError(
            f"{variable_name} is missing. Copy .env.example to .env and add your key."
        )
    return api_key


def get_default_model(provider: str) -> str:
    """Get the configured default chat model for a provider.

    Args:
        provider: Either ``openai`` or ``gemini``.

    Returns:
        The matching chat model name from ``config.py``.
    """
    return config.OPENAI_MODEL if provider == "openai" else config.GEMINI_MODEL


def get_embedding_model_name(provider: str) -> str:
    """Get the configured embedding model name for a provider.

    Args:
        provider: Either ``openai`` or ``gemini``.

    Returns:
        The matching embedding model name from ``config.py``.
    """
    if provider == "openai":
        return config.OPENAI_EMBEDDING_MODEL
    return config.GEMINI_EMBEDDING_MODEL


def create_chat_model(
    provider: str,
    model_name: str,
    temperature: float,
) -> Any:
    """Initialize an OpenAI or Gemini chat model.

    The rest of the project can call the returned model in the same way, so
    provider-specific setup stays in this one beginner-friendly file.

    Args:
        provider: Either ``openai`` or ``gemini``.
        model_name: Provider model used to rewrite and answer questions.
        temperature: Creativity setting; lower values are more predictable.

    Returns:
        A LangChain chat model with a common ``invoke`` method.
    """
    api_key = require_api_key(provider)
    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
        )
    raise ValueError("Provider must be 'openai' or 'gemini'.")


def create_embeddings(provider: str) -> Any:
    """Initialize the embedding model used to turn text into vectors.

    An embedding is a list of numbers representing meaning. Similar passages
    receive nearby vectors, which lets FAISS find relevant chunks quickly.

    Args:
        provider: Either ``openai`` or ``gemini``.

    Returns:
        A LangChain embedding client for the selected provider.
    """
    api_key = require_api_key(provider)
    if provider == "openai":
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=api_key,
        )
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=api_key,
        )
    raise ValueError("Provider must be 'openai' or 'gemini'.")
