"""Embedding providers supporting Ollama nomic-embed-text and reliable local fallback."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


@dataclass
class OllamaDiagnostics:
    """Detailed diagnostic information for Ollama connectivity and model availability."""
    is_installed: bool
    is_server_running: bool
    is_model_available: bool
    error_code: str  # OK, NOT_INSTALLED, SERVER_STOPPED, CONNECTION_REFUSED, TIMEOUT, MODEL_MISSING, HTTP_ERROR
    message: str
    action_hint: str
    resolved_url: str
    available_models: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.is_server_running and self.is_model_available


def probe_ollama_installation() -> Tuple[bool, Optional[str]]:
    """Check if ollama executable is installed in PATH or standard Windows paths."""
    # 1. Check PATH
    cmd = shutil.which("ollama")
    if cmd:
        return True, cmd

    # 2. Check standard Windows installation directories
    local_app_data = os.getenv("LOCALAPPDATA", "")
    program_files = os.getenv("ProgramFiles", "C:\\Program Files")
    candidates = [
        Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe" if local_app_data else None,
        Path(program_files) / "Ollama" / "ollama.exe",
    ]
    for c in candidates:
        if c and c.is_file():
            return True, str(c)

    return False, None


def check_ollama_endpoint(
    base_url: str,
    target_model: Optional[str] = None,
    timeout_seconds: float = 3.0,
) -> OllamaDiagnostics:
    """
    Perform a safe, thorough diagnostic check on Ollama connectivity,
    supporting dual localhost/127.0.0.1 probing on Windows.
    """
    is_installed, exe_path = probe_ollama_installation()

    clean_url = base_url.rstrip("/")
    probe_urls = [clean_url]
    # If base_url uses localhost, also prepare 127.0.0.1 fallback for Windows IPv6 mismatch
    if "localhost" in clean_url:
        probe_urls.append(clean_url.replace("localhost", "127.0.0.1"))
    elif "127.0.0.1" in clean_url:
        probe_urls.append(clean_url.replace("127.0.0.1", "localhost"))

    last_error = ""
    last_error_code = "CONNECTION_REFUSED"
    connected_url = None
    resp = None

    for url in probe_urls:
        try:
            r = requests.get(f"{url}/api/tags", timeout=timeout_seconds)
            if r.status_code == 200:
                connected_url = url
                resp = r
                break
            else:
                last_error_code = "HTTP_ERROR"
                last_error = f"HTTP {r.status_code}: {r.text[:100]}"
        except requests.exceptions.Timeout:
            last_error_code = "TIMEOUT"
            last_error = f"Connection to {url} timed out after {timeout_seconds}s."
        except requests.exceptions.ConnectionError as exc:
            last_error_code = "CONNECTION_REFUSED"
            last_error = f"Connection refused at {url}."
        except Exception as exc:
            last_error_code = "UNKNOWN_ERROR"
            last_error = str(exc)

    # If connection could not be established
    if not connected_url or not resp:
        if not is_installed:
            return OllamaDiagnostics(
                is_installed=False,
                is_server_running=False,
                is_model_available=False,
                error_code="NOT_INSTALLED",
                message="Ollama is not installed or not in PATH.",
                action_hint="Install Ollama using 'winget install Ollama.Ollama' or download from https://ollama.com",
                resolved_url=clean_url,
            )
        return OllamaDiagnostics(
            is_installed=True,
            is_server_running=False,
            is_model_available=False,
            error_code=last_error_code,
            message=f"Ollama server is not responding ({last_error}).",
            action_hint=f"Start the Ollama application or run 'ollama serve' in PowerShell.",
            resolved_url=clean_url,
        )

    # Server is responding! Parse models
    try:
        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "") for m in models]
    except Exception as exc:
        return OllamaDiagnostics(
            is_installed=is_installed,
            is_server_running=True,
            is_model_available=False,
            error_code="PARSE_ERROR",
            message=f"Failed to parse Ollama model list: {str(exc)}",
            action_hint="Check Ollama logs for server errors.",
            resolved_url=connected_url,
        )

    if not target_model:
        return OllamaDiagnostics(
            is_installed=is_installed,
            is_server_running=True,
            is_model_available=True,
            error_code="OK",
            message=f"Ollama is online at {connected_url}.",
            action_hint="",
            resolved_url=connected_url,
            available_models=model_names,
        )

    # Check if target_model is downloaded
    matched = any(
        name == target_model or name.startswith(f"{target_model}:")
        for name in model_names
    )

    if not matched:
        return OllamaDiagnostics(
            is_installed=is_installed,
            is_server_running=True,
            is_model_available=False,
            error_code="MODEL_MISSING",
            message=f"Model '{target_model}' is not installed in Ollama.",
            action_hint=f"Run: ollama pull {target_model}",
            resolved_url=connected_url,
            available_models=model_names,
        )

    return OllamaDiagnostics(
        is_installed=is_installed,
        is_server_running=True,
        is_model_available=True,
        error_code="OK",
        message=f"Ollama is online and model '{target_model}' is ready.",
        action_hint="",
        resolved_url=connected_url,
        available_models=model_names,
    )


class BaseEmbedder(ABC):
    """Abstract base class for text embedding generation."""

    @property
    @abstractmethod
    def is_fallback(self) -> bool:
        """Indicates whether this embedder is a pseudo-fallback or real semantic model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single query."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of documents."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the vector dimension."""
        pass

    @abstractmethod
    def check_health(self) -> Tuple[bool, str, OllamaDiagnostics]:
        """Check if embedding service and model are ready."""
        pass


class OllamaEmbedder(BaseEmbedder):
    """Generates dense semantic embeddings using a local Ollama instance (e.g. nomic-embed-text)."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_seconds: int = 30,
    ):
        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self._dimension: Optional[int] = None
        self._last_diagnostics: Optional[OllamaDiagnostics] = None

    @property
    def is_fallback(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return self._model_name

    def check_health(self) -> Tuple[bool, str, OllamaDiagnostics]:
        diag = check_ollama_endpoint(self.base_url, target_model=self._model_name, timeout_seconds=3.0)
        self._last_diagnostics = diag
        if diag.resolved_url and diag.resolved_url != self.base_url:
            self.base_url = diag.resolved_url
        return diag.is_healthy, diag.message, diag

    def _call_ollama_embed(self, input_data: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed (recommended) or fallback to /api/embeddings."""
        # Try modern /api/embed endpoint
        embed_url = f"{self.base_url}/api/embed"
        payload = {"model": self._model_name, "input": input_data}
        try:
            resp = requests.post(embed_url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                if "embeddings" in data:
                    return data["embeddings"]
        except Exception as exc:
            logger.debug("/api/embed endpoint failed (%s), attempting /api/embeddings", exc)

        # Fallback to single-item /api/embeddings loop
        single_url = f"{self.base_url}/api/embeddings"
        results = []
        for text in input_data:
            try:
                resp = requests.post(
                    single_url,
                    json={"model": self._model_name, "prompt": text},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                results.append(resp.json()["embedding"])
            except requests.exceptions.ConnectionError as exc:
                raise ConnectionError(
                    f"Ollama connection error at {self.base_url}: {exc}. Is Ollama running?"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Ollama embedding request error for model '{self._model_name}': {str(exc)}"
                ) from exc
        return results

    def embed_query(self, text: str) -> List[float]:
        res = self._call_ollama_embed([text])
        if not res or not res[0]:
            raise RuntimeError(f"Ollama returned empty embedding for text: {text[:50]}...")
        if self._dimension is None:
            self._dimension = len(res[0])
        return res[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._call_ollama_embed(texts)
        if embeddings and self._dimension is None:
            self._dimension = len(embeddings[0])
        return embeddings

    def get_dimension(self) -> int:
        if self._dimension is None:
            # Query dimension with a tiny probe
            probe = self.embed_query("dimension probe")
            self._dimension = len(probe)
        return self._dimension


class DeterministicFallbackEmbedder(BaseEmbedder):
    """
    Zero-dependency deterministic hashing embedder.
    Provides offline/testing resilience when Ollama is not yet installed or running.
    Produces stable, normalized, 384-dimensional vector representations.
    LIMITATION: Lexical hashing provides keyword-proximity indexing, NOT deep semantic understanding.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._model_name = "deterministic-fallback-hash-384"

    @property
    def is_fallback(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._model_name

    def check_health(self) -> Tuple[bool, str, OllamaDiagnostics]:
        diag = OllamaDiagnostics(
            is_installed=True,
            is_server_running=False,
            is_model_available=False,
            error_code="FALLBACK_ACTIVE",
            message="Deterministic Fallback Embedder is active (pseudo-embeddings).",
            action_hint="Start Ollama and pull nomic-embed-text for genuine semantic embeddings.",
            resolved_url="",
        )
        return True, "Deterministic Fallback Embedder is active.", diag

    def get_dimension(self) -> int:
        return self._dimension

    def _hash_embed(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self._dimension
        if not tokens:
            return vec

        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_query(self, text: str) -> List[float]:
        return self._hash_embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embed(t) for t in texts]


def get_embedder(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    allow_fallback: bool = True,
) -> BaseEmbedder:
    """
    Factory function returning an Ollama embedder if available,
    or a deterministic fallback embedder when offline if allowed.
    """
    target_model = model_name or DEFAULT_EMBED_MODEL
    target_url = base_url or DEFAULT_OLLAMA_BASE_URL

    ollama_embedder = OllamaEmbedder(model_name=target_model, base_url=target_url)
    healthy, reason, diag = ollama_embedder.check_health()
    if healthy:
        logger.info("Using Ollama embedder (%s) at %s", target_model, ollama_embedder.base_url)
        return ollama_embedder

    if allow_fallback:
        logger.warning(
            "Ollama embedding unavailable (%s: %s). Falling back to DeterministicFallbackEmbedder.",
            diag.error_code,
            diag.message,
        )
        return DeterministicFallbackEmbedder(dimension=384)

    raise ConnectionError(
        f"Ollama embedding unavailable ({diag.error_code}): {diag.message} -> {diag.action_hint}"
    )
