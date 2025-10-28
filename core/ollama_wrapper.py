# core/ollama_wrapper.py
"""
Ollama wrapper for embeddings and chat functionality.
Provides compatibility between Ollama API and LlamaIndex requirements.
"""

from typing import Any, Optional, Sequence, List
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
import requests
import json
import time
from utils.logger import log


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding wrapper for LlamaIndex."""

    _base_url: str = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        **kwargs
    ):
        # Call parent __init__ without these custom params
        super().__init__(**kwargs)

        # Store as private attributes
        self._base_url = base_url.rstrip('/')
        self._model_name = model_name

        log.info(f"Initialized Ollama Embedding with model: {model_name} at {base_url}")

    @classmethod
    def class_name(cls) -> str:
        return "OllamaEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a single query - blocking synchronous call."""
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                url = f"{self._base_url}/api/embeddings"
                payload = {
                    "model": self._model_name,
                    "prompt": query
                }

                log.debug(f"Embedding API call (attempt {attempt + 1}/{max_retries})")
                
                # Blocking call with extended timeout for slow models
                response = requests.post(
                    url, 
                    json=payload, 
                    timeout=(10, 120)  # (connect, read) timeout
                )
                response.raise_for_status()

                data = response.json()
                
                # Validate response structure
                if "embedding" not in data:
                    raise ValueError("Invalid response: missing 'embedding' field")
                
                embedding = data["embedding"]
                
                # Validate embedding is a list of numbers
                if not isinstance(embedding, list) or len(embedding) == 0:
                    raise ValueError("Invalid embedding: expected non-empty list")
                
                log.debug(f"Embedding received: {len(embedding)} dimensions")
                return embedding

            except requests.exceptions.Timeout as e:
                log.error(f"Timeout on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
            except requests.exceptions.ConnectionError as e:
                log.error(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                raise
            except Exception as e:
                log.error(f"Error getting Ollama embedding: {e}")
                raise
        
        raise RuntimeError("Failed to get embedding after all retries")

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts - blocking synchronous calls."""
        try:
            # Process in batches to avoid overwhelming Ollama
            batch_size = 5  # Reduced for better reliability
            all_embeddings = []
            total_texts = len(texts)

            log.info(f"Processing {total_texts} texts for embeddings in batches of {batch_size}")

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_texts + batch_size - 1) // batch_size
                
                log.debug(f"Processing batch {batch_num}/{total_batches}")

                for text in batch:
                    # Each call blocks until response received
                    embedding = self._get_text_embedding(text)
                    all_embeddings.append(embedding)
                    
                # Small delay between batches to avoid overwhelming server
                if i + batch_size < total_texts:
                    time.sleep(0.5)

            log.info(f"Completed {total_texts} embeddings successfully")
            return all_embeddings

        except Exception as e:
            log.error(f"Error getting Ollama text embeddings: {e}")
            raise

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async version - falls back to sync for now."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async version - falls back to sync for now."""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Async version - falls back to sync for now."""
        return self._get_text_embeddings(texts)


class OllamaLLM(CustomLLM):
    """
    Ollama LLM integration for LlamaIndex.
    Uses Ollama API to generate text completions.
    """

    model_name: str = "llama2"
    max_tokens: int = 4000
    temperature: float = 0.1

    _base_url: str = PrivateAttr()

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama2",
        max_tokens: int = 4000,
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        self._base_url = base_url.rstrip('/')

        log.info(f"Initialized Ollama LLM with model: {model_name} at {base_url}")

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=4096,  # Conservative estimate, adjust based on model
            num_output=self.max_tokens,
            model_name=self.model_name,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OllamaLLM"

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Generate completion using Ollama API.
        Implements retry logic with exponential backoff for connection issues.
        """
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 2.0)

        for attempt in range(max_retries):
            try:
                url = f"{self._base_url}/api/generate"
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self.temperature),
                        "num_predict": kwargs.get("max_tokens", self.max_tokens)
                    }
                }

                log.info(f"LLM API call (attempt {attempt + 1}/{max_retries}) - blocking until response")
                start_time = time.time()

                # Blocking synchronous call with extended timeout
                response = requests.post(
                    url,
                    json=payload,
                    timeout=(15, 600)  # (connect timeout, read timeout) - increased for large contexts
                )
                
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    text = data.get("response", "")
                    
                    # Validate response
                    if not text:
                        log.warning("Empty response from LLM API")
                        if attempt < max_retries - 1:
                            log.info("Retrying due to empty response...")
                            time.sleep(retry_delay)
                            continue
                        return CompletionResponse(text="Error: Empty response from API")
                    
                    log.info(f"LLM response received in {elapsed:.2f}s ({len(text)} chars)")
                    return CompletionResponse(text=text)
                else:
                    log.error(f"Ollama API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        log.info(f"Retrying after error response...")
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    return CompletionResponse(text=f"Error: Ollama API returned status {response.status_code}")

            except requests.exceptions.ConnectionError as e:
                log.error(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    backoff_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    log.info(f"Retrying in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                    continue
                else:
                    return CompletionResponse(text=f"Connection failed after {max_retries} attempts: {str(e)}")

            except requests.exceptions.Timeout as e:
                log.error(f"Timeout error on attempt {attempt + 1}/{max_retries}: {e}")
                log.error("This may indicate the model is processing a large context or server is overloaded")
                if attempt < max_retries - 1:
                    backoff_time = retry_delay * (2 ** attempt)
                    log.info(f"Retrying in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                    continue
                else:
                    error_msg = f"Request timed out after {max_retries} attempts. Try reducing context size or check server status."
                    log.error(error_msg)
                    return CompletionResponse(text=f"Error: {error_msg}")

            except requests.exceptions.RequestException as e:
                log.error(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    backoff_time = retry_delay * (2 ** attempt)
                    log.info(f"Retrying in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                    continue
                else:
                    return CompletionResponse(text=f"Request failed after {max_retries} attempts: {str(e)}")

            except Exception as e:
                log.error(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    backoff_time = retry_delay * (2 ** attempt)
                    log.info(f"Retrying in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                    continue
                else:
                    return CompletionResponse(text=f"Error: {str(e)}")

        # Should never reach here, but just in case
        return CompletionResponse(text="Error: Maximum retries exceeded")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """Streaming not implemented, falls back to complete."""
        response = self.complete(prompt, **kwargs)
        yield response