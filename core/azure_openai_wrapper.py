# core/azure_openai_wrapper.py
"""
Custom wrapper for Azure OpenAI to work with LlamaIndex.
Provides compatibility between Azure OpenAI API and LlamaIndex requirements.
"""

import warnings

# Suppress Pydantic warnings BEFORE importing pydantic-based modules
warnings.filterwarnings(
    "ignore",
    message=".*validate_default.*",
)

from typing import Any
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata
)
from llama_index.core.llms.callbacks import llm_completion_callback
from openai import AzureOpenAI
import httpx
from utils.logger import log


class AzureOpenAIEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding wrapper for LlamaIndex."""
    
    _client: Any = PrivateAttr()
    
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str = "2023-05-15",
        deployment_name: str = "text-embedding-3-large",
        dimensions: int = 3072,
        **kwargs
    ):
        # Call parent __init__ without these custom params
        super().__init__(**kwargs)
        
        # Store as private attributes
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._deployment_name = deployment_name
        self._dimensions = dimensions
        
        # Initialize Azure OpenAI client
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=httpx.Timeout(60.0, connect=5.0),
            max_retries=3
        )
        
        log.info(f"Initialized Azure OpenAI Embedding with deployment: {deployment_name}")
    
    @classmethod
    def class_name(cls) -> str:
        return "AzureOpenAIEmbedding"
    
    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a single query - blocking synchronous call."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                log.debug(f"Azure embedding API call (attempt {attempt + 1})")
                
                # Blocking call - waits for complete response
                response = self._client.embeddings.create(
                    input=[query],
                    model=self._deployment_name
                )
                
                # Validate response
                if not response.data or len(response.data) == 0:
                    raise ValueError("Empty response from Azure API")
                
                embedding = response.data[0].embedding
                log.debug(f"Azure embedding received: {len(embedding)} dims")
                return embedding
                
            except Exception as e:
                log.error(f"Error getting query embedding (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
    
    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        return self._get_query_embedding(text)
    
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts - blocking synchronous calls."""
        try:
            # Process in batches to avoid rate limits
            batch_size = 100
            all_embeddings = []
            total_texts = len(texts)
            
            log.info(f"Processing {total_texts} texts for Azure embeddings")
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_texts + batch_size - 1) // batch_size
                
                log.debug(f"Batch {batch_num}/{total_batches}")
                
                # Blocking call - waits for complete response
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._deployment_name
                )
                
                # Validate response
                if not response.data:
                    raise ValueError(f"Empty response for batch {batch_num}")
                
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)
            
            log.info(f"Completed {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            log.error(f"Error getting text embeddings: {e}")
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


class PoeLLM(CustomLLM):
    """
    Poe Service LLM integration for LlamaIndex using fastapi_poe library.
    Uses a dedicated thread pool with persistent event loop to properly
    handle async operations in Python 3.13.
    """
    
    model_name: str = "Claude-Sonnet-4"
    max_tokens: int = 4000
    temperature: float = 0.1
    
    _api_key: str = PrivateAttr()
    _executor: Any = PrivateAttr()
    _loop: Any = PrivateAttr()
    _loop_thread: Any = PrivateAttr()
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "Claude-Sonnet-4",
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
        self._api_key = api_key
        self._executor = None
        self._loop = None
        self._loop_thread = None
        
        # Initialize the dedicated event loop thread
        self._ensure_event_loop()
        
        log.info(f"Initialized Poe LLM with model: {model_name}")
    
    def _ensure_event_loop(self):
        """
        Create and start a dedicated event loop in a background thread.
        This loop persists for the lifetime of the PoeLLM instance,
        ensuring proper async cleanup and avoiding context switching issues.
        """
        if self._loop is not None:
            return
        
        import threading
        import asyncio
        
        # Event to signal when loop is ready
        loop_ready = threading.Event()
        
        def run_event_loop():
            """Background thread that runs the persistent event loop"""
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            
            # Signal that loop is ready
            loop_ready.set()
            
            # Run the loop forever (until program exit)
            try:
                loop.run_forever()
            finally:
                loop.close()
        
        # Start the background thread
        self._loop_thread = threading.Thread(
            target=run_event_loop,
            daemon=True,
            name="PoeLLM-EventLoop"
        )
        self._loop_thread.start()
        
        # Wait for loop to be ready
        loop_ready.wait(timeout=5)
        
        if self._loop is None:
            raise RuntimeError("Failed to initialize event loop thread")
        
        log.debug("✓ Dedicated event loop thread started")
    
    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=8000,
            num_output=self.max_tokens,
            model_name=self.model_name,
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "PoeLLM"
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        Generate completion using Poe API.
        Runs in dedicated event loop thread to avoid async context issues.
        """
        import time
        import asyncio
        import httpx
        try:
            import fastapi_poe as fp
        except ImportError:
            log.error(
                "fastapi_poe not installed. "
                "Install with: pip install fastapi-poe"
            )
            return CompletionResponse(
                text="Error: fastapi_poe library not installed"
            )
        
        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 2.0)
        
        async def _get_response_async():
            """
            Async function that runs in the dedicated event loop.
            This ensures all async operations happen in the same context.
            """
            messages = [
                fp.ProtocolMessage(role="user", content=prompt)
            ]
            
            full_text = ""
            
            # Create httpx client with proper timeout
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0)
            ) as session:
                async for partial in fp.get_bot_response(
                    messages=messages,
                    bot_name=self.model_name,
                    api_key=self._api_key,
                    temperature=kwargs.get("temperature", self.temperature),
                    session=session,
                ):
                    if hasattr(partial, 'text') and partial.text:
                        full_text += partial.text
            
            return full_text
        
        for attempt in range(max_retries):
            try:
                log.info(
                    f"Poe API call (attempt {attempt + 1}/{max_retries}) "
                    f"- blocking until response"
                )
                start_time = time.time()
                
                # Submit the coroutine to the dedicated event loop
                # and wait for the result
                future = asyncio.run_coroutine_threadsafe(
                    _get_response_async(),
                    self._loop
                )
                
                # Block until we get the result (with timeout)
                full_text = future.result(timeout=120)
                
                elapsed = time.time() - start_time
                
                # Validate response
                if not full_text:
                    log.warning("Empty response from Poe API")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return CompletionResponse(
                        text="Error: Empty response from Poe API"
                    )
                
                log.info(
                    f"Poe API response in {elapsed:.2f}s "
                    f"({len(full_text)} chars)"
                )
                return CompletionResponse(text=full_text)
                
            except Exception as e:
                error_msg = str(e)
                log.error(
                    f"Error on attempt {attempt + 1}/{max_retries}: "
                    f"{error_msg}"
                )
                
                if attempt < max_retries - 1:
                    backoff_time = retry_delay * (2 ** attempt)
                    log.info(f"Retrying in {backoff_time:.1f} seconds...")
                    time.sleep(backoff_time)
                    continue
                else:
                    return CompletionResponse(
                        text=f"Error after {max_retries} attempts: "
                        f"{error_msg}"
                    )
        
        return CompletionResponse(text="Error: Maximum retries exceeded")
    
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs
    ) -> CompletionResponseGen:
        """
        Streaming completion using Poe API.
        Runs in dedicated event loop thread to avoid async context issues.
        """
        import time
        import asyncio
        import httpx
        try:
            import fastapi_poe as fp
        except ImportError:
            log.error("fastapi_poe not installed")
            yield CompletionResponse(
                text="Error: fastapi_poe library not installed"
            )
            return
        
        async def _stream_response_async():
            """
            Async function that runs in the dedicated event loop.
            Collects all partials with proper session management.
            """
            messages = [
                fp.ProtocolMessage(role="user", content=prompt)
            ]
            
            partials = []
            
            # Create httpx client with proper timeout
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0)
            ) as session:
                async for partial in fp.get_bot_response(
                    messages=messages,
                    bot_name=self.model_name,
                    api_key=self._api_key,
                    temperature=kwargs.get("temperature", self.temperature),
                    session=session,
                ):
                    if hasattr(partial, 'text') and partial.text:
                        partials.append(partial)
            
            return partials
        
        try:
            log.info("Poe API streaming call initiated")
            start_time = time.time()
            
            # Submit the coroutine to the dedicated event loop
            future = asyncio.run_coroutine_threadsafe(
                _stream_response_async(),
                self._loop
            )
            
            # Block until we get all partials
            partials = future.result(timeout=120)
            
            # Now yield each partial
            for partial in partials:
                if hasattr(partial, 'text') and partial.text:
                    yield CompletionResponse(
                        text=partial.text, delta=partial.text
                    )
            
            elapsed = time.time() - start_time
            log.info(f"Poe API streaming completed in {elapsed:.2f}s")
            
        except Exception as e:
            log.error(f"Error in streaming: {e}")
            yield CompletionResponse(text=f"Error: {str(e)}")



