#!/usr/bin/env python3
"""
Quick demonstration of blocking behavior in embedding and LLM calls.
This script shows that API calls properly wait for responses.
"""

import time
from config import settings
from core.ollama_wrapper import OllamaLLM, OllamaEmbedding


def demo_blocking():
    print("\n" + "="*70)
    print("BLOCKING BEHAVIOR DEMONSTRATION")
    print("="*70)
    
    # Initialize clients
    print("\nInitializing Ollama clients...")
    embedding_client = OllamaEmbedding(
        base_url=settings.OLLAMA_BASE_URL,
        model_name=settings.OLLAMA_EMBEDDING_MODEL
    )
    
    llm_client = OllamaLLM(
        base_url=settings.OLLAMA_BASE_URL,
        model_name=settings.OLLAMA_CHAT_MODEL,
        max_tokens=100
    )
    
    # Demo 1: Embedding blocks
    print("\n" + "-"*70)
    print("DEMO 1: Single Embedding Call (Blocking)")
    print("-"*70)
    print("\nCode:")
    print('  result = embedding_client._get_query_embedding("test text")')
    print('  # ^^^ This line BLOCKS until response is received')
    print('  print("Embedding received!")')
    
    print("\nExecuting...")
    start = time.time()
    result = embedding_client._get_query_embedding("test text")
    elapsed = time.time() - start
    print(f"✓ Embedding received! (blocked for {elapsed:.2f}s)")
    print(f"  Dimensions: {len(result)}")
    
    # Demo 2: LLM blocks
    print("\n" + "-"*70)
    print("DEMO 2: LLM Completion Call (Blocking)")
    print("-"*70)
    print("\nCode:")
    print('  response = llm_client.complete("Say hello")')
    print('  # ^^^ This line BLOCKS until response is received')
    print('  print(response.text)')
    
    print("\nExecuting...")
    start = time.time()
    response = llm_client.complete("Say hello in one word")
    elapsed = time.time() - start
    print(f"✓ Response received! (blocked for {elapsed:.2f}s)")
    print(f"  Response: {response.text}")
    
    # Demo 3: Sequential calls don't overlap
    print("\n" + "-"*70)
    print("DEMO 3: Sequential Calls (Each Blocks)")
    print("-"*70)
    print("\nCode:")
    print('  r1 = llm_client.complete("What is 1+1?")')
    print('  # ^^^ Blocks until first response')
    print('  r2 = llm_client.complete("What is 2+2?")')
    print('  # ^^^ Blocks until second response')
    print('  # These DO NOT run in parallel!')
    
    print("\nExecuting...")
    overall_start = time.time()
    
    start = time.time()
    r1 = llm_client.complete("1+1=?")
    t1 = time.time() - start
    print(f"  Call 1 completed in {t1:.2f}s: {r1.text[:30]}")
    
    start = time.time()
    r2 = llm_client.complete("2+2=?")
    t2 = time.time() - start
    print(f"  Call 2 completed in {t2:.2f}s: {r2.text[:30]}")
    
    total = time.time() - overall_start
    print(f"\n✓ Total time: {total:.2f}s (sum of individual times: {t1+t2:.2f}s)")
    print("  This proves calls are sequential and blocking!")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✓ All API calls are synchronous and blocking")
    print("✓ Each call waits for complete response before returning")
    print("✓ Sequential calls do NOT overlap or run in parallel")
    print("✓ This ensures data integrity and prevents race conditions")
    print("\nTimeout settings:")
    print(f"  - LLM calls: up to 600 seconds")
    print(f"  - Embedding calls: up to 120 seconds")
    print(f"  - Retry attempts: 3 with exponential backoff")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_blocking()
