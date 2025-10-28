# core/search_engine.py
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import Dict, List, Optional
from datetime import datetime

from config import settings
from utils.logger import log


class SearchEngine:
    """Advanced search engine with multiple query modes"""
    
    def __init__(self, context_manager, memory_manager):
        self.context_manager = context_manager
        self.memory_manager = memory_manager
        self.query_engine = context_manager.query_engine
        
        log.info("SearchEngine initialized")
    
    def search(
        self, 
        query: str, 
        mode: str = "enhanced",
        save_to_memory: bool = True
    ) -> Dict:
        """
        Execute search with different modes
        
        Modes:
        - simple: Basic vector search
        - enhanced: Search with conversation context
        - summarize: Get summary of results
        - analyze_all: Analyze ALL documents comprehensively
        """
        try:
            log.info(f"Executing search: '{query}' (mode: {mode})")
            
            # Add query to memory
            if save_to_memory:
                self.memory_manager.add_message("user", query)
            
            # Build context based on mode
            if mode == "simple":
                response = self._simple_search(query)
            elif mode == "enhanced":
                response = self._enhanced_search(query)
            elif mode == "summarize":
                response = self._summarize_search(query)
            elif mode == "analyze_all":
                response = self._analyze_all_documents(query)
            else:
                raise ValueError(f"Unknown search mode: {mode}")
            
            # Add response to memory
            if save_to_memory:
                self.memory_manager.add_message(
                    "assistant", 
                    response["answer"],
                    metadata={"sources": response.get("sources", [])}
                )
                self.memory_manager.save()
            
            log.info(f"Search completed: {response['answer'][:100]}...")
            return response
            
        except Exception as e:
            log.error(f"Error in search: {e}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _simple_search(self, query: str) -> Dict:
        """Simple vector similarity search"""
        
        # Get contexts
        contexts = self.context_manager.retrieve_context(query)
        
        # Query the engine
        response = self.query_engine.query(query)
        
        return {
            "query": query,
            "answer": str(response),
            "mode": "simple",
            "sources": self.context_manager.get_context_sources(query),
            "context_summary": self.context_manager.summarize_context(contexts),
            "timestamp": datetime.now().isoformat()
        }
    
    def _enhanced_search(self, query: str) -> Dict:
        """Search with conversation history context and graph"""
        from llama_index.core import Settings
        
        # Build enhanced context with more documents for diversity
        # This now includes graph context automatically
        context = self.context_manager.build_enhanced_context(
            query, 
            include_memory=True,
            top_k=30
        )
        
        # Add graph insights if available
        graph_info = ""
        if self.context_manager.graph_manager is not None:
            try:
                graph_result = self.context_manager.graph_manager.query_graph(query)
                if graph_result.get('nodes'):
                    graph_info = f"\n\nKnowledge Graph Insights:\n"
                    graph_info += f"Related concepts: {', '.join([n['label'] for n in graph_result['nodes'][:10]])}\n"
                    if graph_result.get('relationships'):
                        graph_info += f"Key relationships: {len(graph_result['relationships'])} connections found\n"
            except Exception as e:
                log.warning(f"Could not get graph insights: {e}")
        
        # Format prompt with all context
        prompt = self.context_manager.format_prompt(query, context)
        if graph_info:
            prompt = prompt + graph_info
        
        # Use LLM directly for enhanced context (already has retrieved docs)
        if Settings.llm is not None and hasattr(Settings.llm, 'complete'):
            llm_response = Settings.llm.complete(prompt)
            answer = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)
        else:
            # Fallback to query engine
            response = self.query_engine.query(query)
            answer = str(response)
        
        return {
            "query": query,
            "answer": answer,
            "mode": "enhanced",
            "sources": context.get("sources", []),
            "conversation_context_used": True,
            "retrieved_chunks": context["retrieved_chunks"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _summarize_search(self, query: str) -> Dict:
        """Get summarized results"""
        
        contexts = self.context_manager.retrieve_context(query, top_k=10)
        
        # Create summary prompt
        summary_query = f"Provide a comprehensive summary answering: {query}"
        response = self.query_engine.query(summary_query)
        
        return {
            "query": query,
            "answer": str(response),
            "mode": "summarize",
            "sources": self.context_manager.get_context_sources(query),
            "context_summary": self.context_manager.summarize_context(contexts),
            "total_chunks_analyzed": len(contexts),
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_all_documents(self, query: str) -> Dict:
        """Comprehensive analysis of ALL documents in the knowledge base using batched processing."""
        from llama_index.core import Settings
        import random
        import time
        
        log.info("Starting comprehensive analysis of all documents")
        
        # Retrieve representative chunks from ALL documents
        all_contexts = self.context_manager.retrieve_all_documents(chunks_per_doc=2)
        
        if not all_contexts:
            return {
                "query": query,
                "answer": "No documents found in the knowledge base.",
                "mode": "analyze_all",
                "sources": [],
                "retrieved_chunks": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Group contexts by document
        docs_map = {}
        for ctx in all_contexts:
            file_name = ctx['metadata'].get('file_name', 'Unknown')
            if file_name != 'Unknown':
                if file_name not in docs_map:
                    docs_map[file_name] = []
                docs_map[file_name].append(ctx)
        
        unique_docs = sorted(docs_map.keys())
        log.info(f"Analyzing {len(unique_docs)} documents with {len(all_contexts)} total chunks")
        
        # Process in batches to avoid token limits
        BATCH_SIZE = 1  # Process 1 paper at a time
        all_analyses = []
        
        for batch_start in range(0, len(unique_docs), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(unique_docs))
            batch_docs = unique_docs[batch_start:batch_end]
            
            log.info(f"Processing batch {batch_start//BATCH_SIZE + 1}: papers {batch_start+1}-{batch_end}")
            
            # Build context for this batch
            batch_contexts = []
            for doc_name in batch_docs:
                batch_contexts.extend(docs_map[doc_name])
            
            doc_context_text = "\\n\\n".join([
                f"[Document: {ctx['metadata'].get('file_name', 'Unknown')}]\\n{ctx['text']}"
                for ctx in batch_contexts
            ])
            
            # Create batch analysis prompt
            prompt = f"""You are analyzing a batch of research papers. Provide a structured analysis for EACH paper.

PAPERS IN THIS BATCH ({len(batch_docs)} papers):
{', '.join(batch_docs)}

DOCUMENT CONTENT:
{doc_context_text}

USER QUERY: {query}

For EACH of the {len(batch_docs)} papers listed above, provide:
1. Paper title/name
2. Main research question or objective
3. Key methodology
4. Main findings or contributions
5. Relevance to the query

Number each paper (Paper {batch_start+1}, Paper {batch_start+2}, etc.) and ensure ALL {len(batch_docs)} papers are analyzed."""
            
            # Get LLM response for this batch
            if Settings.llm is not None and hasattr(Settings.llm, 'complete'):
                log.info(f"Sending batch {batch_start//BATCH_SIZE + 1} ({len(prompt)} chars) to LLM")
                
                # Retry logic for failed batches
                batch_success = False
                batch_retries = 2
                
                for batch_attempt in range(batch_retries):
                    try:
                        llm_response = Settings.llm.complete(
                            prompt,
                            max_retries=3,
                            retry_delay=3.0
                        )
                        batch_answer = llm_response.text if hasattr(llm_response, 'text') else str(llm_response)
                        
                        # Check if response indicates an error
                        if batch_answer.startswith("Error:") or batch_answer.startswith("Connection failed"):
                            if batch_attempt < batch_retries - 1:
                                log.warning(f"Batch {batch_start//BATCH_SIZE + 1} returned error, retrying...")
                                time.sleep(5.0)
                                continue
                            else:
                                log.error(f"Batch {batch_start//BATCH_SIZE + 1} failed after retries")
                                all_analyses.append(f"\\n=== BATCH {batch_start//BATCH_SIZE + 1}: Error ===\\nFailed to analyze papers {batch_start+1}-{batch_end}: {batch_answer}")
                        else:
                            all_analyses.append(f"\\n=== BATCH {batch_start//BATCH_SIZE + 1}: Papers {batch_start+1}-{batch_end} ===\\n{batch_answer}")
                            batch_success = True
                            log.info(f"Batch {batch_start//BATCH_SIZE + 1} completed successfully")
                        
                        break  # Exit retry loop on success
                        
                    except Exception as e:
                        log.error(f"Exception in batch {batch_start//BATCH_SIZE + 1}, attempt {batch_attempt + 1}: {e}")
                        if batch_attempt < batch_retries - 1:
                            log.info(f"Retrying batch {batch_start//BATCH_SIZE + 1}...")
                            time.sleep(5.0)
                        else:
                            all_analyses.append(f"\\n=== BATCH {batch_start//BATCH_SIZE + 1}: Error ===\\nFailed to analyze papers {batch_start+1}-{batch_end} after {batch_retries} attempts: {str(e)}")
                
                # Add random pause between batches to avoid rate limiting
                if batch_end < len(unique_docs):  # Don't pause after last batch
                    pause_duration = random.uniform(1.0, 5.0)
                    log.info(f"Pausing for {pause_duration:.1f} seconds before next batch...")
                    time.sleep(pause_duration)
        
        # Combine all batch analyses
        combined_answer = "\\n\\n".join(all_analyses)
        
        # Add summary header
        final_answer = f"""COMPREHENSIVE ANALYSIS OF ALL {len(unique_docs)} RESEARCH PAPERS

Total papers analyzed: {len(unique_docs)}
Processed in {len(range(0, len(unique_docs), BATCH_SIZE))} batches

{combined_answer}

---
Summary: Successfully analyzed all {len(unique_docs)} papers in the knowledge base.
"""
        
        # Build sources list
        sources = []
        for file_name in unique_docs:
            sources.append({
                "file_name": file_name,
                "file_path": docs_map[file_name][0]['metadata'].get('file_path', ''),
                "relevance_score": 1.0
            })
        
        log.info(f"Analysis complete for {len(sources)} documents")
        
        return {
            "query": query,
            "answer": final_answer,
            "mode": "analyze_all",
            "sources": sources,
            "retrieved_chunks": len(all_contexts),
            "documents_analyzed": len(unique_docs),
            "batches_processed": len(range(0, len(unique_docs), BATCH_SIZE)),
            "timestamp": datetime.now().isoformat()
        }
    
    def multi_query_search(self, queries: List[str]) -> List[Dict]:
        """Execute multiple queries and return combined results"""
        
        results = []
        for query in queries:
            result = self.search(query, mode="simple", save_to_memory=False)
            results.append(result)
        
        return results
    
    def get_search_history(self, limit: Optional[int] = 10) -> List[Dict]:
        """Get recent search history"""
        from llama_index.core.llms import ChatMessage
        
        messages = self.memory_manager.get_messages(limit=limit * 2)
        
        searches = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                query_msg = messages[i]
                response_msg = messages[i + 1]
                
                # Handle both ChatMessage objects and dictionaries
                if isinstance(query_msg, ChatMessage):
                    query_role = query_msg.role
                    query_content = query_msg.content
                    query_timestamp = query_msg.additional_kwargs.get("timestamp", "")
                else:
                    query_role = query_msg.get("role")
                    query_content = query_msg.get("content")
                    query_timestamp = query_msg.get("timestamp")
                
                if isinstance(response_msg, ChatMessage):
                    response_content = response_msg.content
                    response_sources = response_msg.additional_kwargs.get("metadata", {}).get("sources", [])
                else:
                    response_content = response_msg.get("content")
                    response_sources = response_msg.get("metadata", {}).get("sources", [])
                
                if query_role == "user":
                    searches.append({
                        "query": query_content,
                        "answer": response_content,
                        "timestamp": query_timestamp,
                        "sources": response_sources
                    })
        
        return searches[-limit:]