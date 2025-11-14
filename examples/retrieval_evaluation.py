#!/usr/bin/env python3
"""
examples/retrieval_evaluation.py

Comprehensive RAG evaluation framework implementing the metrics described in Week 3 blog post.
Provides systematic quality measurement across retrieval, generation, and user experience dimensions.

Usage:
    python examples/retrieval_evaluation.py --mode evaluate_all
    python examples/retrieval_evaluation.py --mode retrieval_only --dataset test_queries.json
"""

import json
import re
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse

from agents.pdf_agent import PDFAgent
from core.context_manager import ContextManager
from core.search_engine import SearchEngine
from core.memory_manager import MemoryManager
from utils.logger import log


class ResponseQualityMetrics:
    """Track and analyze response quality across different search modes"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def evaluate_response(self, response: str, context: str, 
                         query: str, mode: str) -> Dict[str, float]:
        """Compute comprehensive quality metrics for a response"""
        
        metrics = {}
        
        # Citation density: citations per 100 words
        citation_patterns = [
            r'\[Source[s]?: [^\]]+\]',  # [Source: filename.pdf] or [Sources: file1.pdf, file2.pdf]
            r'\(Source: [^)]+\)',       # (Source: filename.pdf)
            r'Source: [^\n\.]+'         # Source: filename.pdf
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, response, re.IGNORECASE))
        
        word_count = len(response.split())
        metrics['citation_density'] = (citation_count / word_count) * 100 if word_count > 0 else 0
        
        # Context utilization: fraction of context sources mentioned
        context_sources = set()
        context_patterns = [
            r'\*\*Source: ([^(]+)',      # **Source: filename.pdf (format from context_manager)
            r'Document: ([^\]]+)',       # [Document: filename.pdf]
            r'File: ([^\n\]]+)'         # File: filename.pdf
        ]
        
        for pattern in context_patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                # Clean up source name
                clean_source = match.strip().split('(')[0].strip()
                if clean_source and clean_source != 'Unknown':
                    context_sources.add(clean_source)
        
        # Extract sources mentioned in response
        response_sources = set()
        for pattern in citation_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                # Extract filename from citation
                # Handle patterns like [Source: filename.pdf] or [Sources: file1.pdf, file2.pdf]
                source_content = re.sub(r'^\[Sources?:\s*', '', match, flags=re.IGNORECASE)
                source_content = re.sub(r'\]$', '', source_content)
                
                for source in source_content.split(','):
                    clean_source = source.strip().split('(')[0].strip()
                    if clean_source:
                        response_sources.add(clean_source)
        
        if context_sources:
            # Count how many context sources are actually cited
            cited_context_sources = len(response_sources & context_sources)
            metrics['context_utilization'] = cited_context_sources / len(context_sources)
        else:
            metrics['context_utilization'] = 0
        
        # Response structure analysis
        metrics['response_length'] = word_count
        metrics['has_summary'] = 1 if re.search(r'\*\*Summary\*\*:', response, re.IGNORECASE) else 0
        metrics['has_analysis'] = 1 if re.search(r'\*\*.*Analysis\*\*:', response, re.IGNORECASE) else 0
        metrics['has_insights'] = 1 if re.search(r'\*\*.*Insights?\*\*:', response, re.IGNORECASE) else 0
        metrics['has_limitations'] = 1 if re.search(r'\*\*Limitations?\*\*:', response, re.IGNORECASE) else 0
        
        # Structure completeness score
        structure_components = [
            metrics['has_summary'], 
            metrics['has_analysis'],
            metrics['has_insights'], 
            metrics['has_limitations']
        ]
        metrics['structure_completeness'] = sum(structure_components) / len(structure_components)
        
        # Citation accuracy: verify citations reference actual context sources
        if response_sources and context_sources:
            accurate_citations = len(response_sources & context_sources)
            metrics['citation_accuracy'] = accurate_citations / len(response_sources) if response_sources else 0
        else:
            metrics['citation_accuracy'] = 0
        
        # Store for analysis
        self.metrics[mode].append(metrics)
        
        return metrics
    
    def get_mode_statistics(self, mode: str) -> Dict[str, float]:
        """Get aggregate statistics for a specific search mode"""
        if mode not in self.metrics:
            return {}
        
        mode_metrics = self.metrics[mode]
        
        if not mode_metrics:
            return {}
        
        return {
            'avg_citation_density': np.mean([m['citation_density'] for m in mode_metrics]),
            'avg_context_utilization': np.mean([m['context_utilization'] for m in mode_metrics]),
            'avg_response_length': np.mean([m['response_length'] for m in mode_metrics]),
            'avg_citation_accuracy': np.mean([m['citation_accuracy'] for m in mode_metrics]),
            'structure_completeness': np.mean([m['structure_completeness'] for m in mode_metrics]),
            'total_queries': len(mode_metrics),
            'std_citation_density': np.std([m['citation_density'] for m in mode_metrics]),
            'std_context_utilization': np.std([m['context_utilization'] for m in mode_metrics])
        }
    
    def compare_modes(self) -> Dict[str, Dict[str, float]]:
        """Compare metrics across all evaluated modes"""
        comparison = {}
        for mode in self.metrics.keys():
            comparison[mode] = self.get_mode_statistics(mode)
        return comparison


class RAGEvaluationFramework:
    """Comprehensive evaluation framework for RAG system quality"""
    
    def __init__(self, agent: PDFAgent):
        self.agent = agent
        self.evaluation_metrics = {
            'retrieval': ['precision', 'recall', 'mrr', 'ndcg'],
            'generation': ['faithfulness', 'answer_relevancy', 'context_precision'],
            'citation': ['citation_accuracy', 'citation_completeness'],
            'user_experience': ['response_time', 'user_satisfaction']
        }
        self.response_metrics = ResponseQualityMetrics()
    
    def evaluate_retrieval_quality(self, queries: List[str], 
                                  ground_truth: Dict,
                                  mode: str = "enhanced") -> Dict:
        """Evaluate retrieval performance using standard IR metrics"""
        
        results = {
            'precision_at_k': [],
            'recall_at_k': [],
            'mrr_scores': [],
            'ndcg_scores': []
        }
        
        log.info(f"Evaluating retrieval quality for {len(queries)} queries in {mode} mode")
        
        for query_id, query in enumerate(queries):
            try:
                # Get search results
                search_result = self.agent.search(query, mode=mode)
                
                # Extract document IDs from sources
                retrieved_docs = []
                for source in search_result.get('sources', []):
                    if isinstance(source, dict):
                        doc_id = source.get('file_name', '')
                        if doc_id:
                            retrieved_docs.append(doc_id)
                    else:
                        retrieved_docs.append(str(source))
                
                # Get ground truth relevant documents
                relevant_docs = ground_truth.get(str(query_id), ground_truth.get(query_id, []))
                
                if not relevant_docs:
                    log.warning(f"No ground truth for query {query_id}: {query[:50]}...")
                    continue
                
                # Calculate metrics for different k values
                for k in [5, 10, 20]:
                    retrieved_k = retrieved_docs[:k]
                    relevant_retrieved = set(retrieved_k) & set(relevant_docs)
                    
                    # Precision@K
                    precision_k = len(relevant_retrieved) / len(retrieved_k) if retrieved_k else 0
                    results['precision_at_k'].append(precision_k)
                    
                    # Recall@K
                    recall_k = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
                    results['recall_at_k'].append(recall_k)
                
                # Mean Reciprocal Rank (MRR)
                mrr_score = 0
                for rank, doc_id in enumerate(retrieved_docs, 1):
                    if doc_id in relevant_docs:
                        mrr_score = 1.0 / rank
                        break
                results['mrr_scores'].append(mrr_score)
                
                # NDCG (simplified version)
                dcg = sum((1 if doc_id in relevant_docs else 0) / np.log2(rank + 1) 
                         for rank, doc_id in enumerate(retrieved_docs, 1) if rank <= 10)
                
                ideal_length = min(len(relevant_docs), 10)
                idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_length + 1))
                
                ndcg = dcg / idcg if idcg > 0 else 0
                results['ndcg_scores'].append(ndcg)
                
            except Exception as e:
                log.error(f"Error evaluating query {query_id}: {e}")
                continue
        
        # Aggregate results
        aggregated = {}
        if results['precision_at_k']:
            aggregated = {
                'mean_precision_at_k': np.mean(results['precision_at_k']),
                'mean_recall_at_k': np.mean(results['recall_at_k']),
                'mean_mrr': np.mean(results['mrr_scores']),
                'mean_ndcg': np.mean(results['ndcg_scores']),
                'std_precision_at_k': np.std(results['precision_at_k']),
                'std_recall_at_k': np.std(results['recall_at_k']),
                'queries_evaluated': len(results['mrr_scores'])
            }
        
        return aggregated
    
    def evaluate_generation_quality(self, queries: List[str], 
                                   mode: str = "enhanced") -> Dict:
        """Evaluate response generation quality"""
        
        generation_results = []
        
        log.info(f"Evaluating generation quality for {len(queries)} queries")
        
        for query in queries:
            try:
                # Measure response time
                start_time = time.time()
                result = self.agent.search(query, mode=mode)
                end_time = time.time()
                
                response_time = end_time - start_time
                response = result.get('answer', '')
                
                # Get context information for quality analysis
                context = ""
                if hasattr(self.agent, 'context_manager'):
                    context_data = self.agent.context_manager.build_enhanced_context(query)
                    context = context_data.get('document_context', '')
                
                # Evaluate response quality
                quality_metrics = self.response_metrics.evaluate_response(
                    response, context, query, mode
                )
                
                generation_results.append({
                    'query': query,
                    'response_time': response_time,
                    'response_length': len(response.split()),
                    **quality_metrics
                })
                
            except Exception as e:
                log.error(f"Error evaluating generation for query '{query[:50]}...': {e}")
                continue
        
        if not generation_results:
            return {}
        
        # Aggregate results
        aggregated = {
            'mean_response_time': np.mean([r['response_time'] for r in generation_results]),
            'mean_citation_density': np.mean([r['citation_density'] for r in generation_results]),
            'mean_context_utilization': np.mean([r['context_utilization'] for r in generation_results]),
            'mean_citation_accuracy': np.mean([r['citation_accuracy'] for r in generation_results]),
            'mean_structure_completeness': np.mean([r['structure_completeness'] for r in generation_results]),
            'responses_evaluated': len(generation_results)
        }
        
        return aggregated
    
    def evaluate_citation_quality(self, queries: List[str], mode: str = "enhanced") -> Dict:
        """Evaluate citation accuracy and completeness"""
        
        citation_results = []
        
        for query in queries:
            try:
                result = self.agent.search(query, mode=mode)
                response = result.get('answer', '')
                
                # Get context for comparison
                context = ""
                if hasattr(self.agent, 'context_manager'):
                    context_data = self.agent.context_manager.build_enhanced_context(query)
                    context = context_data.get('document_context', '')
                
                # Extract citations from response
                citation_pattern = r'\[Source[s]?: ([^\]]+)\]'
                response_citations = re.findall(citation_pattern, response, re.IGNORECASE)
                
                # Expand multiple citations
                expanded_citations = set()
                for citation in response_citations:
                    for source in citation.split(','):
                        expanded_citations.add(source.strip())
                
                # Extract available sources from context
                context_sources = set()
                context_pattern = r'\*\*Source: ([^(]+)'
                context_matches = re.findall(context_pattern, context)
                for match in context_matches:
                    clean_source = match.strip().split('(')[0].strip()
                    if clean_source and clean_source != 'Unknown':
                        context_sources.add(clean_source)
                
                # Citation Accuracy: Are cited sources actually in context?
                citation_accuracy = 0
                if expanded_citations:
                    accurate_citations = expanded_citations & context_sources
                    citation_accuracy = len(accurate_citations) / len(expanded_citations)
                
                # Citation Completeness: Are all context sources utilized?
                citation_completeness = 0
                if context_sources:
                    citation_completeness = len(expanded_citations & context_sources) / len(context_sources)
                
                citation_results.append({
                    'query': query,
                    'citation_accuracy': citation_accuracy,
                    'citation_completeness': citation_completeness,
                    'total_citations': len(expanded_citations),
                    'available_sources': len(context_sources)
                })
                
            except Exception as e:
                log.error(f"Error evaluating citations for query '{query[:50]}...': {e}")
                continue
        
        if not citation_results:
            return {}
        
        return {
            'mean_citation_accuracy': np.mean([r['citation_accuracy'] for r in citation_results]),
            'mean_citation_completeness': np.mean([r['citation_completeness'] for r in citation_results]),
            'std_citation_accuracy': np.std([r['citation_accuracy'] for r in citation_results]),
            'std_citation_completeness': np.std([r['citation_completeness'] for r in citation_results]),
            'total_evaluations': len(citation_results)
        }
    
    def benchmark_search_modes(self, test_queries: List[str], 
                              modes: List[str] = None) -> Dict:
        """Comprehensive benchmark across multiple search modes"""
        
        if modes is None:
            modes = ['simple', 'enhanced', 'summarize']
        
        benchmark_results = {}
        
        for mode in modes:
            log.info(f"Benchmarking mode: {mode}")
            
            mode_results = {
                'generation_quality': self.evaluate_generation_quality(test_queries, mode),
                'citation_quality': self.evaluate_citation_quality(test_queries, mode),
                'mode': mode
            }
            
            benchmark_results[mode] = mode_results
        
        # Add comparison summary
        comparison = self.response_metrics.compare_modes()
        benchmark_results['comparison_summary'] = comparison
        
        return benchmark_results


def load_test_dataset(dataset_path: Path) -> Tuple[List[str], Dict]:
    """Load test queries and ground truth data"""
    
    if not dataset_path.exists():
        # Create a sample dataset
        log.info(f"Creating sample test dataset at {dataset_path}")
        
        sample_data = {
            "queries": [
                "What are agentic workflows?",
                "How does the Agent2Agent protocol work?",
                "What are the key challenges in PDF parsing for academic documents?",
                "Compare vector search with knowledge graph approaches",
                "What are the benefits of hybrid retrieval systems?",
                "How can retrieval-augmented generation reduce hallucination?",
                "What evaluation metrics are important for RAG systems?",
                "How does context assembly impact response quality?",
                "What are the trade-offs between different embedding providers?",
                "How can caching improve RAG system performance?"
            ],
            "ground_truth": {
                "0": ["Week1.md", "ARCHITECTURE.md"],
                "1": ["sample_paper.pdf", "protocol_spec.pdf"],
                "2": ["Week2.md", "pdf_parser.py"],
                "3": ["Week3.md", "graph_manager.py"],
                "4": ["Week3.md", "context_manager.py"],
                "5": ["Week1.md", "Week3.md"],
                "6": ["Week3.md", "retrieval_evaluation.py"],
                "7": ["Week3.md", "context_manager.py"],
                "8": ["Week2.md", "settings.py"],
                "9": ["Week3.md", "caching_system.py"]
            }
        }
        
        with open(dataset_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    return data['queries'], data['ground_truth']


def main():
    parser = argparse.ArgumentParser(description="RAG System Evaluation Framework")
    parser.add_argument("--mode", type=str, default="evaluate_all", 
                       choices=["evaluate_all", "retrieval_only", "generation_only", "benchmark_modes"],
                       help="Evaluation mode")
    parser.add_argument("--dataset", type=str, default="test_queries.json",
                       help="Path to test dataset")
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--search-modes", nargs='+', default=["simple", "enhanced"],
                       help="Search modes to benchmark")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset_path = Path(args.dataset)
    test_queries, ground_truth = load_test_dataset(dataset_path)
    
    # Initialize evaluation framework
    log.info("Initializing PDF Agent for evaluation...")
    agent = PDFAgent()
    evaluator = RAGEvaluationFramework(agent)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_mode': args.mode,
        'test_queries_count': len(test_queries),
        'search_modes': args.search_modes
    }
    
    try:
        if args.mode == "evaluate_all":
            log.info("Running comprehensive evaluation...")
            
            # Evaluate each search mode
            for mode in args.search_modes:
                log.info(f"Evaluating search mode: {mode}")
                
                mode_results = {
                    'retrieval_quality': evaluator.evaluate_retrieval_quality(
                        test_queries, ground_truth, mode
                    ),
                    'generation_quality': evaluator.evaluate_generation_quality(
                        test_queries, mode
                    ),
                    'citation_quality': evaluator.evaluate_citation_quality(
                        test_queries, mode
                    )
                }
                
                results[f"{mode}_mode"] = mode_results
        
        elif args.mode == "retrieval_only":
            log.info("Evaluating retrieval quality only...")
            for mode in args.search_modes:
                results[f"{mode}_retrieval"] = evaluator.evaluate_retrieval_quality(
                    test_queries, ground_truth, mode
                )
        
        elif args.mode == "generation_only":
            log.info("Evaluating generation quality only...")
            for mode in args.search_modes:
                results[f"{mode}_generation"] = evaluator.evaluate_generation_quality(
                    test_queries, mode
                )
        
        elif args.mode == "benchmark_modes":
            log.info("Running comprehensive mode benchmark...")
            results['benchmark'] = evaluator.benchmark_search_modes(
                test_queries, args.search_modes
            )
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        log.info(f"Evaluation completed. Results saved to {output_path}")
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        if args.mode == "benchmark_modes" and 'benchmark' in results:
            comparison = results['benchmark'].get('comparison_summary', {})
            for mode, stats in comparison.items():
                print(f"\n{mode.upper()} MODE:")
                print(f"  Citation Density: {stats.get('avg_citation_density', 0):.1f} cites/100 words")
                print(f"  Context Utilization: {stats.get('avg_context_utilization', 0):.2f}")
                print(f"  Citation Accuracy: {stats.get('avg_citation_accuracy', 0):.2f}")
                print(f"  Structure Completeness: {stats.get('structure_completeness', 0):.2f}")
                print(f"  Queries Evaluated: {stats.get('total_queries', 0)}")
        
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        results['error'] = str(e)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        raise


if __name__ == "__main__":
    main()