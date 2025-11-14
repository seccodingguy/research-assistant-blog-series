"""
examples/benchmark_search_modes.py

Benchmark script for reproducing search mode performance comparisons.
Implements the performance analysis described in Week 3 blog post.

Usage:
    python examples/benchmark_search_modes.py --dataset test_queries.json
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.research_assistant_agent import ResearchAssistantAgent
from core.search_engine import SearchEngine
from examples.retrieval_evaluation import RAGEvaluationFramework
from utils.logger import log


class SearchModeBenchmark:
    """Benchmark different search modes for performance comparison"""
    
    def __init__(self):
        self.agent = ResearchAssistantAgent()
        self.search_engine = SearchEngine()
        self.evaluator = RAGEvaluationFramework()
        
        # Search modes to benchmark
        self.search_modes = [
            'basic',
            'enhanced',
            'semantic',
            'hybrid'
        ]
        
        # Performance metrics
        self.metrics = {}
        
    def benchmark_query(self, query: str, ground_truth: Dict = None) -> Dict:
        """Benchmark a single query across all search modes"""
        
        log.info(f"Benchmarking query: '{query[:50]}...'")
        
        results = {}
        
        for mode in self.search_modes:
            log.info(f"  Testing mode: {mode}")
            
            start_time = time.time()
            
            try:
                # Use search engine directly for consistent comparison
                if mode == 'basic':
                    search_results = self.search_engine.search_papers(
                        query, limit=10
                    )
                    response = self._format_basic_response(search_results)
                    context = search_results
                    
                else:
                    # Use agent for enhanced modes
                    response, context = self.agent.process_query(query, mode=mode)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Evaluate response quality if ground truth provided
                quality_metrics = {}
                if ground_truth:
                    quality_metrics = self.evaluator.evaluate_response(
                        query=query,
                        response=response,
                        retrieved_contexts=context,
                        ground_truth=ground_truth
                    )
                
                results[mode] = {
                    'response': response,
                    'context': context,
                    'response_time': response_time,
                    'context_count': len(context) if isinstance(context, list) else 1,
                    'response_length': len(response) if response else 0,
                    'quality_metrics': quality_metrics,
                    'success': True
                }
                
                log.info(f"    ✓ Completed in {response_time:.2f}s")
                
            except Exception as e:
                log.error(f"    ✗ Failed: {str(e)}")
                results[mode] = {
                    'error': str(e),
                    'response_time': time.time() - start_time,
                    'success': False
                }
        
        return results
    
    def _format_basic_response(self, search_results: List[Dict]) -> str:
        """Format basic search results into response format"""
        
        if not search_results:
            return "No relevant papers found."
        
        response_parts = [
            "Based on the search results, here are the most relevant papers:\n"
        ]
        
        for i, result in enumerate(search_results[:5], 1):
            title = result.get('title', 'Unknown Title')
            authors = result.get('authors', [])
            year = result.get('year', 'Unknown Year')
            abstract = result.get('abstract', result.get('summary', ''))
            
            author_str = ', '.join(authors[:3]) if authors else 'Unknown Authors'
            if len(authors) > 3:
                author_str += " et al."
            
            response_parts.append(
                f"{i}. **{title}** ({year})\n"
                f"   Authors: {author_str}\n"
                f"   Abstract: {abstract[:200]}...\n"
            )
        
        return '\n'.join(response_parts)
    
    def run_benchmark_suite(self, test_queries: List[Dict]) -> Dict:
        """Run comprehensive benchmark across all test queries"""
        
        log.info(f"Starting benchmark suite with {len(test_queries)} queries")
        log.info("="*60)
        
        all_results = {}
        performance_summary = {mode: {
            'total_time': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0,
            'avg_context_count': 0,
            'avg_response_length': 0,
            'quality_scores': []
        } for mode in self.search_modes}
        
        start_time = time.time()
        
        for i, query_data in enumerate(test_queries, 1):
            query = query_data.get('query', query_data.get('question', ''))
            ground_truth = query_data.get('ground_truth')
            
            log.info(f"\n[{i}/{len(test_queries)}] Processing: {query[:50]}...")
            
            try:
                query_results = self.benchmark_query(query, ground_truth)
                all_results[f"query_{i}"] = {
                    'query': query,
                    'results': query_results
                }
                
                # Update performance summary
                for mode, result in query_results.items():
                    if result.get('success', False):
                        performance_summary[mode]['successful_queries'] += 1
                        performance_summary[mode]['total_time'] += result['response_time']
                        performance_summary[mode]['avg_context_count'] += result.get('context_count', 0)
                        performance_summary[mode]['avg_response_length'] += result.get('response_length', 0)
                        
                        # Add quality metrics if available
                        quality_metrics = result.get('quality_metrics', {})
                        if quality_metrics:
                            overall_score = quality_metrics.get('overall_score', 0)
                            performance_summary[mode]['quality_scores'].append(overall_score)
                    else:
                        performance_summary[mode]['failed_queries'] += 1
                
            except Exception as e:
                log.error(f"Error processing query {i}: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate averages
        for mode in self.search_modes:
            stats = performance_summary[mode]
            successful = stats['successful_queries']
            
            if successful > 0:
                stats['avg_response_time'] = stats['total_time'] / successful
                stats['avg_context_count'] = stats['avg_context_count'] / successful
                stats['avg_response_length'] = stats['avg_response_length'] / successful
                
                if stats['quality_scores']:
                    stats['avg_quality_score'] = statistics.mean(stats['quality_scores'])
                    stats['quality_std'] = statistics.stdev(stats['quality_scores']) if len(stats['quality_scores']) > 1 else 0
        
        # Generate comparison report
        self._generate_comparison_report(performance_summary, total_time)
        
        return {
            'results': all_results,
            'performance_summary': performance_summary,
            'total_time': total_time,
            'query_count': len(test_queries)
        }
    
    def _generate_comparison_report(self, performance_summary: Dict, total_time: float):
        """Generate detailed comparison report"""
        
        print("\n" + "="*80)
        print("SEARCH MODE BENCHMARK RESULTS")
        print("="*80)
        
        # Performance table
        print(f"\n{'Mode':<12} {'Success':<8} {'Avg Time':<10} {'Contexts':<10} {'Length':<10} {'Quality':<10}")
        print("-" * 70)
        
        for mode in self.search_modes:
            stats = performance_summary[mode]
            success_rate = stats['successful_queries'] / (stats['successful_queries'] + stats['failed_queries']) * 100 if (stats['successful_queries'] + stats['failed_queries']) > 0 else 0
            
            quality_str = f"{stats.get('avg_quality_score', 0):.2f}" if 'avg_quality_score' in stats else "N/A"
            
            print(f"{mode:<12} {success_rate:>6.1f}% {stats['avg_response_time']:>8.2f}s {stats['avg_context_count']:>8.1f} {stats['avg_response_length']:>8.0f} {quality_str:>8}")
        
        # Speed comparison
        print(f"\nSPEED COMPARISON:")
        print("-" * 30)
        fastest_mode = min(
            [mode for mode in self.search_modes if performance_summary[mode]['successful_queries'] > 0],
            key=lambda m: performance_summary[m]['avg_response_time']
        )
        fastest_time = performance_summary[fastest_mode]['avg_response_time']
        
        for mode in self.search_modes:
            if performance_summary[mode]['successful_queries'] > 0:
                time_ratio = performance_summary[mode]['avg_response_time'] / fastest_time
                print(f"{mode:<12}: {time_ratio:.1f}x slower than fastest")
        
        # Quality comparison (if available)
        quality_modes = [mode for mode in self.search_modes 
                        if 'avg_quality_score' in performance_summary[mode]]
        
        if quality_modes:
            print(f"\nQUALITY COMPARISON:")
            print("-" * 30)
            quality_modes.sort(key=lambda m: performance_summary[m]['avg_quality_score'], reverse=True)
            
            for mode in quality_modes:
                score = performance_summary[mode]['avg_quality_score']
                std = performance_summary[mode]['quality_std']
                print(f"{mode:<12}: {score:.3f} ± {std:.3f}")
        
        # Recommendations
        print(f"\nRECOMMENDations:")
        print("-" * 30)
        
        if fastest_mode:
            print(f"• Fastest mode: {fastest_mode} ({fastest_time:.2f}s avg)")
        
        if quality_modes:
            best_quality = quality_modes[0]
            best_score = performance_summary[best_quality]['avg_quality_score']
            print(f"• Highest quality: {best_quality} ({best_score:.3f} score)")
        
        # Context efficiency
        most_efficient = min(
            [mode for mode in self.search_modes if performance_summary[mode]['successful_queries'] > 0],
            key=lambda m: performance_summary[m]['avg_context_count']
        )
        print(f"• Most context-efficient: {most_efficient} ({performance_summary[most_efficient]['avg_context_count']:.1f} contexts avg)")
        
        print(f"\nTotal benchmark time: {total_time:.2f}s")


def load_test_queries(file_path: str) -> List[Dict]:
    """Load test queries from JSON file"""
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning(f"Test file {file_path} not found, using default queries")
        return get_default_test_queries()
    except json.JSONDecodeError as e:
        log.error(f"Error parsing JSON file: {e}")
        return get_default_test_queries()


def get_default_test_queries() -> List[Dict]:
    """Get default test queries for benchmarking"""
    
    return [
        {
            "query": "neural networks for natural language processing",
            "ground_truth": {
                "key_concepts": ["neural networks", "NLP", "transformers"],
                "expected_papers": 5
            }
        },
        {
            "query": "reinforcement learning in robotics applications",
            "ground_truth": {
                "key_concepts": ["reinforcement learning", "robotics", "control"],
                "expected_papers": 3
            }
        },
        {
            "query": "computer vision object detection algorithms",
            "ground_truth": {
                "key_concepts": ["computer vision", "object detection", "CNN"],
                "expected_papers": 4
            }
        },
        {
            "query": "machine learning optimization techniques",
            "ground_truth": {
                "key_concepts": ["optimization", "gradient descent", "ML"],
                "expected_papers": 5
            }
        },
        {
            "query": "graph neural networks for social network analysis",
            "ground_truth": {
                "key_concepts": ["graph neural networks", "social networks", "GNN"],
                "expected_papers": 3
            }
        },
        {
            "query": "transfer learning in medical image analysis",
            "ground_truth": {
                "key_concepts": ["transfer learning", "medical imaging", "CNN"],
                "expected_papers": 4
            }
        },
        {
            "query": "attention mechanisms in transformer architectures",
            "ground_truth": {
                "key_concepts": ["attention", "transformers", "self-attention"],
                "expected_papers": 5
            }
        },
        {
            "query": "federated learning privacy preservation methods",
            "ground_truth": {
                "key_concepts": ["federated learning", "privacy", "distributed"],
                "expected_papers": 3
            }
        }
    ]


def create_sample_test_file(output_path: str):
    """Create sample test queries file"""
    
    queries = get_default_test_queries()
    
    with open(output_path, 'w') as f:
        json.dump(queries, f, indent=2)
    
    log.info(f"Sample test queries saved to {output_path}")


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(
        description="Benchmark search mode performance"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help="JSON file containing test queries"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Output file for detailed results (JSON)"
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help="Create sample test queries file"
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        default=['basic', 'enhanced', 'semantic', 'hybrid'],
        help="Search modes to benchmark"
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        output_path = args.dataset or "test_queries.json"
        create_sample_test_file(output_path)
        return
    
    # Load test queries
    if args.dataset:
        test_queries = load_test_queries(args.dataset)
    else:
        test_queries = get_default_test_queries()
    
    if not test_queries:
        log.error("No test queries available")
        return
    
    # Initialize benchmark
    benchmark = SearchModeBenchmark()
    benchmark.search_modes = args.modes
    
    # Run benchmark
    results = benchmark.run_benchmark_suite(test_queries)
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()