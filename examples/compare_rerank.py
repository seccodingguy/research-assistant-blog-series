"""
examples/compare_rerank.py

Comparison script demonstrating MMR vs basic ranking performance.
Implements the comparison analysis described in Week 3 blog post.

Usage:
    python examples/compare_rerank.py --query "machine learning optimization" --limit 10
"""

import sys
import argparse
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.search_engine import SearchEngine
from core.re_ranker import DocumentReRanker
from utils.logger import log


def compare_ranking_strategies(query: str, limit: int = 10):
    """Compare basic vs MMR re-ranking strategies"""
    
    log.info(f"Comparing ranking strategies for query: '{query}'")
    log.info("="*60)
    
    # Initialize components
    search_engine = SearchEngine()
    re_ranker = DocumentReRanker()
    
    # Get initial search results (more than we need for re-ranking)
    log.info("1. Performing initial search...")
    start_time = time.time()
    
    # Search with higher limit for re-ranking
    initial_results = search_engine.search_papers(query, limit=limit*2)
    
    if not initial_results:
        log.error("No search results found")
        return
    
    search_time = time.time() - start_time
    log.info(f"   Found {len(initial_results)} initial results in {search_time:.2f}s")
    
    # Basic ranking (just top-k by relevance score)
    log.info("\n2. Basic Ranking (Top-K by relevance)")
    basic_results = initial_results[:limit]
    
    print("\nBASIC RANKING RESULTS:")
    print("-" * 40)
    for i, result in enumerate(basic_results, 1):
        title = result.get('title', 'Unknown Title')[:60]
        score = result.get('relevance_score', result.get('score', 0))
        year = result.get('year', 'Unknown')
        authors = result.get('authors', [])
        author_str = ', '.join(authors[:2]) if authors else 'Unknown Authors'
        if len(authors) > 2:
            author_str += f" et al."
        
        print(f"{i:2d}. [{score:.3f}] {title}")
        print(f"     {author_str} ({year})")
        print()
    
    # MMR re-ranking
    log.info("3. MMR Re-ranking (Diversity optimization)")
    start_time = time.time()
    
    # Convert results to format expected by re-ranker
    documents = []
    for result in initial_results:
        doc = {
            'id': result.get('id', f"doc_{len(documents)}"),
            'title': result.get('title', ''),
            'abstract': result.get('abstract', result.get('summary', '')),
            'content': result.get('content', ''),
            'year': result.get('year'),
            'authors': result.get('authors', []),
            'score': result.get('relevance_score', result.get('score', 0)),
            'metadata': result
        }
        documents.append(doc)
    
    # Apply MMR re-ranking
    mmr_results = re_ranker.rerank_documents(
        documents=documents,
        query=query,
        top_k=limit,
        diversity_lambda=0.7  # Balance between relevance and diversity
    )
    
    rerank_time = time.time() - start_time
    log.info(f"   MMR re-ranking completed in {rerank_time:.2f}s")
    
    print("\nMMR RE-RANKING RESULTS:")
    print("-" * 40)
    for i, result in enumerate(mmr_results, 1):
        title = result['title'][:60]
        score = result['score']
        year = result.get('year', 'Unknown')
        authors = result.get('authors', [])
        author_str = ', '.join(authors[:2]) if authors else 'Unknown Authors'
        if len(authors) > 2:
            author_str += f" et al."
        
        print(f"{i:2d}. [{score:.3f}] {title}")
        print(f"     {author_str} ({year})")
        print()
    
    # Analyze differences
    log.info("4. Analyzing ranking differences...")
    
    basic_ids = [r.get('id', f"doc_{i}") for i, r in enumerate(basic_results)]
    mmr_ids = [r['id'] for r in mmr_results]
    
    # Calculate overlap
    overlap = len(set(basic_ids) & set(mmr_ids))
    overlap_percentage = (overlap / limit) * 100
    
    # Find position changes
    position_changes = []
    for i, mmr_id in enumerate(mmr_ids):
        if mmr_id in basic_ids:
            basic_pos = basic_ids.index(mmr_id)
            mmr_pos = i
            if basic_pos != mmr_pos:
                position_changes.append({
                    'id': mmr_id,
                    'title': mmr_results[i]['title'][:40],
                    'basic_pos': basic_pos + 1,
                    'mmr_pos': mmr_pos + 1,
                    'change': mmr_pos - basic_pos
                })
    
    # Calculate diversity metrics
    basic_years = [r.get('year') for r in basic_results if r.get('year')]
    mmr_years = [r.get('year') for r in mmr_results if r.get('year')]
    
    basic_year_diversity = len(set(basic_years)) if basic_years else 0
    mmr_year_diversity = len(set(mmr_years)) if mmr_years else 0
    
    # Calculate author diversity
    basic_authors = set()
    for result in basic_results:
        authors = result.get('authors', [])
        basic_authors.update(authors[:2])  # First 2 authors per paper
    
    mmr_authors = set()
    for result in mmr_results:
        authors = result.get('authors', [])
        mmr_authors.update(authors[:2])
    
    print("\nRANKING COMPARISON ANALYSIS:")
    print("=" * 40)
    print(f"Overlap between rankings: {overlap}/{limit} ({overlap_percentage:.1f}%)")
    print(f"Position changes: {len(position_changes)} documents")
    print(f"Year diversity - Basic: {basic_year_diversity}, MMR: {mmr_year_diversity}")
    print(f"Author diversity - Basic: {len(basic_authors)}, MMR: {len(mmr_authors)}")
    
    if position_changes:
        print(f"\nTop position changes:")
        for change in sorted(position_changes, key=lambda x: abs(x['change']), reverse=True)[:5]:
            direction = "↑" if change['change'] < 0 else "↓"
            print(f"  {direction} {change['title']}: #{change['basic_pos']} → #{change['mmr_pos']}")
    
    # Performance summary
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"Search time: {search_time:.2f}s")
    print(f"Re-ranking time: {rerank_time:.2f}s")
    print(f"Total time: {search_time + rerank_time:.2f}s")
    
    return {
        'basic_results': basic_results,
        'mmr_results': mmr_results,
        'analysis': {
            'overlap_percentage': overlap_percentage,
            'position_changes': len(position_changes),
            'basic_year_diversity': basic_year_diversity,
            'mmr_year_diversity': mmr_year_diversity,
            'basic_author_diversity': len(basic_authors),
            'mmr_author_diversity': len(mmr_authors),
            'search_time': search_time,
            'rerank_time': rerank_time
        }
    }


def run_batch_comparison(queries: list, limit: int = 10):
    """Run comparison across multiple queries"""
    
    log.info(f"Running batch comparison for {len(queries)} queries")
    
    results = []
    total_start = time.time()
    
    for i, query in enumerate(queries, 1):
        log.info(f"\n[{i}/{len(queries)}] Testing query: {query}")
        
        try:
            result = compare_ranking_strategies(query, limit)
            if result:
                results.append({
                    'query': query,
                    'analysis': result['analysis']
                })
        except Exception as e:
            log.error(f"Error processing query '{query}': {e}")
    
    total_time = time.time() - total_start
    
    # Aggregate results
    if results:
        avg_overlap = sum(r['analysis']['overlap_percentage'] for r in results) / len(results)
        avg_position_changes = sum(r['analysis']['position_changes'] for r in results) / len(results)
        avg_year_improvement = sum(
            r['analysis']['mmr_year_diversity'] - r['analysis']['basic_year_diversity'] 
            for r in results
        ) / len(results)
        avg_author_improvement = sum(
            r['analysis']['mmr_author_diversity'] - r['analysis']['basic_author_diversity'] 
            for r in results
        ) / len(results)
        
        print(f"\n" + "="*60)
        print(f"BATCH COMPARISON SUMMARY ({len(results)} queries)")
        print(f"="*60)
        print(f"Average overlap: {avg_overlap:.1f}%")
        print(f"Average position changes: {avg_position_changes:.1f}")
        print(f"Average year diversity improvement: {avg_year_improvement:.1f}")
        print(f"Average author diversity improvement: {avg_author_improvement:.1f}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per query: {total_time/len(results):.2f}s")


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(
        description="Compare basic vs MMR re-ranking strategies"
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        default="machine learning optimization techniques",
        help="Search query to test"
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10,
        help="Number of results to return"
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help="Run batch comparison with predefined queries"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Predefined queries for comprehensive testing
        test_queries = [
            "neural networks deep learning",
            "reinforcement learning algorithms",
            "natural language processing transformers",
            "computer vision object detection",
            "machine learning optimization",
            "artificial intelligence ethics",
            "data mining classification techniques",
            "graph neural networks applications"
        ]
        run_batch_comparison(test_queries, args.limit)
    else:
        compare_ranking_strategies(args.query, args.limit)


if __name__ == "__main__":
    main()