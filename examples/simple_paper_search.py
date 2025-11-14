#!/usr/bin/env python3
"""
Simple example: Search for papers and download PDFs.

This is a minimal example showing how to use the paper search service.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.paper_search_service import PaperSearchService


async def simple_search():
    """Simple search example."""
    print("="*60)
    print("Simple Paper Search Example")
    print("="*60)
    
    # Initialize service
    service = PaperSearchService(download_directory="./downloads/examples")
    await service.initialize()
    
    try:
        # Search for papers
        query = "machine learning"
        print(f"\nSearching for: '{query}'")
        
        results = await service.search(
            query=query,
            max_results_per_source=5,
            download_pdfs=False
        )
        
        # Display results
        print(f"\n✓ Found {results['total_results']} papers")
        print(f"  Sources: {', '.join(results['sources_searched'])}")
        
        if results['results']:
            print("\nTop 5 results:")
            for i, paper in enumerate(results['results'][:5], 1):
                print(f"\n{i}. {paper['title']}")
                if paper['authors']:
                    authors = ", ".join(paper['authors'][:3])
                    if len(paper['authors']) > 3:
                        authors += " et al."
                    print(f"   Authors: {authors}")
                print(f"   Year: {paper['year']}")
                print(f"   Source: {paper['source']}")
                if paper['doi']:
                    print(f"   DOI: {paper['doi']}")
    
    finally:
        await service.shutdown()


async def search_and_download():
    """Search and download example."""
    print("\n" + "="*60)
    print("Search and Download Example")
    print("="*60)
    
    # Initialize service
    service = PaperSearchService(download_directory="./downloads/examples")
    await service.initialize()
    
    try:
        # Search and download
        query = "neural networks"
        print(f"\nSearching and downloading: '{query}'")
        
        results = await service.search_and_download(
            query=query,
            sources=['arxiv'],  # Use only arXiv for fast results
            max_results=3,
            max_concurrent_downloads=2
        )
        
        # Display results
        print(f"\n✓ Found {results['total_results']} papers")
        
        if 'downloads' in results:
            downloads = results['downloads']
            print(f"\nDownload Summary:")
            print(f"  Successful: {downloads['successful']}")
            print(f"  Failed: {downloads['failed']}")
            print(f"  Skipped (already exist): {downloads['skipped']}")
            
            # Show successful downloads
            successful = [r for r in downloads['results'] 
                         if r['success'] and r.get('reason') != 'already_exists']
            
            if successful:
                print(f"\nDownloaded {len(successful)} new PDFs:")
                for result in successful:
                    print(f"  ✓ {result['title'][:60]}")
                    if 'path' in result:
                        print(f"    → {result['path']}")
    
    finally:
        await service.shutdown()


async def view_statistics():
    """View service statistics."""
    print("\n" + "="*60)
    print("Service Statistics Example")
    print("="*60)
    
    # Initialize service
    service = PaperSearchService(download_directory="./downloads/examples")
    await service.initialize()
    
    try:
        # Perform a few searches
        queries = ["AI", "machine learning", "deep learning"]
        
        print("\nPerforming sample searches...")
        for query in queries:
            print(f"  Searching: {query}")
            await service.search(query, max_results_per_source=3, download_pdfs=False)
        
        # Get and display statistics
        stats = service.get_stats()
        
        print("\n" + "-"*60)
        print("Statistics:")
        print("-"*60)
        print(f"Total searches: {stats['total_searches']}")
        print(f"Total results: {stats['total_results']}")
        print(f"Total downloads: {stats['total_downloads']}")
        print(f"Successful: {stats['successful_downloads']}")
        print(f"Failed: {stats['failed_downloads']}")
        
        print(f"\nEnabled engines: {', '.join(stats['enabled_engines'])}")
        
        if stats['searches_by_engine']:
            print("\nPer-engine statistics:")
            for engine, count in stats['searches_by_engine'].items():
                results_count = stats['results_by_engine'].get(engine, 0)
                print(f"  {engine}:")
                print(f"    Searches: {count}")
                print(f"    Results: {results_count}")
                if count > 0:
                    avg = results_count / count
                    print(f"    Avg results/search: {avg:.1f}")
    
    finally:
        await service.shutdown()


async def main():
    """Run all examples."""
    try:
        # Run examples
        await simple_search()
        await search_and_download()
        await view_statistics()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
