"""
Demo script for paper search and download functionality.

This demonstrates how to use the PaperSearchService to search for
academic papers and download PDFs.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.paper_search_service import PaperSearchService
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


async def demo_basic_search():
    """Demonstrate basic paper search."""
    console.print(Panel("[bold cyan]Demo 1: Basic Paper Search[/bold cyan]"))
    
    # Initialize service
    service = PaperSearchService(download_directory="./downloads/demo")
    await service.initialize()
    
    try:
        # Perform search
        query = "machine learning transformers"
        console.print(f"\n[yellow]Searching for:[/yellow] '{query}'")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            
            results = await service.search(
                query=query,
                max_results_per_source=5,
                download_pdfs=False
            )
            
            progress.update(task, completed=True)
        
        # Display results
        console.print(f"\n[green]Found {results['total_results']} papers from "
                     f"{len(results['sources_searched'])} sources[/green]")
        
        if results['results']:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Title", style="cyan", width=50)
            table.add_column("Authors", style="green", width=30)
            table.add_column("Year", style="yellow", width=6)
            table.add_column("Source", style="blue", width=15)
            
            for i, paper in enumerate(results['results'][:10], 1):
                authors = ", ".join(paper['authors'][:3]) if paper['authors'] else "N/A"
                if paper['authors'] and len(paper['authors']) > 3:
                    authors += " et al."
                
                table.add_row(
                    paper['title'][:50] + "..." if len(paper['title']) > 50 
                        else paper['title'],
                    authors[:30],
                    str(paper['year']) if paper['year'] else "N/A",
                    paper['source']
                )
            
            console.print("\n", table)
    
    finally:
        await service.shutdown()


async def demo_search_and_download():
    """Demonstrate searching and downloading PDFs."""
    console.print(Panel("[bold cyan]Demo 2: Search and Download PDFs[/bold cyan]"))
    
    # Initialize service
    service = PaperSearchService(download_directory="./downloads/demo")
    await service.initialize()
    
    try:
        # Search and download
        query = "neural networks deep learning"
        console.print(f"\n[yellow]Searching and downloading:[/yellow] '{query}'")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching and downloading...", total=None)
            
            results = await service.search_and_download(
                query=query,
                sources=['arxiv', 'semantic_scholar'],  # Use specific sources
                max_results=5,
                max_concurrent_downloads=2
            )
            
            progress.update(task, completed=True)
        
        # Display results
        console.print(f"\n[green]Search Results:[/green]")
        console.print(f"  Papers found: {results['total_results']}")
        console.print(f"  Sources: {', '.join(results['sources_searched'])}")
        
        if 'downloads' in results:
            downloads = results['downloads']
            console.print(f"\n[green]Download Results:[/green]")
            console.print(f"  Downloadable: {downloads['downloadable']}")
            console.print(f"  Successful: {downloads['successful']}")
            console.print(f"  Failed: {downloads['failed']}")
            console.print(f"  Skipped: {downloads['skipped']}")
            
            # Show successful downloads
            successful = [r for r in downloads['results'] if r['success']]
            if successful:
                console.print("\n[cyan]Downloaded PDFs:[/cyan]")
                for result in successful[:5]:
                    console.print(f"  ✓ {result['title']}")
                    if 'path' in result:
                        console.print(f"    → {result['path']}")
    
    finally:
        await service.shutdown()


async def demo_search_by_doi():
    """Demonstrate searching by DOI."""
    console.print(Panel("[bold cyan]Demo 3: Search by DOI[/bold cyan]"))
    
    # Initialize service
    service = PaperSearchService(download_directory="./downloads/demo")
    await service.initialize()
    
    try:
        # Example DOI
        doi = "10.1038/nature14539"
        console.print(f"\n[yellow]Searching for DOI:[/yellow] {doi}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Looking up DOI...", total=None)
            
            result = await service.download_by_doi(doi)
            
            progress.update(task, completed=True)
        
        # Display result
        if result['success']:
            paper = result['paper']
            console.print("\n[green]Paper found:[/green]")
            console.print(f"  Title: {paper['title']}")
            console.print(f"  Authors: {', '.join(paper['authors'][:5])}")
            console.print(f"  Year: {paper['year']}")
            
            if 'download' in result:
                download = result['download']
                if download['success']:
                    console.print(f"\n[green]✓ PDF downloaded:[/green]")
                    console.print(f"  {download.get('path', 'N/A')}")
                else:
                    console.print(f"\n[red]✗ Download failed:[/red]")
                    console.print(f"  {download.get('message', 'Unknown error')}")
        else:
            console.print(f"[red]Paper not found for DOI: {doi}[/red]")
    
    finally:
        await service.shutdown()


async def demo_service_stats():
    """Demonstrate service statistics."""
    console.print(Panel("[bold cyan]Demo 4: Service Statistics[/bold cyan]"))
    
    # Initialize service
    service = PaperSearchService(download_directory="./downloads/demo")
    await service.initialize()
    
    try:
        # Perform some searches
        console.print("\n[yellow]Performing multiple searches...[/yellow]")
        
        queries = [
            "quantum computing",
            "natural language processing",
            "computer vision"
        ]
        
        for query in queries:
            console.print(f"  Searching: {query}")
            await service.search(
                query=query,
                max_results_per_source=3,
                download_pdfs=False
            )
        
        # Display statistics
        stats = service.get_stats()
        
        console.print("\n[cyan]Service Statistics:[/cyan]")
        console.print(f"  Total searches: {stats['total_searches']}")
        console.print(f"  Total results: {stats['total_results']}")
        console.print(f"  Enabled engines: {', '.join(stats['enabled_engines'])}")
        
        if stats['searches_by_engine']:
            console.print("\n[cyan]Searches by engine:[/cyan]")
            for engine, count in stats['searches_by_engine'].items():
                results = stats['results_by_engine'].get(engine, 0)
                console.print(f"  {engine}: {count} searches, {results} results")
    
    finally:
        await service.shutdown()


async def demo_interactive():
    """Interactive demo - search for papers based on user input."""
    console.print(Panel("[bold cyan]Interactive Paper Search Demo[/bold cyan]"))
    
    # Initialize service
    service = PaperSearchService(download_directory="./downloads/interactive")
    await service.initialize()
    
    try:
        console.print("\n[yellow]Available search engines:[/yellow]")
        for engine in service.get_enabled_sources():
            console.print(f"  • {engine}")
        
        # Get user input
        console.print("\n")
        query = console.input("[bold blue]Enter search query:[/bold blue] ").strip()
        
        if not query:
            console.print("[red]No query provided, exiting[/red]")
            return
        
        download = console.input(
            "[bold blue]Download PDFs? (y/n):[/bold blue] "
        ).strip().lower()
        download_pdfs = download == 'y'
        
        max_results = console.input(
            "[bold blue]Max results per source (default 10):[/bold blue] "
        ).strip()
        max_results = int(max_results) if max_results.isdigit() else 10
        
        # Perform search
        console.print(f"\n[yellow]Searching...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            
            results = await service.search(
                query=query,
                max_results_per_source=max_results,
                download_pdfs=download_pdfs
            )
            
            progress.update(task, completed=True)
        
        # Display results
        console.print(f"\n[green]Found {results['total_results']} papers[/green]")
        
        if results['results']:
            for i, paper in enumerate(results['results'], 1):
                console.print(f"\n[bold cyan]{i}. {paper['title']}[/bold cyan]")
                if paper['authors']:
                    authors = ", ".join(paper['authors'][:3])
                    if len(paper['authors']) > 3:
                        authors += " et al."
                    console.print(f"   Authors: {authors}")
                if paper['year']:
                    console.print(f"   Year: {paper['year']}")
                console.print(f"   Source: {paper['source']}")
                if paper['doi']:
                    console.print(f"   DOI: {paper['doi']}")
                if paper['pdf_url']:
                    console.print(f"   PDF: {paper['pdf_url']}")
        
        # Show download results
        if download_pdfs and 'downloads' in results:
            downloads = results['downloads']
            console.print(f"\n[green]Downloads:[/green] {downloads['successful']} "
                         f"successful, {downloads['failed']} failed")
    
    finally:
        await service.shutdown()


async def main():
    """Run all demos."""
    console.print(Panel.fit(
        "[bold green]Paper Search Service Demo[/bold green]\n"
        "Demonstrates searching and downloading academic papers",
        border_style="green"
    ))
    
    try:
        # Run demos
        await demo_basic_search()
        console.print("\n" + "="*80 + "\n")
        
        await demo_search_and_download()
        console.print("\n" + "="*80 + "\n")
        
        await demo_search_by_doi()
        console.print("\n" + "="*80 + "\n")
        
        await demo_service_stats()
        console.print("\n" + "="*80 + "\n")
        
        # Interactive demo
        run_interactive = console.input(
            "\n[bold yellow]Run interactive demo? (y/n):[/bold yellow] "
        ).strip().lower()
        
        if run_interactive == 'y':
            console.print("\n" + "="*80 + "\n")
            await demo_interactive()
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
