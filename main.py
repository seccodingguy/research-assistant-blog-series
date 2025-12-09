# main.py
from agents.research_assistant_agent import ResearchAssistantAgent
from utils.logger import log
from utils.response_saver import ResponseSaver
from utils.response_formatter import terminal_formatter
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table
from pathlib import Path
import asyncio
import sys
from datetime import datetime
from services.literature_review_service import LiteratureReviewService
import json

console = Console()


def print_welcome():
    """Print welcome message"""
    welcome_text = """
    # PDF Agent - Intelligent Document Assistant
    
    Commands:
    - `search <query>` - Search your documents
    - `chat <message>` - Interactive chat with context
    - `process <path>` - Process a PDF file or folder
    - `sources <query>` - Get source documents for a query
    - `history` - View conversation history
    - `stats` - View agent statistics
    - `graph stats` - View knowledge graph statistics
    - `graph viz [query]` - Visualize knowledge graph (optional: filter by query)
    - `graph query <query>` - Query the knowledge graph
    - `graph merge [threshold]` - Merge similar entities in graph (default: 0.7)
    - `graph normalize` - Normalize relationship types in graph
    - `graph reclassify` - Reclassify all nodes with updated logic ⭐ NEW!
    - `graph reclassify-hybrid [--batch=N] [--dry-run]` - Hybrid LLM+pattern classification ⭐ PHASE 4!
    - `ontology stats` - Show ontology configuration statistics ⭐ NEW!
    - `ontology validate` - Validate ontology configuration ⭐ NEW!
    - `ontology reload` - Reload ontology from disk ⭐ NEW!
    - `ontology categories` - Show concept type categories ⭐ NEW!
    - `papers search <query>` - Search for academic papers online
    - `papers download <query>` - Search and download papers
    - `papers sources` - List available search engines
    - `papers stats` - View paper search statistics
    - `review list` - List literature reviews
    - `review create` - Create a new literature review
    - `review get <id>` - Get details of a review
    - `review export <id>` - Export review to CSV
    - `review resume <id>` - Resume processing for a review
    - `review delete <id>` - Delete a literature review
    - `workflow execute <prompt>` - Execute a complex research workflow ⭐ NEW!
    - `workflow status` - Check current workflow status ⭐ NEW!
    - `clear` - Clear conversation memory
    - `session start <name>` - Start new session
    - `session end` - End current session
    - `providers` - Show current AI providers
    - `set embedding <azure|ollama>` - Switch embedding provider
    - `set llm <poe|ollama>` - Switch LLM provider
    - `help` - Show this message
    - `exit` - Exit the application
    
    **Natural Language Paper Workflows** ⭐ NEW!
    Use natural language to search, download, and process papers:
    - "search and download papers about agent2agent protocols"
    - "search, download, and process papers on transformers"
    - "find and download papers about reinforcement learning"
    
    **Advanced Research Workflows** ⭐ NEW!
    Use natural language for complex multi-step research tasks:
    - "Propose 5 topics on AI agents, download 10 papers each, select best topic"
    - Doctoral dissertation planning workflows
    - Multi-stage literature reviews with topic selection
    
    **After search/chat responses, you'll be prompted to save in formats:**
    txt, md, json, html, csv (with optional auto-timestamping)
    """
    console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="green"))


def print_stats(stats: dict):
    """Print agent statistics"""
    console.print("\n[bold cyan]Agent Statistics:[/bold cyan]")
    console.print(f"User ID: {stats['user_id']}")
    console.print(f"Total Documents: {stats['index_stats'].get('total_documents', 0)}")
    console.print(f"Total Messages: {stats['memory_stats']['total_messages']}")
    console.print(f"Sessions: {stats['memory_stats']['sessions']}")
    console.print(f"Auto-watch Active: {stats['watch_active']}\n")


def prompt_save_response(response_data: dict, command_type: str):
    """
    Prompt user to save the response.
    
    Args:
        response_data: Response data dictionary from search/chat
        command_type: Type of command ("search" or "chat")
    """
    try:
        # Ask if user wants to save
        save_prompt = Prompt.ask(
            "\n[bold yellow]Save this response?[/bold yellow]",
            choices=["y", "n"],
            default="n"
        )
        
        if save_prompt.lower() != "y":
            return
        
        # Get file path
        console.print("\n[cyan]Enter save path (e.g., ./outputs/response.txt or ~/documents/result.json)[/cyan]")
        file_path = Prompt.ask("[bold]File path[/bold]").strip()
        
        if not file_path:
            console.print("[yellow]No path provided, save cancelled[/yellow]")
            return
        
        file_path = Path(file_path).expanduser()
        
        # Determine format from extension
        extension = file_path.suffix.lstrip('.').lower()
        
        if extension not in ResponseSaver.SUPPORTED_FORMATS:
            console.print(f"[yellow]Unsupported format: {extension}[/yellow]")
            console.print(f"[cyan]Supported formats: {', '.join(ResponseSaver.SUPPORTED_FORMATS)}[/cyan]")
            
            # Let user choose format
            format_choice = Prompt.ask(
                "[bold]Choose format[/bold]",
                choices=ResponseSaver.SUPPORTED_FORMATS,
                default="txt"
            )
            
            # Update file extension
            file_path = file_path.with_suffix(f".{format_choice}")
            console.print(f"[cyan]Saving as: {file_path}[/cyan]")
        
        # Ask about auto-naming
        auto_name = Prompt.ask(
            "[bold yellow]Add timestamp to filename?[/bold yellow]",
            choices=["y", "n"],
            default="n"
        )
        
        # Save the response
        success = ResponseSaver.save_response(
            response_data=response_data,
            file_path=file_path,
            auto_name=(auto_name.lower() == "y")
        )
        
        if success:
            # Get actual path (might have timestamp)
            if auto_name.lower() == "y":
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                actual_path = file_path.parent / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            else:
                actual_path = file_path
            
            console.print(f"[green]✓ Response saved successfully to:[/green]")
            console.print(f"[green]  {actual_path.absolute()}[/green]")
        else:
            console.print("[red]✗ Failed to save response[/red]")
    
    except Exception as e:
        console.print(f"[red]Error saving response: {e}[/red]")
        log.exception("Error in save prompt")


# ==================== Command Handlers ====================

def handle_paper_workflow(agent, intent):
    """Handle natural language paper search/download/process workflow"""
    search_query = intent["search_query"]
    actions = intent["actions"]
    extracted_max_results = intent.get("max_results")
    extracted_save_location = intent.get("save_location")
    
    console.print(f"\n[bold cyan]Detected Paper Workflow:[/bold cyan]")
    console.print(f"  Search Query: [yellow]{search_query}[/yellow]")
    console.print(f"  Actions: [yellow]{', '.join(actions)}[/yellow]")
    if extracted_max_results:
        console.print(f"  Max Results: [yellow]{extracted_max_results}[/yellow]")
    if extracted_save_location:
        console.print(f"  Save Location: [yellow]{extracted_save_location}[/yellow]")
    
    # Confirm with user
    confirm = Prompt.ask(
        "\n[bold yellow]Proceed with this workflow?[/bold yellow]",
        choices=["y", "n"],
        default="y"
    )
    
    if confirm.lower() != "y":
        console.print("[yellow]Workflow cancelled[/yellow]")
        return
    
    # Ask for max results if downloading and not specified in query
    max_results = extracted_max_results or 5
    if 'download' in actions and not extracted_max_results:
        max_input = Prompt.ask(
            "[bold]Maximum papers to download[/bold]",
            default="5"
        )
        try:
            max_results = int(max_input)
        except ValueError:
            console.print("[yellow]Invalid number, using default: 5[/yellow]")
            max_results = 5
    
    # Ask for save location if downloading and not specified in query
    save_location = extracted_save_location
    if 'download' in actions and not extracted_save_location:
        save_input = Prompt.ask(
            "[bold]Save location (leave empty for default)[/bold]",
            default=""
        )
        if save_input.strip():
            save_location = save_input.strip()
    
    # Execute workflow
    with console.status("[bold green]Executing workflow..."):
        result = asyncio.run(agent.execute_paper_workflow(
            search_query=search_query,
            actions=actions,
            max_results=max_results,
            save_location=save_location
        ))
    
    # Display results
    if not result["success"]:
        console.print(f"\n[red]Workflow failed: {result.get('error')}[/red]")
        return
    
    console.print("\n[bold green]✓ Workflow Completed![/bold green]")
    console.print(f"Actions: {', '.join(result['actions_completed'])}")
    
    # Show search results
    if result.get("search_results"):
        search_res = result["search_results"]
        console.print(f"\n[cyan]Search Results:[/cyan]")
        console.print(f"  Found: {search_res.get('total_results', 0)} papers")
    
    # Show download results
    if result.get("download_results"):
        dl_res = result["download_results"]
        console.print(f"\n[cyan]Download Results:[/cyan]")
        console.print(f"  Successful: {dl_res.get('successful', 0)}")
        console.print(f"  Failed: {dl_res.get('failed', 0)}")
        console.print(f"  Skipped: {dl_res.get('skipped', 0)}")
        
        if result["downloaded_files"]:
            console.print(f"\n[cyan]Downloaded Files:[/cyan]")
            for pdf_path_str in result["downloaded_files"][:10]:
                pdf_path = Path(pdf_path_str)
                console.print(f"  ✓ {pdf_path.name}")
    
    # Show process results
    if result.get("process_results"):
        proc_res = result["process_results"]
        console.print(f"\n[cyan]Processing Results:[/cyan]")
        console.print(f"  Processed: {proc_res.get('processed', 0)}/{proc_res.get('total_files', 0)}")
        console.print(f"  Failed: {proc_res.get('failed', 0)}")
    
    # Prompt to process if files were downloaded but not processed
    if result["downloaded_files"] and 'process' not in actions:
        console.print(f"\n[bold yellow]Downloaded {len(result['downloaded_files'])} PDF files[/bold yellow]")
        process_now = Prompt.ask(
            "[bold]Process these files into knowledge base now?[/bold]",
            choices=["y", "n"],
            default="y"
        )
        
        if process_now.lower() == "y":
            with console.status("[bold green]Processing PDFs..."):
                processed_count = 0
                failed_count = 0
                
                for pdf_path_str in result["downloaded_files"]:
                    pdf_path = Path(pdf_path_str)
                    if pdf_path.exists():
                        proc_result = agent.process_pdf(pdf_path)
                        if proc_result["success"]:
                            processed_count += 1
                            console.print(f"[green]✓ {pdf_path.name}[/green]")
                        else:
                            failed_count += 1
                            console.print(f"[red]✗ {pdf_path.name}[/red]")
            
            console.print(f"\n[green]Processed {processed_count} files successfully[/green]")
            if failed_count > 0:
                console.print(f"[yellow]Failed to process {failed_count} files[/yellow]")


def handle_search(agent, args):
    """Handle search command"""
    if not args:
        console.print("[red]Please provide a search query[/red]")
        return
    
    with console.status("[bold green]Searching..."):
        result = agent.search(args, mode="enhanced")
    
    # Format result for terminal display
    response_data = {
        "response": result.get("answer", ""),
        "sources": result.get("sources", []),
        "metadata": result.get("metadata", {})
    }
    terminal_formatter.format_response(response_data, format_type="terminal")
    
    prompt_save_response(result, "search")


def handle_chat(agent, args):
    """Handle chat command"""
    if not args:
        console.print("[red]Please provide a message[/red]")
        return
    
    with console.status("[bold green]Thinking..."):
        result = agent.chat(args)
    
    # Use the response formatter for better display
    terminal_formatter.format_response(result, format_type="terminal")
    
    prompt_save_response(result, "chat")


def handle_process(agent, args):
    """Handle process command"""
    if not args:
        console.print("[red]Please provide a file or folder path[/red]")
        return
    
    path = Path(args)
    
    if not path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        return
    
    with console.status("[bold green]Processing..."):
        if path.is_file():
            result = agent.process_pdf(path)
            if result["success"]:
                console.print(f"[green]✓ {result['message']}[/green]")
            else:
                console.print(f"[red]✗ {result['message']}[/red]")
        else:
            result = agent.process_folder(path)
            if result["success"]:
                console.print(f"[green]✓ Processed {result['documents_processed']} documents[/green]")
            else:
                console.print(f"[red]✗ Processing failed: {result.get('error')}[/red]")


def handle_sources(agent, args):
    """Handle sources command"""
    if not args:
        console.print("[red]Please provide a query[/red]")
        return
    
    sources = agent.get_sources(args)
    
    if sources:
        console.print("\n[bold cyan]Source Documents:[/bold cyan]")
        for i, source in enumerate(sources, 1):
            console.print(f"{i}. {source['file_name']} (relevance: {source['relevance_score']:.2f})")
    else:
        console.print("[yellow]No sources found[/yellow]")


def handle_history(agent):
    """Handle history command"""
    history = agent.get_conversation_history(limit=10)
    
    if history:
        console.print("\n[bold cyan]Conversation History:[/bold cyan]")
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            console.print(f"[{'blue' if role == 'user' else 'green'}]{role.upper()}:[/] {content}...")
    else:
        console.print("[yellow]No conversation history[/yellow]")


def handle_graph(agent, args):
    """Handle graph commands"""
    if not args:
        console.print(
            "[red]Please specify graph command: stats, viz, query, "
            "merge, or normalize[/red]"
        )
        return
    
    graph_cmd = args.split()[0]
    graph_args = ' '.join(args.split()[1:]) if len(args.split()) > 1 else None
    
    if graph_cmd == "stats":
        stats = agent.get_graph_stats()
        if stats:
            console.print("\n[bold cyan]Knowledge Graph Statistics:[/bold cyan]")
            console.print(f"Total Nodes: {stats.get('total_nodes', 0)}")
            console.print(f"Total Edges: {stats.get('total_edges', 0)}")
            console.print(f"Average Degree: {stats.get('avg_degree', 0):.2f}")
            console.print(f"Graph Density: {stats.get('density', 0):.4f}")
            console.print(f"Is Connected: {stats.get('is_connected', False)}")
            if stats.get('node_types'):
                console.print("\n[bold]Node Types:[/bold]")
                for node_type, count in stats['node_types'].items():
                    console.print(f"  {node_type}: {count}")
        else:
            console.print(
                "[yellow]Knowledge graph not available or is empty[/yellow]"
            )
    
    elif graph_cmd == "viz":
        output_path = Path("./outputs/graph_viz.json")
        result = agent.visualize_graph(output_path, max_nodes=100)
        
        if result["success"] and result["data"].get('nodes'):
            console.print(
                f"\n[green]✓ Graph visualization saved to: "
                f"{output_path}[/green]"
            )
            console.print(
                f"Nodes: {len(result['data']['nodes'])}, "
                f"Edges: {len(result['data']['edges'])}"
            )
            
            console.print("\n[bold cyan]Top Concepts:[/bold cyan]")
            sorted_nodes = sorted(
                result['data']['nodes'],
                key=lambda x: x['degree'],
                reverse=True
            )[:10]
            for i, node in enumerate(sorted_nodes, 1):
                console.print(
                    f"{i}. {node['label']} (connections: {node['degree']})"
                )
        else:
            error_msg = result.get(
                'error',
                'Graph is empty or visualization failed'
            )
            console.print(f"[yellow]{error_msg}[/yellow]")
    
    elif graph_cmd == "query":
        if not graph_args:
            console.print("[red]Please provide a query for the graph[/red]")
            return
        
        with console.status("[bold green]Querying knowledge graph..."):
            result = agent.query_graph(graph_args)
        
        if result["success"] and result.get('response'):
            console.print("\n[bold cyan]Graph Query Response:[/bold cyan]")
            console.print(Markdown(result['response']))
            
            if result.get('nodes'):
                console.print(
                    f"\n[bold]Related Nodes:[/bold] {len(result['nodes'])}"
                )
                for node in result['nodes'][:10]:
                    console.print(f"  • {node['label']}")
            
            if result.get('relationships'):
                console.print(
                    f"\n[bold]Relationships:[/bold] "
                    f"{len(result['relationships'])}"
                )
        else:
            error_msg = result.get('error', 'No graph results found')
            console.print(f"[yellow]{error_msg}[/yellow]")
    
    elif graph_cmd == "merge":
        # Parse threshold if provided
        threshold = 0.7
        if graph_args:
            try:
                threshold = float(graph_args)
                if not 0 < threshold < 1:
                    console.print(
                        "[red]Threshold must be between 0 and 1[/red]"
                    )
                    return
            except ValueError:
                console.print(
                    "[red]Invalid threshold value. Using default: 0.7[/red]"
                )
        
        with console.status(
            f"[bold green]Merging similar entities "
            f"(threshold: {threshold})..."
        ):
            result = agent.merge_graph_entities(threshold)
        
        if result["success"]:
            merged = result.get('merged_count', 0)
            console.print(
                f"\n[green]✓ Successfully merged {merged} "
                f"similar entities[/green]"
            )
            if result.get('merged_pairs'):
                console.print("\n[bold cyan]Merged Entities:[/bold cyan]")
                for pair in result['merged_pairs'][:10]:
                    console.print(f"  • {pair[0]} ← {pair[1]}")
                if len(result['merged_pairs']) > 10:
                    remaining = len(result['merged_pairs']) - 10
                    console.print(f"  ... and {remaining} more")
        else:
            error_msg = result.get('error', 'Merge operation failed')
            console.print(f"[red]✗ {error_msg}[/red]")
    
    elif graph_cmd == "normalize":
        with console.status(
            "[bold green]Normalizing relationship types..."
        ):
            result = agent.normalize_graph_relationships()
        
        if result["success"]:
            normalized = result.get('normalized_count', 0)
            console.print(
                f"\n[green]✓ Successfully normalized {normalized} "
                f"relationships[/green]"
            )
            if result.get('mapping_summary'):
                console.print(
                    "\n[bold cyan]Normalization Summary:[/bold cyan]"
                )
                for old_type, new_type in sorted(
                    result['mapping_summary'].items()
                )[:15]:
                    console.print(f"  {old_type} → {new_type}")
                if len(result['mapping_summary']) > 15:
                    remaining = len(result['mapping_summary']) - 15
                    console.print(f"  ... and {remaining} more mappings")
        else:
            error_msg = result.get('error', 'Normalization failed')
            console.print(f"[red]✗ {error_msg}[/red]")
    
    elif graph_cmd == "reclassify":
        with console.status(
            "[bold green]Reclassifying all nodes with updated logic..."
        ):
            result = agent.reclassify_graph_nodes()
        
        if result["success"]:
            reclassified = result.get('reclassified', 0)
            total = result.get('total_nodes', 0)
            console.print(
                f"\n[green]✓ Reclassified {reclassified} out of "
                f"{total} nodes[/green]"
            )
            
            if result.get('changes'):
                console.print("\n[bold cyan]Classification Changes:[/bold cyan]")
                sorted_changes = sorted(
                    result['changes'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:15]
                for change, count in sorted_changes:
                    console.print(f"  {change}: {count} nodes")
                if len(result['changes']) > 15:
                    remaining = len(result['changes']) - 15
                    console.print(f"  ... and {remaining} more changes")
            
            # Show updated stats
            console.print("\n[bold cyan]Running graph stats...[/bold cyan]")
            stats = agent.get_graph_stats()
            if stats and stats.get('node_types'):
                console.print("\n[bold]Updated Node Type Distribution:[/bold]")
                sorted_types = sorted(
                    stats['node_types'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:15]
                for node_type, count in sorted_types:
                    console.print(f"  {node_type}: {count}")
        else:
            error_msg = result.get('error', 'Reclassification failed')
            console.print(f"[red]✗ {error_msg}[/red]")
    
    elif graph_cmd == "reclassify-hybrid":
        # Parse options
        batch_size = 100
        dry_run = False
        
        if len(args) > 1:
            for arg in args[1:]:
                if arg.startswith("--batch="):
                    try:
                        batch_size = int(arg.split("=")[1])
                    except ValueError:
                        console.print(
                            f"[yellow]Invalid batch size: {arg}[/yellow]"
                        )
                elif arg == "--dry-run":
                    dry_run = True
        
        action = "Would reclassify" if dry_run else "Reclassifying"
        with console.status(
            f"[bold green]{action} unknown nodes with hybrid approach..."
        ):
            result = agent.reclassify_hybrid(
                batch_size=batch_size,
                dry_run=dry_run
            )
        
        if result.get("success"):
            total = result.get('total_processed', 0)
            non_concepts = result.get('non_concepts', 0)
            llm_candidates = result.get('llm_candidates', 0)
            llm_classified = result.get('llm_classified', 0)
            still_unknown = result.get('still_unknown', 0)
            
            console.print(f"\n[green]✓ Hybrid classification complete[/green]")
            console.print(f"\nTotal unknown nodes processed: {total}")
            console.print(
                f"Non-concepts filtered: {non_concepts} "
                f"({100*non_concepts/total:.1f}%)" if total > 0 else ""
            )
            console.print(
                f"LLM candidates: {llm_candidates} "
                f"({100*llm_candidates/total:.1f}%)" if total > 0 else ""
            )
            
            if not dry_run:
                console.print(
                    f"LLM classified: {llm_classified} "
                    f"({100*llm_classified/llm_candidates:.1f}%)"
                    if llm_candidates > 0 else ""
                )
                console.print(
                    f"Still unknown: {still_unknown} "
                    f"({100*still_unknown/total:.1f}%)" if total > 0 else ""
                )
                
                # Show updated stats
                console.print(
                    "\n[bold cyan]Running graph stats...[/bold cyan]"
                )
                stats = agent.get_graph_stats()
                if stats and stats.get('node_types'):
                    unknown_count = stats['node_types'].get('unknown', 0)
                    total_nodes = sum(stats['node_types'].values())
                    classified = total_nodes - unknown_count
                    console.print(
                        f"\n[bold]Classification Rate:[/bold] "
                        f"{classified}/{total_nodes} "
                        f"({100*classified/total_nodes:.1f}%)"
                    )
        else:
            error_msg = result.get('error', 'Hybrid classification failed')
            console.print(f"[red]✗ {error_msg}[/red]")
    
    else:
        console.print(
            "[red]Invalid graph command. "
            "Use: stats, viz, query, merge, normalize, reclassify, "
            "or reclassify-hybrid[/red]"
        )


def handle_ontology(agent, args):
    """Handle ontology management commands"""
    if not args:
        console.print(
            "[red]Please specify ontology command: stats, validate, "
            "reload, or categories[/red]"
        )
        return
    
    ontology_cmd = args.split()[0]
    
    if ontology_cmd == "stats":
        # Get ontology statistics
        from config.ontology_loader import get_ontology_loader
        ontology = get_ontology_loader()
        stats = ontology.get_statistics()
        
        console.print("\n[bold cyan]Ontology Configuration Statistics:[/bold cyan]")
        console.print(f"Relationship Types: {stats['relationship_types']}")
        console.print(f"Concept Types: {stats['concept_types']}")
        console.print(f"Keyword Mappings: {stats['keyword_mappings']}")
        console.print(f"Classification Patterns: {stats['classification_patterns']}")
        console.print(f"Relationship Mappings: {stats['relationship_mappings']}")
        console.print(f"Categories: {stats['categories']}")
        
        console.print("\n[dim]Configuration file: config/graph_ontology.yaml[/dim]")
    
    elif ontology_cmd == "validate":
        # Validate ontology configuration
        from config.ontology_loader import get_ontology_loader
        ontology = get_ontology_loader()
        is_valid, errors = ontology.validate_config()
        
        if is_valid:
            console.print("[green]✓ Ontology configuration is valid![/green]")
        else:
            console.print("[red]✗ Validation errors found:[/red]")
            for error in errors:
                console.print(f"  • {error}")
    
    elif ontology_cmd == "reload":
        # Reload ontology and optionally reclassify graph
        with console.status("[bold green]Reloading ontology configuration..."):
            result = agent.reload_graph_ontology()
        
        if result.get("success"):
            stats = result.get('stats', {})
            console.print("[green]✓ Ontology configuration reloaded![/green]")
            console.print(f"\nConcept Types: {stats.get('concept_types', 0)}")
            console.print(f"Keywords: {stats.get('keyword_mappings', 0)}")
            console.print(f"Patterns: {stats.get('classification_patterns', 0)}")
            
            # Ask if user wants to reclassify existing nodes
            console.print("\n[yellow]To apply changes to existing nodes, run:[/yellow]")
            console.print("[cyan]  graph reclassify[/cyan]")
        else:
            error_msg = result.get('error', 'Reload failed')
            console.print(f"[red]✗ {error_msg}[/red]")
    
    elif ontology_cmd == "categories":
        # Show concept categories
        from config.ontology_loader import get_ontology_loader
        ontology = get_ontology_loader()
        categories = ontology.get_concept_categories()
        
        console.print("\n[bold cyan]Concept Type Categories:[/bold cyan]\n")
        
        for category, concepts in sorted(categories.items()):
            console.print(f"[bold]{category.upper()}:[/bold] ({len(concepts)} types)")
            # Show first 5 concepts
            for concept in sorted(concepts)[:5]:
                console.print(f"  • {concept}")
            if len(concepts) > 5:
                console.print(f"  ... and {len(concepts) - 5} more")
            console.print()
    
    else:
        console.print(
            "[red]Invalid ontology command. "
            "Use: stats, validate, reload, or categories[/red]"
        )


def handle_clear(agent):
    """Handle clear memory command"""
    result = agent.clear_memory()
    if result["success"]:
        console.print("[green]✓ Memory cleared[/green]")
    else:
        console.print(f"[red]✗ Failed to clear memory: {result.get('error')}[/red]")


def handle_session(agent, args):
    """Handle session commands"""
    if args.startswith("start"):
        session_name = args.split(maxsplit=1)[1] if len(args.split()) > 1 else None
        result = agent.start_session(session_name)
        if result["success"]:
            console.print(f"[green]✓ Started session: {result['session_name']}[/green]")
        else:
            console.print(f"[red]✗ Failed to start session: {result.get('error')}[/red]")
    
    elif args == "end":
        result = agent.end_session()
        if result["success"]:
            console.print("[green]✓ Session ended[/green]")
        else:
            console.print(f"[red]✗ Failed to end session: {result.get('error')}[/red]")
    
    else:
        console.print("[red]Invalid session command. Use 'session start <name>' or 'session end'[/red]")


def handle_providers(agent):
    """Handle providers command"""
    providers = agent.get_providers()
    console.print("\n[bold cyan]Current AI Providers:[/bold cyan]")
    console.print(f"Embedding: {providers['embedding']}")
    console.print(f"LLM: {providers['llm']}")
    console.print("\n[bold yellow]Note:[/bold yellow] Provider changes require restart to take effect")


def handle_papers(agent, args):
    """Handle paper search commands"""
    if not args:
        console.print("[red]Please specify papers command: search, download, sources, or stats[/red]")
        return
    
    paper_parts = args.split(maxsplit=1)
    paper_cmd = paper_parts[0].lower()
    paper_args = paper_parts[1] if len(paper_parts) > 1 else ""
    
    if paper_cmd == "search":
        if not paper_args:
            console.print("[red]Please provide a search query[/red]")
            return
        
        with console.status("[bold green]Searching academic databases..."):
            result = asyncio.run(agent.search_papers(
                query=paper_args,
                max_results_per_source=10,
                download_pdfs=False
            ))
        
        if not result["success"]:
            console.print(f"[red]Search failed: {result.get('error')}[/red]")
            return
        
        console.print(f"\n[green]Found {result['total_results']} papers from {len(result['sources_searched'])} sources[/green]")
        
        if result['results']:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#", width=4)
            table.add_column("Title", style="cyan", width=40)
            table.add_column("Authors", style="green", width=25)
            table.add_column("Year", style="yellow", width=6)
            table.add_column("Source", style="blue", width=12)
            
            for i, paper in enumerate(result['results'][:20], 1):
                authors = ", ".join(paper['authors'][:2]) if paper['authors'] else "N/A"
                if paper['authors'] and len(paper['authors']) > 2:
                    authors += " et al."
                
                title = paper['title'][:40] + "..." if len(paper['title']) > 40 else paper['title']
                
                table.add_row(
                    str(i),
                    title,
                    authors[:25],
                    str(paper['year']) if paper['year'] else "N/A",
                    paper['source']
                )
            
            console.print("\n", table)
    
    elif paper_cmd == "download":
        if not paper_args:
            console.print("[red]Please provide a search query[/red]")
            return
        
        with console.status("[bold green]Searching and downloading papers..."):
            result = asyncio.run(agent.search_and_download_papers(
                query=paper_args,
                max_results=5
            ))
        
        if not result["success"]:
            console.print(f"[red]Download failed: {result.get('error')}[/red]")
            return
        
        console.print(f"\n[green]Search Results:[/green]")
        console.print(f"  Papers found: {result['total_results']}")
        
        if 'downloads' in result:
            downloads = result['downloads']
            console.print(f"\n[green]Download Results:[/green]")
            console.print(f"  Successful: {downloads['successful']}")
            console.print(f"  Failed: {downloads['failed']}")
            console.print(f"  Skipped: {downloads['skipped']}")
            
            successful = [r for r in downloads['results'] if r['success'] and r.get('reason') != 'already_exists']
            if successful:
                console.print("\n[cyan]Downloaded PDFs:[/cyan]")
                for r in successful[:10]:
                    console.print(f"  ✓ {r['title'][:60]}")
    
    elif paper_cmd == "sources":
        sources = agent.get_paper_sources()
        console.print("\n[bold cyan]Available Search Engines:[/bold cyan]")
        for source in sources:
            console.print(f"  • {source}")
    
    elif paper_cmd == "stats":
        stats = agent.get_paper_stats()
        console.print("\n[bold cyan]Paper Search Statistics:[/bold cyan]")
        console.print(f"  Total searches: {stats['total_searches']}")
        console.print(f"  Total results: {stats['total_results']}")
        console.print(f"  Downloads: {stats['total_downloads']}")
        console.print(f"  Successful: {stats['successful_downloads']}")
        console.print(f"  Failed: {stats['failed_downloads']}")
        
        if stats.get('searches_by_engine'):
            console.print("\n[bold cyan]By Engine:[/bold cyan]")
            for engine, count in stats['searches_by_engine'].items():
                results = stats['results_by_engine'].get(engine, 0)
                console.print(f"  {engine}: {count} searches, {results} results")
    
    else:
        console.print("[red]Invalid papers command. Use: search, download, sources, or stats[/red]")


def handle_set(agent, args):
    """Handle set provider commands"""
    if not args:
        console.print("[red]Please specify what to set. Use 'set embedding <provider>' or 'set llm <provider>'[/red]")
        return
    
    set_parts = args.split(maxsplit=1)
    if len(set_parts) != 2:
        console.print("[red]Invalid set command. Use 'set embedding <azure|ollama>' or 'set llm <poe|ollama>'[/red]")
        return
    
    set_type, set_value = set_parts
    
    if set_type == "embedding":
        result = agent.set_embedding_provider(set_value)
        if result["success"]:
            console.print(f"[green]✓ {result['message']}[/green]")
        else:
            console.print(f"[red]✗ {result['error']}[/red]")
    
    elif set_type == "llm":
        result = agent.set_llm_provider(set_value)
        if result["success"]:
            console.print(f"[green]✓ {result['message']}[/green]")
        else:
            console.print(f"[red]✗ {result['error']}[/red]")
    
    else:
        console.print("[red]Invalid set type. Use 'embedding' or 'llm'[/red]")


def handle_literature_review(agent, args):
    """Handle literature review commands"""
    if not args:
        console.print(
            "[red]Please specify literature review command: list, create, get, export, resume, or delete[/red]"
        )
        return
    
    parts = args.split(maxsplit=1)
    cmd = parts[0].lower()
    cmd_args = parts[1] if len(parts) > 1 else ""
    
    service = LiteratureReviewService()
    
    if cmd == "list":
        reviews = service.list_literature_reviews()
        if reviews:
            console.print("\n[bold cyan]Literature Reviews:[/bold cyan]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("ID", style="dim", width=36)
            table.add_column("Title", style="cyan", width=40)
            table.add_column("Docs", style="green", width=6)
            table.add_column("Topics", style="yellow", width=6)
            table.add_column("Created", style="blue", width=20)
            
            for review in reviews:
                summary = review.get('summary', {})
                table.add_row(
                    review['review_id'],
                    review['title'],
                    str(summary.get('item_count', 0)),
                    str(summary.get('topic_count', 0)),
                    review['created_at'][:19]
                )
            console.print(table)
        else:
            console.print("[yellow]No literature reviews found[/yellow]")
            
    elif cmd == "create":
        # Interactive creation
        console.print("\n[bold cyan]Create Literature Review[/bold cyan]")
        title = Prompt.ask("[bold]Title[/bold]")
        description = Prompt.ask("[bold]Description (optional)[/bold]", default="")
        
        # Get available documents
        db = service.db
        docs = db.list_documents(limit=100)
        
        if not docs:
            console.print("[red]No documents available to create a review[/red]")
            return
            
        console.print(f"\n[cyan]Available Documents ({len(docs)}):[/cyan]")
        for i, doc in enumerate(docs, 1):
            console.print(f"{i}. {doc.get('title', 'Untitled')} ({doc['document_id']})")
            
        doc_indices = Prompt.ask("\n[bold]Select documents (comma-separated numbers, 'all', or 'dir:path/to/folder')[/bold]")
        
        try:
            if doc_indices.lower() == 'all':
                selected_ids = [doc['document_id'] for doc in docs]
            elif doc_indices.lower().startswith('dir:'):
                target_dir_str = doc_indices[4:].strip()
                target_dir = Path(target_dir_str).resolve()
                
                console.print(f"[cyan]Searching for documents in: {target_dir}[/cyan]")
                
                # Fetch all documents to filter by path
                all_docs = db.list_documents(limit=None)
                selected_ids = []
                
                for doc in all_docs:
                    try:
                        doc_path = Path(doc['file_path']).resolve()
                        # Check if file is directly in the folder
                        if doc_path.parent == target_dir:
                            selected_ids.append(doc['document_id'])
                    except Exception:
                        continue
                        
                if not selected_ids:
                    console.print(f"[yellow]No documents found in {target_dir}[/yellow]")
                    return
                
                console.print(f"[green]Found {len(selected_ids)} documents in directory[/green]")
                
            else:
                indices = [int(idx.strip()) - 1 for idx in doc_indices.split(',')]
                selected_ids = [docs[i]['document_id'] for i in indices if 0 <= i < len(docs)]
        except (ValueError, IndexError):
            console.print("[red]Invalid selection[/red]")
            return
            
        if not selected_ids:
            console.print("[red]No documents selected[/red]")
            return
            
        auto_extract = Prompt.ask(
            "[bold]Auto-extract details using AI?[/bold]",
            choices=["y", "n"],
            default="y"
        ) == "y"
        
        with console.status("[bold green]Creating literature review..."):
            try:
                result = asyncio.run(service.create_literature_review(
                    title=title,
                    document_ids=selected_ids,
                    description=description,
                    created_by="cli_user",
                    auto_extract=auto_extract
                ))
                console.print(f"\n[green]✓ Created review: {result['review_id']}[/green]")
                console.print(f"  Documents: {result['document_count']}")
                console.print(f"  Topics: {result['topic_count']}")
                if auto_extract:
                    stats = result['extraction_stats']
                    console.print(f"  Extraction: {stats['successful']} successful, {stats['failed']} failed")
            except Exception as e:
                console.print(f"[red]Error creating review: {e}[/red]")

    elif cmd == "get":
        if not cmd_args:
            console.print("[red]Please provide a review ID[/red]")
            return
            
        try:
            result = service.get_literature_review(cmd_args, include_items=True)
            review = result['review']
            items = result.get('items', [])
            
            console.print(f"\n[bold cyan]{review['title']}[/bold cyan]")
            console.print(f"[dim]{review['review_id']}[/dim]")
            if review.get('description'):
                console.print(f"\n{review['description']}")
                
            console.print(f"\n[bold]Documents ({len(items)}):[/bold]")
            for item in items:
                status = "[green]✓[/green]" if item.get('research_question') else "[yellow]pending[/yellow]"
                console.print(f"  {status} {item.get('title', 'Unknown Title')}")
                
            if result.get('topic_groups'):
                console.print("\n[bold]Topics:[/bold]")
                for topic, doc_ids in result['topic_groups'].items():
                    console.print(f"  • {topic}: {len(doc_ids)} papers")
                    
        except Exception as e:
            console.print(f"[red]Error getting review: {e}[/red]")

    elif cmd == "export":
        if not cmd_args:
            console.print("[red]Please provide a review ID[/red]")
            return
            
        try:
            csv_content = service.export_to_csv(cmd_args)
            
            # Save to file
            filename = f"review_{cmd_args[:8]}.csv"
            output_path = Path("outputs") / filename
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, "w") as f:
                f.write(csv_content)
                
            console.print(f"[green]✓ Exported to {output_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error exporting review: {e}[/red]")

    elif cmd == "resume":
        if not cmd_args:
            console.print("[red]Please provide a review ID[/red]")
            return
            
        with console.status("[bold green]Resuming processing..."):
            try:
                result = asyncio.run(service.process_missing_items(cmd_args))
                console.print(f"\n[green]✓ {result['message']}[/green]")
                if result.get('stats'):
                    stats = result['stats']
                    console.print(f"  Processed: {stats['processed']}")
                    console.print(f"  Successful: {stats['successful']}")
                    console.print(f"  Failed: {stats['failed']}")
            except Exception as e:
                console.print(f"[red]Error resuming review: {e}[/red]")

    elif cmd == "delete":
        if not cmd_args:
            console.print("[red]Please provide a review ID[/red]")
            return
            
        confirm = Prompt.ask(
            f"[bold red]Are you sure you want to delete review {cmd_args}?[/bold red]",
            choices=["y", "n"],
            default="n"
        )
        
        if confirm == "y":
            result = service.soft_delete_literature_review(cmd_args, "cli_user")
            if result['success']:
                console.print("[green]✓ Review deleted[/green]")
            else:
                console.print(f"[red]✗ {result['message']}[/red]")

    else:
        console.print(
            "[red]Invalid command. Use: list, create, get, export, resume, or delete[/red]"
        )


def handle_literature_review_workflow(agent, intent):
    """Handle natural language literature review creation workflow"""
    title = intent["title"]
    topic = intent["topic"]
    select_all = intent["select_all"]
    
    console.print(f"\n[bold cyan]Detected Literature Review Creation:[/bold cyan]")
    console.print(f"  Title: [yellow]{title}[/yellow]")
    if topic:
        console.print(f"  Topic Filter: [yellow]{topic}[/yellow]")
    console.print(f"  Select All: [yellow]{select_all}[/yellow]")
    
    # Confirm with user
    confirm = Prompt.ask(
        "\n[bold yellow]Proceed with creation?[/bold yellow]",
        choices=["y", "n"],
        default="y"
    )
    
    if confirm.lower() != "y":
        console.print("[yellow]Creation cancelled[/yellow]")
        return
    
    service = LiteratureReviewService()
    db = service.db
    
    # Get documents
    docs = db.list_documents(limit=1000)
    
    if not docs:
        console.print("[red]No documents available[/red]")
        return
        
    selected_ids = []
    
    if select_all and not topic:
        # Select all documents
        selected_ids = [doc['document_id'] for doc in docs]
        console.print(f"[cyan]Selected all {len(selected_ids)} documents[/cyan]")
        
    elif topic:
        # Filter documents by topic (simple text search in title/abstract/keywords)
        console.print(f"[cyan]Filtering documents for '{topic}'...[/cyan]")
        topic_lower = topic.lower()
        for doc in docs:
            # Check title
            if doc.get('title') and topic_lower in doc['title'].lower():
                selected_ids.append(doc['document_id'])
                continue
                
            # Check abstract
            if doc.get('abstract') and topic_lower in doc['abstract'].lower():
                selected_ids.append(doc['document_id'])
                continue
                
            # Check keywords
            if doc.get('keywords'):
                try:
                    keywords = json.loads(doc['keywords']) if isinstance(doc['keywords'], str) else doc['keywords']
                    if any(topic_lower in k.lower() for k in keywords):
                        selected_ids.append(doc['document_id'])
                        continue
                except:
                    pass
        
        console.print(f"[cyan]Found {len(selected_ids)} matching documents[/cyan]")
    
    else:
        # Fallback to manual selection if intent was unclear
        console.print("[yellow]Could not determine document selection criteria. Switching to manual selection.[/yellow]")
        # ... (could call handle_literature_review(agent, "create") here but let's just exit for now)
        return

    if not selected_ids:
        console.print("[red]No documents matched your criteria[/red]")
        return

    with console.status("[bold green]Creating literature review..."):
        try:
            result = asyncio.run(service.create_literature_review(
                title=title,
                document_ids=selected_ids,
                description=f"Auto-generated review on {topic}" if topic else "Auto-generated review",
                created_by="cli_user",
                auto_extract=True
            ))
            console.print(f"\n[green]✓ Created review: {result['review_id']}[/green]")
            console.print(f"  Documents: {result['document_count']}")
            console.print(f"  Topics: {result['topic_count']}")
            stats = result['extraction_stats']
            console.print(f"  Extraction: {stats['successful']} successful, {stats['failed']} failed")
        except Exception as e:
            console.print(f"[red]Error creating review: {e}[/red]")


def handle_workflow(agent, args):
    """Handle workflow commands"""
    if not args:
        console.print("[red]Please specify workflow command: execute or status[/red]")
        return
    
    parts = args.split(maxsplit=1)
    workflow_cmd = parts[0].lower()
    workflow_args = parts[1] if len(parts) > 1 else ""
    
    if workflow_cmd == "execute":
        if not workflow_args:
            console.print("[red]Please provide a workflow prompt[/red]")
            return
        
        console.print("\n[bold cyan]Executing Research Workflow[/bold cyan]")
        console.print(f"[dim]Prompt: {workflow_args[:100]}...[/dim]\n")
        
        # Setup progress callback
        def progress_callback(event: str, data: dict):
            if event == "workflow_parsed":
                console.print(f"[cyan]📋 Workflow Type: {data['type']}[/cyan]")
                console.print(f"[cyan]📊 Steps: {data['step_count']} ({data.get('estimated_duration', 'unknown')})[/cyan]\n")
            
            elif event == "step_started":
                console.print(f"[yellow]▶ {data['description']}[/yellow]")
            
            elif event == "step_completed":
                summary = data.get('result_summary', 'Completed')
                console.print(f"[green]✓ {summary}[/green]")
            
            elif event == "step_failed":
                console.print(f"[red]✗ Error: {data.get('error', 'Unknown')}[/red]")
            
            elif event == "downloading_for_topic":
                console.print(f"  [dim]Downloading for topic {data['index']}/{data['total']}: {data['topic'][:50]}[/dim]")
            
            elif event == "workflow_completed":
                console.print(f"\n[bold green]✓ Workflow Completed![/bold green]")
                console.print(f"[green]Successful: {data['successful_steps']}/{data['total_steps']}[/green]")
        
        # Execute workflow
        try:
            result = asyncio.run(
                agent.execute_research_workflow(
                    workflow_args,
                    progress_callback=progress_callback
                )
            )
            
            if result.get("success"):
                console.print("\n[bold cyan]Generating Reports...[/bold cyan]")
                
                # Generate formatted reports
                from utils.academic_formatter import WorkflowReportGenerator
                from pathlib import Path
                
                output_dir = Path("outputs") / f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                report_gen = WorkflowReportGenerator(result)
                generated_files = report_gen.generate_complete_report(output_dir, format_type="markdown")
                
                console.print(f"\n[green]✓ Reports saved to: {output_dir}[/green]")
                console.print("\n[cyan]Generated Files:[/cyan]")
                for component, filepath in generated_files.items():
                    console.print(f"  • {component}: {filepath.name}")
                
                # Ask if user wants to save summary
                save_prompt = Prompt.ask(
                    "\n[bold yellow]View workflow summary?[/bold yellow]",
                    choices=["y", "n"],
                    default="y"
                )
                
                if save_prompt.lower() == "y":
                    summary_file = generated_files.get("summary")
                    if summary_file and summary_file.exists():
                        with open(summary_file, 'r') as f:
                            console.print(Markdown(f.read()))
            else:
                console.print(f"\n[red]✗ Workflow failed: {result.get('error')}[/red]")
                
        except Exception as e:
            console.print(f"[red]Error executing workflow: {e}[/red]")
            log.exception("Error in workflow execution")
    
    elif workflow_cmd == "status":
        progress = agent.get_workflow_progress()
        
        if progress.get("status") == "no_active_workflow":
            console.print("[yellow]No active workflow[/yellow]")
        else:
            console.print("\n[bold cyan]Workflow Status:[/bold cyan]")
            console.print(f"Status: {progress['status']}")
            console.print(f"Progress: {progress['percent_complete']:.1f}%")
            console.print(f"Completed: {progress['completed']}/{progress['total_steps']}")
            console.print(f"Failed: {progress['failed']}")
            
            if progress.get('current_step'):
                console.print(f"\nCurrent Step: {progress['current_step']}")
    
    else:
        console.print("[red]Invalid workflow command. Use: execute or status[/red]")


def main():
    """Main application loop"""
    
    console.print("[bold green]Initializing Research Assistant...[/bold green]")
    
    try:
        # Initialize agent
        agent = ResearchAssistantAgent(user_id="default", auto_watch=True)
        
        # Print welcome
        print_welcome()
        
        # Print initial stats
        print_stats(agent.get_stats())
        
        # Command dispatch dictionary
        commands = {
            "help": lambda _, __: print_welcome(),
            "search": handle_search,
            "chat": handle_chat,
            "process": handle_process,
            "sources": handle_sources,
            "history": lambda agent, _: handle_history(agent),
            "stats": lambda agent, _: print_stats(agent.get_stats()),
            "graph": handle_graph,
            "ontology": handle_ontology,
            "clear": lambda agent, _: handle_clear(agent),
            "session": handle_session,
            "providers": lambda agent, _: handle_providers(agent),
            "papers": handle_papers,
            "set": handle_set,
            "review": lambda agent, args: handle_literature_review(agent, args),
            "workflow": handle_workflow
        }

        # Main loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
                
                if not user_input:
                    continue
                
                # Check for complex research workflow FIRST (highest priority)
                workflow_intent = agent.detect_research_workflow_intent(user_input)
                if workflow_intent["is_research_workflow"] and workflow_intent["confidence"] > 0.4:
                    console.print(f"\n[bold cyan]Detected {workflow_intent['workflow_type']} workflow[/bold cyan]")
                    confirm = Prompt.ask("[bold yellow]Execute this workflow?[/bold yellow]", choices=["y", "n"], default="y")
                    
                    if confirm.lower() == "y":
                        handle_workflow(agent, f"execute {user_input}")
                        continue
                
                # Check for natural language paper workflow
                intent = agent.detect_paper_workflow_intent(user_input)
                if intent["is_paper_workflow"]:
                    if len(user_input.split()) > 15 and 'write' in user_input.lower():
                        pass
                    else:
                        handle_paper_workflow(agent, intent)
                        continue
                
                # Check for natural language literature review workflow
                intent = agent.detect_literature_review_intent(user_input)
                if intent["is_literature_review"]:
                    handle_literature_review_workflow(agent, intent)
                    continue
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ["exit", "quit"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if command in commands:
                    commands[command](agent, args)
                else:
                    console.print(f"[red]Unknown command: {command}. Type 'help' for available commands.[/red]")
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue
            
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                log.exception("Error in main loop")
    
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        log.exception("Fatal error")
        sys.exit(1)
    
    finally:
        # Cleanup
        if 'agent' in locals():
            agent.shutdown()


if __name__ == "__main__":
    main()