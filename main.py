# main.py
from agents.pdf_agent import PDFAgent
from utils.logger import log
from utils.response_saver import ResponseSaver
from config import settings
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from pathlib import Path
import sys

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
    - `clear` - Clear conversation memory
    - `session start <name>` - Start new session
    - `session end` - End current session
    - `providers` - Show current AI providers
    - `set embedding <azure|ollama>` - Switch embedding provider
    - `set llm <poe|ollama>` - Switch LLM provider
    - `help` - Show this message
    - `exit` - Exit the application
    
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


def main():
    """Main application loop"""
    
    console.print("[bold green]Initializing PDF Agent...[/bold green]")
    
    try:
        # Initialize agent
        agent = PDFAgent(user_id="default", auto_watch=True)
        
        # Print welcome
        print_welcome()
        
        # Print initial stats
        print_stats(agent.get_stats())
        
        # Main loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command == "exit" or command == "quit":
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                elif command == "help":
                    print_welcome()
                
                elif command == "search":
                    if not args:
                        console.print("[red]Please provide a search query[/red]")
                        continue
                    
                    with console.status("[bold green]Searching..."):
                        result = agent.search(args, mode="enhanced")
                    
                    console.print(f"\n[bold cyan]Answer:[/bold cyan]")
                    console.print(Panel(result["answer"], border_style="blue"))
                    
                    if result.get("sources"):
                        console.print("\n[bold cyan]Sources:[/bold cyan]")
                        for source in result["sources"]:
                            console.print(f"  • {source['file_name']} (relevance: {source['relevance_score']:.2f})")
                    
                    # Prompt to save response
                    prompt_save_response(result, "search")
                
                elif command == "chat":
                    if not args:
                        console.print("[red]Please provide a message[/red]")
                        continue
                    
                    with console.status("[bold green]Thinking..."):
                        response = agent.chat(args)
                    
                    console.print(f"\n[bold green]Assistant:[/bold green]")
                    console.print(Panel(response, border_style="green"))
                    
                    # Build response data for saving
                    # Get the full result from search_engine for metadata
                    with console.status("[bold green]Preparing response data..."):
                        result = agent.search(args, mode="enhanced", save_to_memory=False)
                    
                    # Prompt to save response
                    prompt_save_response(result, "chat")
                
                elif command == "process":
                    if not args:
                        console.print("[red]Please provide a file or folder path[/red]")
                        continue
                    
                    path = Path(args)
                    
                    if not path.exists():
                        console.print(f"[red]Path not found: {path}[/red]")
                        continue
                    
                    with console.status("[bold green]Processing..."):
                        if path.is_file():
                            success = agent.process_pdf(path)
                            if success:
                                console.print(f"[green]✓ Successfully processed: {path.name}[/green]")
                            else:
                                console.print(f"[red]✗ Failed to process: {path.name}[/red]")
                        else:
                            result = agent.process_folder(path)
                            if result["success"]:
                                console.print(f"[green]✓ Processed {result['documents_processed']} documents[/green]")
                            else:
                                console.print(f"[red]✗ Processing failed: {result.get('error')}[/red]")
                
                elif command == "sources":
                    if not args:
                        console.print("[red]Please provide a query[/red]")
                        continue
                    
                    sources = agent.get_sources(args)
                    
                    if sources:
                        console.print("\n[bold cyan]Source Documents:[/bold cyan]")
                        for i, source in enumerate(sources, 1):
                            console.print(f"{i}. {source['file_name']} (relevance: {source['relevance_score']:.2f})")
                    else:
                        console.print("[yellow]No sources found[/yellow]")
                
                elif command == "history":
                    history = agent.get_conversation_history(limit=10)
                    
                    if history:
                        console.print("\n[bold cyan]Conversation History:[/bold cyan]")
                        for msg in history:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")[:200]
                            console.print(f"[{'blue' if role == 'user' else 'green'}]{role.upper()}:[/] {content}...")
                    else:
                        console.print("[yellow]No conversation history[/yellow]")
                
                elif command == "stats":
                    print_stats(agent.get_stats())
                
                elif command == "graph":
                    if not args:
                        console.print("[red]Please specify graph command: stats, viz, or query[/red]")
                        continue
                    
                    graph_cmd = args.split()[0]
                    graph_args = ' '.join(args.split()[1:]) if len(args.split()) > 1 else None
                    
                    if graph_cmd == "stats":
                        # Get graph statistics
                        if hasattr(agent, 'pdf_parser') and hasattr(agent.pdf_parser, 'graph_manager'):
                            graph_mgr = agent.pdf_parser.graph_manager
                            if graph_mgr is not None:
                                stats = graph_mgr.get_graph_statistics()
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
                                    console.print("[yellow]Graph is empty[/yellow]")
                            else:
                                console.print("[yellow]Knowledge graph not initialized[/yellow]")
                        else:
                            console.print("[yellow]Knowledge graph not available[/yellow]")
                    
                    elif graph_cmd == "viz":
                        # Visualize graph
                        if hasattr(agent, 'pdf_parser') and hasattr(agent.pdf_parser, 'graph_manager'):
                            graph_mgr = agent.pdf_parser.graph_manager
                            if graph_mgr is not None:
                                output_path = Path("./outputs/graph_viz.json")
                                viz_data = graph_mgr.visualize_graph(
                                    output_path=output_path,
                                    max_nodes=100
                                )
                                
                                if viz_data.get('nodes'):
                                    console.print(f"\n[green]✓ Graph visualization saved to: {output_path}[/green]")
                                    console.print(f"Nodes: {len(viz_data['nodes'])}, Edges: {len(viz_data['edges'])}")
                                    
                                    # Show top concepts
                                    console.print("\n[bold cyan]Top Concepts:[/bold cyan]")
                                    sorted_nodes = sorted(viz_data['nodes'], key=lambda x: x['degree'], reverse=True)[:10]
                                    for i, node in enumerate(sorted_nodes, 1):
                                        console.print(f"{i}. {node['label']} (connections: {node['degree']})")
                                else:
                                    console.print("[yellow]Graph is empty or visualization failed[/yellow]")
                            else:
                                console.print("[yellow]Knowledge graph not initialized[/yellow]")
                        else:
                            console.print("[yellow]Knowledge graph not available[/yellow]")
                    
                    elif graph_cmd == "query":
                        # Query graph
                        if not graph_args:
                            console.print("[red]Please provide a query for the graph[/red]")
                            continue
                        
                        if hasattr(agent, 'pdf_parser') and hasattr(agent.pdf_parser, 'graph_manager'):
                            graph_mgr = agent.pdf_parser.graph_manager
                            if graph_mgr is not None:
                                with console.status("[bold green]Querying knowledge graph..."):
                                    result = graph_mgr.query_graph(graph_args)
                                
                                if result.get('response'):
                                    console.print("\n[bold cyan]Graph Query Response:[/bold cyan]")
                                    console.print(Markdown(result['response']))
                                    
                                    if result.get('nodes'):
                                        console.print(f"\n[bold]Related Nodes:[/bold] {len(result['nodes'])}")
                                        for node in result['nodes'][:10]:
                                            console.print(f"  • {node['label']}")
                                    
                                    if result.get('relationships'):
                                        console.print(f"\n[bold]Relationships:[/bold] {len(result['relationships'])}")
                                else:
                                    console.print("[yellow]No graph results found[/yellow]")
                            else:
                                console.print("[yellow]Knowledge graph not initialized[/yellow]")
                        else:
                            console.print("[yellow]Knowledge graph not available[/yellow]")
                    
                    else:
                        console.print("[red]Invalid graph command. Use: stats, viz, or query[/red]")
                
                elif command == "clear":
                    agent.clear_memory()
                    console.print("[green]✓ Memory cleared[/green]")
                
                elif command == "session":
                    if args.startswith("start"):
                        session_name = args.split(maxsplit=1)[1] if len(args.split()) > 1 else None
                        agent.start_session(session_name)
                        console.print(f"[green]✓ Started session: {session_name or 'New Session'}[/green]")
                    elif args == "end":
                        agent.end_session()
                        console.print("[green]✓ Session ended[/green]")
                    else:
                        console.print("[red]Invalid session command. Use 'session start <name>' or 'session end'[/red]")
                
                elif command == "providers":
                    console.print("\n[bold cyan]Current AI Providers:[/bold cyan]")
                    console.print(f"Embedding: {settings.EMBEDDING_PROVIDER}")
                    console.print(f"LLM: {settings.LLM_PROVIDER}")
                    console.print("\n[bold yellow]Note:[/bold yellow] Provider changes require restart to take effect")
                
                elif command == "set":
                    if not args:
                        console.print("[red]Please specify what to set. Use 'set embedding <provider>' or 'set llm <provider>'[/red]")
                        continue
                    
                    set_parts = args.split(maxsplit=1)
                    if len(set_parts) != 2:
                        console.print("[red]Invalid set command. Use 'set embedding <azure|ollama>' or 'set llm <poe|ollama>'[/red]")
                        continue
                    
                    set_type, set_value = set_parts
                    
                    if set_type == "embedding":
                        if set_value.lower() in ["azure", "ollama"]:
                            # Update settings
                            settings.EMBEDDING_PROVIDER = set_value.lower()
                            console.print(f"[green]✓ Embedding provider set to: {set_value}[/green]")
                            console.print("[yellow]Note: Restart required for changes to take effect[/yellow]")
                        else:
                            console.print("[red]Invalid embedding provider. Use 'azure' or 'ollama'[/red]")
                    
                    elif set_type == "llm":
                        if set_value.lower() in ["poe", "ollama"]:
                            # Update settings
                            settings.LLM_PROVIDER = set_value.lower()
                            console.print(f"[green]✓ LLM provider set to: {set_value}[/green]")
                            console.print("[yellow]Note: Restart required for changes to take effect[/yellow]")
                        else:
                            console.print("[red]Invalid LLM provider. Use 'poe' or 'ollama'[/red]")
                    else:
                        console.print("[red]Invalid set type. Use 'embedding' or 'llm'[/red]")
                
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