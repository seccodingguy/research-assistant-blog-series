"""
Response Formatter Utility
Formats research assistant responses for better readability in terminal and web UI
"""
import re
from typing import Dict, Any, List
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


class ResponseFormatter:
    """Formats responses for terminal display with rich formatting"""
    
    def __init__(self):
        self.console = Console()
    
    def format_response(self, response_data: Dict[str, Any], format_type: str = "terminal") -> str:
        """
        Format response data based on output type
        
        Args:
            response_data: Response dictionary with 'response' key and optional 'sources', 'metadata'
            format_type: 'terminal' or 'web'
        
        Returns:
            Formatted string
        """
        if format_type == "terminal":
            return self._format_for_terminal(response_data)
        elif format_type == "web":
            return self._format_for_web(response_data)
        else:
            return str(response_data.get("response", ""))
    
    def _format_for_terminal(self, response_data: Dict[str, Any]) -> None:
        """Format and print response for terminal with rich formatting"""
        response = response_data.get("response", "")
        sources = response_data.get("sources", [])
        metadata = response_data.get("metadata", {})
        
        # Print main response with markdown formatting
        self.console.print("\n[bold cyan]═══ Research Assistant Response ═══[/bold cyan]\n")
        
        # Check if response looks like markdown
        if self._has_markdown_formatting(response):
            self.console.print(Markdown(response))
        else:
            # Apply basic formatting to plain text
            formatted_response = self._apply_basic_formatting(response)
            self.console.print(formatted_response)
        
        # Print sources if available
        if sources:
            self.console.print("\n[bold yellow]═══ Source Documents ═══[/bold yellow]\n")
            self._print_sources_table(sources)
        
        # Print metadata if available
        if metadata:
            self._print_metadata(metadata)
        
        self.console.print("\n[dim]─────────────────────────────────────[/dim]\n")
    
    def _format_for_web(self, response_data: Dict[str, Any]) -> str:
        """Format response for web UI with HTML"""
        response = response_data.get("response", "")
        sources = response_data.get("sources", [])
        
        # Convert markdown to HTML-friendly format
        html = self._markdown_to_html(response)
        
        # Add sources section if available
        if sources:
            html += self._sources_to_html(sources)
        
        return html
    
    def _has_markdown_formatting(self, text: str) -> bool:
        """Check if text contains markdown formatting"""
        markdown_patterns = [
            r'#{1,6}\s',  # Headers
            r'\*\*.*?\*\*',  # Bold
            r'\*.*?\*',  # Italic
            r'\[.*?\]\(.*?\)',  # Links
            r'^\s*[-*+]\s',  # Lists
            r'^\s*\d+\.\s',  # Numbered lists
            r'```',  # Code blocks
            r'`[^`]+`',  # Inline code
            r'^\s*>\s',  # Blockquotes
        ]
        
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False
    
    def _apply_basic_formatting(self, text: str) -> Text:
        """Apply basic rich formatting to plain text"""
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        formatted_text = Text()
        
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
            
            # Check if it's a heading (all caps or starts with number)
            if para.isupper() and len(para.split()) <= 10:
                formatted_text.append(para, style="bold cyan")
                formatted_text.append("\n\n")
            # Check if it's a list item
            elif re.match(r'^\s*[-•*]\s', para) or re.match(r'^\s*\d+\.\s', para):
                formatted_text.append(para, style="green")
                formatted_text.append("\n")
            # Check if it's a reference (contains parentheses with year)
            elif re.search(r'\(\d{4}\)', para):
                formatted_text.append(para, style="dim")
                formatted_text.append("\n\n")
            else:
                formatted_text.append(para)
                formatted_text.append("\n\n")
        
        return formatted_text
    
    def _print_sources_table(self, sources: List[Dict[str, Any]]) -> None:
        """Print sources in a formatted table"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="cyan", width=40)
        table.add_column("Relevance", justify="right", width=10)
        
        for i, source in enumerate(sources[:10], 1):
            file_name = source.get('file_name', 'Unknown')
            relevance = source.get('relevance_score', 0)
            relevance_str = f"{relevance:.2%}" if isinstance(relevance, float) else str(relevance)
            
            # Truncate long filenames
            if len(file_name) > 37:
                file_name = file_name[:34] + "..."
            
            table.add_row(str(i), file_name, relevance_str)
        
        self.console.print(table)
    
    def _print_metadata(self, metadata: Dict[str, Any]) -> None:
        """Print metadata information"""
        if not metadata:
            return
        
        self.console.print("\n[bold green]═══ Response Metadata ═══[/bold green]\n")
        
        if 'processing_time' in metadata:
            self.console.print(f"⏱️  Processing Time: {metadata['processing_time']:.2f}s")
        
        if 'document_count' in metadata:
            self.console.print(f"📄 Documents Processed: {metadata['document_count']}")
        
        if 'token_count' in metadata:
            self.console.print(f"🔢 Tokens Used: {metadata['token_count']}")
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown text to HTML with basic styling"""
        html = markdown_text
        
        # Headers
        html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # Bold and italic
        html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)
        html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)
        
        # Inline code
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
        
        # Links
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" target="_blank">\1</a>', html)
        
        # Code blocks
        html = re.sub(
            r'```(\w+)?\n(.*?)```',
            lambda m: f'<pre><code class="language-{m.group(1) or "text"}">{m.group(2)}</code></pre>',
            html,
            flags=re.DOTALL
        )
        
        # Lists
        html = self._format_lists_to_html(html)
        
        # Blockquotes
        html = re.sub(r'^>\s(.+?)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)
        
        # Line breaks and paragraphs
        html = re.sub(r'\n\n', '</p><p>', html)
        html = re.sub(r'\n', '<br>', html)
        
        # Wrap in paragraph tags if not already structured
        if not html.startswith('<'):
            html = f'<p>{html}</p>'
        
        return html
    
    def _format_lists_to_html(self, text: str) -> str:
        """Convert markdown lists to HTML"""
        lines = text.split('\n')
        result = []
        in_ul = False
        in_ol = False
        
        for line in lines:
            # Unordered list
            ul_match = re.match(r'^[\s]*[-*+]\s(.+)$', line)
            if ul_match:
                if not in_ul:
                    result.append('<ul>')
                    in_ul = True
                result.append(f'<li>{ul_match.group(1)}</li>')
                continue
            elif in_ul:
                result.append('</ul>')
                in_ul = False
            
            # Ordered list
            ol_match = re.match(r'^[\s]*\d+\.\s(.+)$', line)
            if ol_match:
                if not in_ol:
                    result.append('<ol>')
                    in_ol = True
                result.append(f'<li>{ol_match.group(1)}</li>')
                continue
            elif in_ol:
                result.append('</ol>')
                in_ol = False
            
            result.append(line)
        
        # Close any open lists
        if in_ul:
            result.append('</ul>')
        if in_ol:
            result.append('</ol>')
        
        return '\n'.join(result)
    
    def _sources_to_html(self, sources: List[Dict[str, Any]]) -> str:
        """Convert sources to HTML"""
        if not sources:
            return ""
        
        html = '<div class="sources-section"><h3>📚 Sources</h3><ul class="sources-list">'
        
        for source in sources[:10]:
            file_name = source.get('file_name', 'Unknown')
            relevance = source.get('relevance_score', 0)
            relevance_pct = f"{relevance:.1%}" if isinstance(relevance, float) else str(relevance)
            
            html += f'<li><span class="source-name">{file_name}</span> '
            html += f'<span class="source-relevance">({relevance_pct})</span></li>'
        
        html += '</ul></div>'
        
        return html


class WebResponseFormatter:
    """Formats responses specifically for web UI with enhanced HTML and styling"""
    
    @staticmethod
    def format_chat_response(response_data: Dict[str, Any]) -> str:
        """
        Format a chat response for web display with enhanced HTML
        
        Args:
            response_data: Response data from agent
            
        Returns:
            HTML-formatted string
        """
        response = response_data.get("response", "")
        sources = response_data.get("sources", [])
        
        # Build HTML structure
        html_parts = ['<div class="formatted-response">']
        
        # Main content with markdown-style formatting
        content_html = WebResponseFormatter._format_content(response)
        html_parts.append(f'<div class="response-content">{content_html}</div>')
        
        # Sources section
        if sources:
            sources_html = WebResponseFormatter._format_sources(sources)
            html_parts.append(sources_html)
        
        html_parts.append('</div>')
        
        return ''.join(html_parts)
    
    @staticmethod
    def _format_content(text: str) -> str:
        """Format content with markdown-like HTML"""
        if not text:
            return ""
        
        # Detect and format paragraphs
        sections = text.split('\n\n')
        formatted_sections = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Check for different content types
            if re.match(r'^#+\s', section):
                # Headers
                formatted_sections.append(WebResponseFormatter._format_header(section))
            elif re.match(r'^\s*>\s', section, re.MULTILINE):
                # Blockquotes
                formatted_sections.append(WebResponseFormatter._format_blockquote(section))
            elif re.match(r'^\s*[-*•]\s', section, re.MULTILINE):
                # Bullet lists
                formatted_sections.append(WebResponseFormatter._format_list(section, ordered=False))
            elif re.match(r'^\s*\d+\.\s', section, re.MULTILINE):
                # Numbered lists
                formatted_sections.append(WebResponseFormatter._format_list(section, ordered=True))
            elif '(' in section and ')' in section and re.search(r'\(\d{4}\)', section):
                # References/citations
                formatted_sections.append(f'<p class="reference">{WebResponseFormatter._format_inline(section)}</p>')
            else:
                # Regular paragraph
                formatted_sections.append(f'<p>{WebResponseFormatter._format_inline(section)}</p>')
        
        return ''.join(formatted_sections)
    
    @staticmethod
    def _format_header(text: str) -> str:
        """Format markdown headers to HTML"""
        match = re.match(r'^(#+)\s(.+)$', text)
        if match:
            level = len(match.group(1))
            content = match.group(2)
            return f'<h{level}>{content}</h{level}>'
        return text
    
    @staticmethod
    def _format_blockquote(text: str) -> str:
        """Format markdown blockquotes to HTML"""
        lines = text.split('\n')
        content = []
        
        for line in lines:
            match = re.match(r'^\s*>\s(.+)$', line)
            if match:
                content.append(WebResponseFormatter._format_inline(match.group(1)))
        
        return f'<blockquote>{" ".join(content)}</blockquote>'
    
    @staticmethod
    def _format_list(text: str, ordered: bool = False) -> str:
        """Format markdown lists to HTML"""
        tag = 'ol' if ordered else 'ul'
        lines = text.split('\n')
        items = []
        
        pattern = r'^\s*\d+\.\s(.+)$' if ordered else r'^\s*[-*•]\s(.+)$'
        
        for line in lines:
            match = re.match(pattern, line)
            if match:
                content = WebResponseFormatter._format_inline(match.group(1))
                items.append(f'<li>{content}</li>')
        
        return f'<{tag}>{"".join(items)}</{tag}>'
    
    @staticmethod
    def _format_inline(text: str) -> str:
        """Format inline markdown elements"""
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        
        # Inline code
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        # Links
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" target="_blank">\1</a>', text)
        
        return text
    
    @staticmethod
    def _format_sources(sources: List[Dict[str, Any]]) -> str:
        """Format sources section"""
        html = '<div class="sources-section"><h4>📚 Sources</h4><div class="sources-list">'
        
        for i, source in enumerate(sources[:10], 1):
            file_name = source.get('file_name', 'Unknown')
            relevance = source.get('relevance_score', 0)
            relevance_pct = f"{relevance:.0%}" if isinstance(relevance, float) else str(relevance)
            
            html += f'''
            <div class="source-item">
                <span class="source-number">{i}.</span>
                <span class="source-name">{file_name}</span>
                <span class="source-relevance">{relevance_pct}</span>
            </div>
            '''
        
        html += '</div></div>'
        
        return html


# Global formatter instances
terminal_formatter = ResponseFormatter()
web_formatter = WebResponseFormatter()
