# utils/response_saver.py
"""
Response saver utility for saving LLM responses in various formats.
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from utils.logger import log


class ResponseSaver:
    """Handles saving LLM responses in multiple formats"""
    
    SUPPORTED_FORMATS = ['txt', 'md', 'json', 'html', 'csv']
    
    @staticmethod
    def save_response(
        response_data: Dict[str, Any],
        file_path: Path,
        format_type: Optional[str] = None,
        auto_name: bool = False
    ) -> bool:
        """
        Save response to file in specified format.
        
        Args:
            response_data: Dictionary containing response information
            file_path: Path where to save the file
            format_type: Format type (txt, md, json, html, csv). If None, inferred from extension
            auto_name: If True, append timestamp to filename
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Ensure path is a Path object
            file_path = Path(file_path)
            
            # Auto-name with timestamp if requested
            if auto_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = file_path.stem
                suffix = file_path.suffix
                file_path = file_path.parent / f"{stem}_{timestamp}{suffix}"
            
            # Infer format from extension if not specified
            if format_type is None:
                format_type = file_path.suffix.lstrip('.').lower()
            
            # Validate format
            if format_type not in ResponseSaver.SUPPORTED_FORMATS:
                log.error(f"Unsupported format: {format_type}")
                return False
            
            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on format
            if format_type == 'txt':
                return ResponseSaver._save_as_txt(response_data, file_path)
            elif format_type == 'md':
                return ResponseSaver._save_as_markdown(response_data, file_path)
            elif format_type == 'json':
                return ResponseSaver._save_as_json(response_data, file_path)
            elif format_type == 'html':
                return ResponseSaver._save_as_html(response_data, file_path)
            elif format_type == 'csv':
                return ResponseSaver._save_as_csv(response_data, file_path)
            
            return False
            
        except Exception as e:
            log.error(f"Error saving response: {e}")
            return False
    
    @staticmethod
    def _save_as_txt(data: Dict[str, Any], file_path: Path) -> bool:
        """Save as plain text"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 80 + "\n")
                f.write("PDF AGENT RESPONSE\n")
                f.write("=" * 80 + "\n\n")
                
                # Timestamp
                timestamp = data.get('timestamp', datetime.now().isoformat())
                f.write(f"Timestamp: {timestamp}\n")
                
                # Mode
                if 'mode' in data:
                    f.write(f"Mode: {data['mode']}\n")
                
                # Query
                f.write(f"\nQuery:\n{'-' * 80}\n")
                f.write(f"{data.get('query', 'N/A')}\n\n")
                
                # Answer
                f.write(f"Answer:\n{'-' * 80}\n")
                f.write(f"{data.get('answer', 'N/A')}\n\n")
                
                # Sources
                if data.get('sources'):
                    f.write(f"Sources ({len(data['sources'])}):\n{'-' * 80}\n")
                    for i, source in enumerate(data['sources'], 1):
                        f.write(f"{i}. {source.get('file_name', 'Unknown')}\n")
                        if 'relevance_score' in source:
                            f.write(f"   Relevance: {source['relevance_score']:.4f}\n")
                        if 'file_path' in source:
                            f.write(f"   Path: {source['file_path']}\n")
                    f.write("\n")
                
                # Metadata
                if 'retrieved_chunks' in data:
                    f.write(f"Retrieved Chunks: {data['retrieved_chunks']}\n")
                if 'documents_analyzed' in data:
                    f.write(f"Documents Analyzed: {data['documents_analyzed']}\n")
                if 'batches_processed' in data:
                    f.write(f"Batches Processed: {data['batches_processed']}\n")
                
                # Footer
                f.write("\n" + "=" * 80 + "\n")
            
            log.info(f"Response saved as TXT: {file_path}")
            return True
            
        except Exception as e:
            log.error(f"Error saving as TXT: {e}")
            return False
    
    @staticmethod
    def _save_as_markdown(data: Dict[str, Any], file_path: Path) -> bool:
        """Save as Markdown"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Header
                f.write("# PDF Agent Response\n\n")
                
                # Metadata
                f.write("## Metadata\n\n")
                timestamp = data.get('timestamp', datetime.now().isoformat())
                f.write(f"- **Timestamp**: {timestamp}\n")
                if 'mode' in data:
                    f.write(f"- **Mode**: {data['mode']}\n")
                if 'retrieved_chunks' in data:
                    f.write(f"- **Retrieved Chunks**: {data['retrieved_chunks']}\n")
                if 'documents_analyzed' in data:
                    f.write(f"- **Documents Analyzed**: {data['documents_analyzed']}\n")
                if 'batches_processed' in data:
                    f.write(f"- **Batches Processed**: {data['batches_processed']}\n")
                f.write("\n")
                
                # Query
                f.write("## Query\n\n")
                f.write(f"{data.get('query', 'N/A')}\n\n")
                
                # Answer
                f.write("## Answer\n\n")
                f.write(f"{data.get('answer', 'N/A')}\n\n")
                
                # Sources
                if data.get('sources'):
                    f.write(f"## Sources ({len(data['sources'])})\n\n")
                    for i, source in enumerate(data['sources'], 1):
                        f.write(f"{i}. **{source.get('file_name', 'Unknown')}**\n")
                        if 'relevance_score' in source:
                            f.write(f"   - Relevance: {source['relevance_score']:.4f}\n")
                        if 'file_path' in source:
                            f.write(f"   - Path: `{source['file_path']}`\n")
                    f.write("\n")
                
                # Footer
                f.write("---\n")
                f.write("*Generated by PDF Agent*\n")
            
            log.info(f"Response saved as Markdown: {file_path}")
            return True
            
        except Exception as e:
            log.error(f"Error saving as Markdown: {e}")
            return False
    
    @staticmethod
    def _save_as_json(data: Dict[str, Any], file_path: Path) -> bool:
        """Save as JSON"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            log.info(f"Response saved as JSON: {file_path}")
            return True
            
        except Exception as e:
            log.error(f"Error saving as JSON: {e}")
            return False
    
    @staticmethod
    def _save_as_html(data: Dict[str, Any], file_path: Path) -> bool:
        """Save as HTML"""
        try:
            timestamp = data.get('timestamp', datetime.now().isoformat())
            
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Agent Response</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metadata-item {{
            margin: 5px 0;
        }}
        .query {{
            background-color: #e8f4f8;
            padding: 20px;
            border-left: 4px solid #3498db;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .answer {{
            background-color: #f0f9f0;
            padding: 20px;
            border-left: 4px solid #27ae60;
            border-radius: 5px;
            margin: 20px 0;
            white-space: pre-wrap;
        }}
        .sources {{
            margin-top: 20px;
        }}
        .source-item {{
            background-color: #fef9e7;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #f39c12;
        }}
        .source-name {{
            font-weight: bold;
            color: #d68910;
        }}
        .source-detail {{
            color: #666;
            font-size: 0.9em;
            margin-left: 20px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PDF Agent Response</h1>
        
        <div class="metadata">
            <h2>Metadata</h2>
            <div class="metadata-item"><strong>Timestamp:</strong> {timestamp}</div>
"""
            
            if 'mode' in data:
                html_content += f'            <div class="metadata-item"><strong>Mode:</strong> {data["mode"]}</div>\n'
            if 'retrieved_chunks' in data:
                html_content += f'            <div class="metadata-item"><strong>Retrieved Chunks:</strong> {data["retrieved_chunks"]}</div>\n'
            if 'documents_analyzed' in data:
                html_content += f'            <div class="metadata-item"><strong>Documents Analyzed:</strong> {data["documents_analyzed"]}</div>\n'
            if 'batches_processed' in data:
                html_content += f'            <div class="metadata-item"><strong>Batches Processed:</strong> {data["batches_processed"]}</div>\n'
            
            html_content += """        </div>
        
        <h2>Query</h2>
        <div class="query">
"""
            html_content += f'            {data.get("query", "N/A")}\n'
            html_content += """        </div>
        
        <h2>Answer</h2>
        <div class="answer">
"""
            html_content += f'{data.get("answer", "N/A")}\n'
            html_content += """        </div>
"""
            
            # Add sources if available
            if data.get('sources'):
                html_content += f"""        
        <h2>Sources ({len(data['sources'])})</h2>
        <div class="sources">
"""
                for i, source in enumerate(data['sources'], 1):
                    file_name = source.get('file_name', 'Unknown')
                    html_content += f'            <div class="source-item">\n'
                    html_content += f'                <div class="source-name">{i}. {file_name}</div>\n'
                    if 'relevance_score' in source:
                        html_content += f'                <div class="source-detail">Relevance: {source["relevance_score"]:.4f}</div>\n'
                    if 'file_path' in source:
                        html_content += f'                <div class="source-detail">Path: {source["file_path"]}</div>\n'
                    html_content += '            </div>\n'
                
                html_content += """        </div>
"""
            
            html_content += """        
        <div class="footer">
            Generated by PDF Agent
        </div>
    </div>
</body>
</html>
"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            log.info(f"Response saved as HTML: {file_path}")
            return True
            
        except Exception as e:
            log.error(f"Error saving as HTML: {e}")
            return False
    
    @staticmethod
    def _save_as_csv(data: Dict[str, Any], file_path: Path) -> bool:
        """Save as CSV"""
        try:
            timestamp = data.get('timestamp', datetime.now().isoformat())
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow(['PDF Agent Response'])
                writer.writerow([])
                
                # Metadata
                writer.writerow(['Metadata'])
                writer.writerow(['Field', 'Value'])
                writer.writerow(['Timestamp', timestamp])
                if 'mode' in data:
                    writer.writerow(['Mode', data['mode']])
                if 'retrieved_chunks' in data:
                    writer.writerow(['Retrieved Chunks', data['retrieved_chunks']])
                if 'documents_analyzed' in data:
                    writer.writerow(['Documents Analyzed', data['documents_analyzed']])
                if 'batches_processed' in data:
                    writer.writerow(['Batches Processed', data['batches_processed']])
                
                writer.writerow([])
                
                # Query
                writer.writerow(['Query'])
                writer.writerow([data.get('query', 'N/A')])
                writer.writerow([])
                
                # Answer
                writer.writerow(['Answer'])
                writer.writerow([data.get('answer', 'N/A')])
                writer.writerow([])
                
                # Sources
                if data.get('sources'):
                    writer.writerow(['Sources'])
                    writer.writerow(['#', 'File Name', 'Relevance Score', 'File Path'])
                    for i, source in enumerate(data['sources'], 1):
                        writer.writerow([
                            i,
                            source.get('file_name', 'Unknown'),
                            source.get('relevance_score', ''),
                            source.get('file_path', '')
                        ])
            
            log.info(f"Response saved as CSV: {file_path}")
            return True
            
        except Exception as e:
            log.error(f"Error saving as CSV: {e}")
            return False
