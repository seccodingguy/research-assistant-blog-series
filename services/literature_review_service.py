"""
Literature Review Service

Provides high-level operations for creating, managing, and analyzing literature reviews.
Integrates with the metadata database and leverages the knowledge graph for topic extraction.
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import csv
import io

from core.metadata_db import get_metadata_db, MetadataDatabase
from agents.research_assistant_agent import ResearchAssistantAgent
from utils.logger import log


class LiteratureReviewService:
    """Service for managing literature reviews with topic classification and analysis"""
    
    # Class-level agent instance to avoid re-initialization
    _shared_agent = None
    
    def __init__(self):
        self.db = get_metadata_db()
    
    def _get_agent(self):
        """Get or create a shared agent instance to avoid re-initialization"""
        if LiteratureReviewService._shared_agent is None:
            log.info("Creating shared ResearchAssistantAgent for literature review service")
            LiteratureReviewService._shared_agent = ResearchAssistantAgent()
        return LiteratureReviewService._shared_agent
    
    def _create_comprehensive_prompt(self) -> str:
        """Create a comprehensive analysis prompt for complete paper analysis."""
        return """
        You are an expert research analyst. Analyze this research paper 
        and conduct a comprehensive literature review.

        The analysis must include detailed information about:
        1. Research question(s): Main research question(s) being addressed
        2. Rationale: Why the authors chose to investigate this question
        3. Significance: Why this question matters in the field
        4. Theoretical background: Theoretical basis supporting the question
        5. Methodology: Research design and methods used
        6. Key findings: Primary results and statistical details
        7. Confidence level: Your confidence in the analysis (high/moderate/low)
        8. Limitations: Study limitations and weaknesses
        9. Implications: Broader implications for the field
        10. Future directions: Potential avenues for future research
        11. Authors: Extract ALL author names from the document
        12. Topics: Extract 3-5 key topics/themes from this paper
        13. Publication Year: Year this paper was published (e.g., "2023", "2021")
        14. DOI: Digital Object Identifier if available (e.g., "10.1234/example.2023")
        15. Source: Journal name, conference name, or publisher information

        IMPORTANT INSTRUCTIONS FOR AUTHOR EXTRACTION:
        - Look carefully at the title page and first few pages of the document
        - Author names often appear:
          * On the title page with the document title
          * After phrases like "by", "author:", "written by", "presented by"
          * Near institutional affiliations (universities, departments, companies)
          * In thesis/dissertation formats near degree information
          * In the header or footer of early pages
          * Before the abstract or introduction section
        - Extract ALL author names you find, even if you're not 100% certain
        - If you find any person's name in the first few pages that could be an author, include it
        - Use the format: ["First Last", "First Last"] for multiple authors
        - If you truly cannot find ANY author names despite careful examination, only then use ["Author names not found in document"]
        - DO NOT use placeholder text like "Not explicitly provided" - either provide actual names or state they were not found

        IMPORTANT INSTRUCTIONS FOR TOPIC EXTRACTION:
        - Identify 3-5 core topics, themes, or research areas covered in this paper
        - Topics should be:
          * Specific enough to be meaningful (e.g., "machine learning", "clinical trials", "organizational behavior")
          * General enough to group with other papers (avoid overly specific terms)
          * Representative of the main subject matter
        - Examples of good topics: "deep learning", "healthcare quality", "student engagement", "knowledge management"
        - Format as a list: ["topic1", "topic2", "topic3"]

        IMPORTANT INSTRUCTIONS FOR PUBLICATION METADATA:
        - Publication Year: Look for the publication date on the title page, header, footer, or citation information
          * Extract just the 4-digit year (e.g., "2023", "2021")
          * If you find a full date like "March 15, 2023", extract just "2023"
          * Common locations: title page, copyright notice, journal header
        - DOI: Digital Object Identifier usually appears as "DOI: 10.xxxx/xxxxx" or "https://doi.org/10.xxxx/xxxxx"
          * Format: Just the DOI string (e.g., "10.1234/example.2023")
          * Do not include "DOI:" prefix or full URL
          * If not found, use null
        - Source: The journal, conference, or publisher name
          * Examples: "Nature", "ACM CHI Conference", "IEEE Transactions on Neural Networks", "Springer"
          * Look for journal names in headers, title pages, or citation information
          * If it's a thesis/dissertation, use the university name
          * If not found, use null

        Respond with valid JSON only in this format:
        {
            "research_question": "...",
            "rationale": "...",
            "significance": "...",
            "theoretical_background": "...",
            "methodology": "...",
            "authors": ["Author 1", "Author 2"],
            "topics": ["topic1", "topic2", "topic3"],
            "publication_year": "2023",
            "doi": "10.1234/example.2023",
            "source": "Journal or Conference Name",
            "key_findings": "...",
            "confidence_level": "high|moderate|low",
            "limitations": "...",
            "implications": "...",
            "future_directions": "..."
        }
        """
    
    async def create_literature_review(
        self, 
        title: str,
        document_ids: List[str],
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        auto_extract: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new literature review and optionally auto-extract paper details.
        
        Args:
            title: Title of the literature review
            document_ids: List of document IDs to include
            description: Optional description
            created_by: Optional creator identifier
            auto_extract: Whether to automatically extract paper details using AI
            
        Returns:
            Dictionary with review details and statistics
        """
        try:
            # Generate unique review ID
            review_id = str(uuid.uuid4())
            
            # Validate documents exist
            valid_docs = []
            for doc_id in document_ids:
                doc = self.db.get_document(doc_id)
                if doc:
                    valid_docs.append(doc_id)
                else:
                    log.warning(f"Document not found: {doc_id}")
            
            if not valid_docs:
                raise ValueError("No valid documents found")
            
            # Extract topics from knowledge graph
            topic_classification = await self._extract_topics_from_documents(valid_docs)
            
            # Create the review
            self.db.create_literature_review(
                review_id=review_id,
                title=title,
                document_ids=valid_docs,
                description=description,
                created_by=created_by,
                topic_classification=topic_classification
            )
            
            # Auto-extract paper details if requested
            extraction_stats = {"processed": 0, "successful": 0, "failed": 0}
            if auto_extract:
                extraction_stats = await self._auto_extract_paper_details(review_id, valid_docs)
            
            log.info(f"Created literature review: {review_id} with {len(valid_docs)} documents")
            
            return {
                "review_id": review_id,
                "title": title,
                "document_count": len(valid_docs),
                "topic_count": len(topic_classification),
                "extraction_stats": extraction_stats,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            log.error(f"Error exporting literature review: {e}")
            raise
    
    def extract_missing_authors(self, document_ids: Optional[List[str]] = None) -> Dict[str, int]:
        """Extract author information for documents that don't have it"""
        stats = {"processed": 0, "updated": 0, "failed": 0}
        
        try:
            # Get documents without authors
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                if document_ids:
                    placeholders = ','.join('?' * len(document_ids))
                    cursor.execute(f"""
                        SELECT document_id, title FROM documents 
                        WHERE document_id IN ({placeholders}) AND (authors IS NULL OR authors = 'null')
                    """, document_ids)
                else:
                    cursor.execute("""
                        SELECT document_id, title FROM documents 
                        WHERE authors IS NULL OR authors = 'null'
                    """)
                
                docs_without_authors = cursor.fetchall()
            
            log.info(f"Found {len(docs_without_authors)} documents without author information")
            
            # Get shared agent to avoid re-initialization
            agent = self._get_agent()
            
            for doc_id, title in docs_without_authors:
                try:
                    stats["processed"] += 1
                    
                    # Create extraction prompt focused on authors
                    prompt = f"""
                    Extract the authors from this research document.
                    
                    Document Title: {title}
                    
                    Please search the document content and extract all author names.
                    Respond with valid JSON only:
                    {{
                        "authors": ["Author 1", "Author 2", "..."] or null if not found
                    }}
                    """
                    
                    result = agent.search(prompt, mode="enhanced", save_to_memory=False)
                    
                    if result and 'answer' in result:
                        answer = result['answer']
                        json_start = answer.find('{')
                        json_end = answer.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_data = json.loads(answer[json_start:json_end])
                            authors = json_data.get('authors')
                            
                            if authors and authors != 'null':
                                # Update document with extracted authors
                                with self.db.get_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        "UPDATE documents SET authors = ? WHERE document_id = ?",
                                        (json.dumps(authors), doc_id)
                                    )
                                    conn.commit()
                                
                                stats["updated"] += 1
                                log.info(f"Updated authors for document {doc_id}: {authors}")
                            else:
                                stats["failed"] += 1
                        else:
                            stats["failed"] += 1
                    else:
                        stats["failed"] += 1
                        
                except Exception as e:
                    log.error(f"Error extracting authors for document {doc_id}: {e}")
                    stats["failed"] += 1
            
            return stats
            
        except Exception as e:
            log.error(f"Error in extract_missing_authors: {e}")
            raise
    
    def get_literature_review(self, review_id: str, include_items: bool = True) -> Dict[str, Any]:
        """
        Get literature review details with optional items.
        
        Args:
            review_id: Review identifier
            include_items: Whether to include detailed review items
            
        Returns:
            Dictionary with review details
        """
        try:
            log.info(f"Getting literature review: {review_id} (include_items={include_items})")
            review = self.db.get_literature_review(review_id)
            if not review:
                raise ValueError(f"Literature review not found: {review_id}")
            
            result = {
                "review": review,
                "topic_groups": self.db.get_literature_review_topics(review_id)
            }
            
            if include_items:
                log.info(f"Loading items for review: {review_id}")
                items = self.db.get_literature_review_items(review_id)
                result["items"] = items
                result["item_count"] = len(items)
                log.info(f"Loaded {len(items)} items for review: {review_id}")
            
            log.info(f"Successfully retrieved literature review: {review_id}")
            return result
            
        except Exception as e:
            log.error(f"Error getting literature review: {e}")
            raise
    
    def list_literature_reviews(
        self, 
        limit: Optional[int] = None, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all literature reviews with summary statistics"""
        try:
            log.info(f"Listing literature reviews (limit={limit}, offset={offset})")
            reviews = self.db.list_literature_reviews(limit=limit, offset=offset)
            
            # Add summary stats for each review (optimized - avoid expensive queries)
            for review in reviews:
                review_id = review['review_id']
                
                # Get basic count from document_ids (may already be parsed or be JSON string)
                document_ids = review.get('document_ids', [])
                if isinstance(document_ids, str):
                    document_ids = json.loads(document_ids)
                document_count = len(document_ids)
                
                # Get topic count from stored topic_classification (may already be parsed or be JSON string)
                topic_classification = review.get('topic_classification')
                topic_count = 0
                if topic_classification:
                    try:
                        if isinstance(topic_classification, str):
                            topics = json.loads(topic_classification)
                        else:
                            topics = topic_classification
                        topic_count = len(topics) if isinstance(topics, dict) else 0
                    except (json.JSONDecodeError, TypeError):
                        topic_count = 0
                
                # For completed items, we need to query the items table, but only count
                completed_count = 0
                try:
                    with self.db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT COUNT(*) FROM literature_review_items 
                            WHERE review_id = ? AND research_question IS NOT NULL AND research_question != ''
                        """, (review_id,))
                        completed_count = cursor.fetchone()[0]
                except Exception as e:
                    log.warning(f"Could not get completed count for review {review_id}: {e}")
                
                review['summary'] = {
                    "item_count": document_count,
                    "topic_count": topic_count,
                    "completed_items": completed_count,
                }
            
            return reviews
            
        except Exception as e:
            log.error(f"Error listing literature reviews: {e}")
            raise
    
    def filter_literature_review_items(
        self,
        review_id: str,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter literature review items based on specified criteria.
        
        Args:
            review_id: Review identifier
            filters: Dictionary of filter criteria
                - topics: List of topics to include
                - confidence_levels: List of confidence levels
                - has_findings: Boolean to filter items with key findings
                - date_range: Dict with 'start' and 'end' dates
                - search_text: Text to search in various fields
                
        Returns:
            Filtered list of review items
        """
        try:
            items = self.db.get_literature_review_items(review_id)
            filtered_items = []
            
            for item in items:
                # Apply topic filter
                if filters.get('topics'):
                    item_topics = item.get('extracted_topics', [])
                    topic_match = any(topic in filters['topics'] for topic in item_topics)
                    if not topic_match:
                        continue
                
                # Apply confidence level filter
                if filters.get('confidence_levels'):
                    if item.get('confidence_level') not in filters['confidence_levels']:
                        continue
                
                # Apply findings filter
                if filters.get('has_findings'):
                    if not item.get('key_findings'):
                        continue
                
                # Apply date range filter
                if filters.get('date_range'):
                    pub_date = item.get('publication_date')
                    if pub_date:
                        try:
                            pub_datetime = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            start_date = datetime.fromisoformat(filters['date_range']['start'])
                            end_date = datetime.fromisoformat(filters['date_range']['end'])
                            
                            if not (start_date <= pub_datetime <= end_date):
                                continue
                        except (ValueError, TypeError):
                            continue
                
                # Apply text search filter
                if filters.get('search_text'):
                    search_text = filters['search_text'].lower()
                    searchable_fields = [
                        item.get('title', ''),
                        item.get('research_question', ''),
                        item.get('key_findings', ''),
                        item.get('methodology', ''),
                        ' '.join(item.get('authors', []) if isinstance(item.get('authors'), list) else [])
                    ]
                    
                    text_match = any(search_text in field.lower() for field in searchable_fields if field)
                    if not text_match:
                        continue
                
                filtered_items.append(item)
            
            log.info(f"Filtered {len(items)} items to {len(filtered_items)} items")
            return filtered_items
            
        except Exception as e:
            log.error(f"Error filtering literature review items: {e}")
            raise
    
    def export_to_csv(
        self,
        review_id: str,
        filters: Optional[Dict[str, Any]] = None,
        include_all_fields: bool = True
    ) -> str:
        """
        Export literature review items to CSV format.
        
        Args:
            review_id: Review identifier
            filters: Optional filters to apply before export
            include_all_fields: Whether to include all available fields
            
        Returns:
            CSV data as string
        """
        try:
            # Get items (filtered if requested)
            if filters:
                items = self.filter_literature_review_items(review_id, filters)
            else:
                items = self.db.get_literature_review_items(review_id)
            
            if not items:
                return "No items found for export"
            
            # Define CSV columns
            if include_all_fields:
                columns = [
                    'title', 'authors', 'publication_date', 'doi',
                    'research_question', 'rationale', 'significance',
                    'theoretical_background', 'methodology', 'key_findings',
                    'confidence_level', 'limitations', 'implications',
                    'future_directions', 'extracted_topics'
                ]
            else:
                columns = [
                    'title', 'authors', 'research_question', 'methodology',
                    'key_findings', 'confidence_level', 'extracted_topics'
                ]
            
            # Create CSV content
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            
            for item in items:
                # Prepare row data
                row = {}
                for col in columns:
                    value = item.get(col, '')
                    
                    # Handle list fields
                    if isinstance(value, list):
                        if col == 'authors':
                            value = '; '.join(value) if value else ''
                        elif col == 'extracted_topics':
                            value = ', '.join(value) if value else ''
                        else:
                            value = str(value)
                    
                    # Clean up None values
                    row[col] = value if value is not None else ''
                
                writer.writerow(row)
            
            csv_content = output.getvalue()
            output.close()
            
            log.info(f"Exported {len(items)} items to CSV")
            return csv_content
            
        except Exception as e:
            log.error(f"Error exporting to CSV: {e}")
            raise
    
    async def _extract_topics_from_documents(self, document_ids: List[str]) -> Dict[str, List[str]]:
        """Extract topics from documents using knowledge graph analysis"""
        try:
            topic_classification = {}
            
            # Get all graph nodes related to these documents
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get nodes that appear in the source documents
                placeholders = ','.join(['?' for _ in document_ids])
                cursor.execute(f"""
                    SELECT DISTINCT gn.label, gn.node_type, gn.frequency,
                           JSON_EXTRACT(gn.source_documents, '$') as source_docs
                    FROM graph_nodes gn
                    WHERE gn.source_documents IS NOT NULL
                """)
                
                nodes = cursor.fetchall()
                topic_groups = defaultdict(list)
                
                for node in nodes:
                    try:
                        source_docs = json.loads(node['source_docs']) if node['source_docs'] else []
                        # Check if this node appears in any of our documents
                        if any(doc_id in source_docs for doc_id in document_ids):
                            node_type = node['node_type']
                            if node_type in ['CONCEPT', 'TECHNIQUE', 'METHOD', 'DOMAIN', 'FIELD']:
                                topic_groups[node['label']].extend([
                                    doc_id for doc_id in document_ids if doc_id in source_docs
                                ])
                    except (json.JSONDecodeError, TypeError):
                        continue
                
                # Convert to final format
                for topic, docs in topic_groups.items():
                    topic_classification[topic] = list(set(docs))  # Remove duplicates
            
            log.info(f"Extracted {len(topic_classification)} topics from {len(document_ids)} documents")
            return topic_classification
            
        except Exception as e:
            log.error(f"Error extracting topics: {e}")
            return {}
    
    async def _auto_extract_paper_details(
        self, 
        review_id: str, 
        document_ids: List[str]
    ) -> Dict[str, int]:
        """Auto-extract paper details using AI analysis"""
        stats = {"processed": 0, "successful": 0, "failed": 0, "skipped_invalid": 0}
        
        try:
            # Get shared agent to avoid re-initialization
            agent = self._get_agent()
            
            for i, doc_id in enumerate(document_ids):
                try:
                    stats["processed"] += 1
                    
                    # Get document info
                    doc = self.db.get_document(doc_id)
                    if not doc:
                        stats["failed"] += 1
                        continue
                    
                    # Validate document is a valid PDF
                    file_type = doc.get('file_type', 'unknown')
                    if file_type != 'pdf':
                        log.warning(f"Skipping non-PDF document {doc_id}: type={file_type}")
                        # Remove from literature review
                        with self.db.get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "DELETE FROM literature_review_items WHERE review_id = ? AND document_id = ?",
                                (review_id, doc_id)
                            )
                        stats["skipped_invalid"] += 1
                        continue
                    
                    # Retrieve document content using file_hash
                    file_hash = doc.get('file_hash')
                    if not file_hash:
                        log.error(f"No file_hash for document {doc_id}")
                        stats["failed"] += 1
                        continue
                    
                    doc_chunks = agent.pdf_agent.context_manager.retrieve_document_content_by_id(
                        doc_id, file_hash, chunks_limit=30, prioritize_early_pages=True
                    )
                    
                    if not doc_chunks:
                        log.error(f"No content retrieved for document {doc_id}")
                        # Fallback: create basic entry
                        self.db.insert_literature_review_item(
                            review_id=review_id,
                            document_id=doc_id,
                            extraction_method='basic',
                            extraction_confidence=0.3
                        )
                        stats["failed"] += 1
                        continue
                    
                    # Create extraction prompt with actual document content
                    base_prompt = self._create_comprehensive_prompt()
                    
                    # Check if we need to extract author information
                    # Consider empty, null, or placeholder text as needing extraction
                    authors = doc.get('authors', '')
                    needs_author_extraction = (
                        not authors or 
                        authors == '' or 
                        'Unknown' in str(authors) or
                        'Not explicitly' in str(authors) or
                        'not found' in str(authors).lower()
                    )
                    
                    # Combine document chunks into context
                    doc_content = "\n\n".join([chunk['text'] for chunk in doc_chunks])
                    
                    # Add document-specific context to the prompt
                    doc_prompt = f"""
                    DOCUMENT TO ANALYZE:
                    Title: {doc.get('title', 'Unknown Title')}
                    Authors: {doc.get('authors', 'Unknown Authors')}
                    DOI: {doc.get('doi', 'Not available')}
                    
                    DOCUMENT CONTENT:
                    {doc_content}
                    
                    {base_prompt}
                    
                    {"IMPORTANT: Also extract and provide the authors' names from the document content if they are not listed above." if needs_author_extraction else ""}
                    """
                    
                    # Query the LLM directly instead of using search (which adds extra context)
                    from llama_index.core import Settings
                    if Settings.llm is not None and hasattr(Settings.llm, 'complete'):
                        llm_response = Settings.llm.complete(doc_prompt)
                        result = {'answer': llm_response.text if hasattr(llm_response, 'text') else str(llm_response)}
                    else:
                        # Fallback to search if LLM not available
                        result = agent.search(
                            doc_prompt,
                            mode="simple",
                            save_to_memory=False
                        )
                    
                    # Try to parse the result as JSON
                    if result and 'answer' in result:
                        try:
                            # Extract JSON from the answer
                            answer = result['answer']
                            json_start = answer.find('{')
                            json_end = answer.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = answer[json_start:json_end]
                                extracted_data = json.loads(json_str)
                                
                                # Handle authors extraction separately
                                extracted_authors = extracted_data.pop('authors', None)
                                if extracted_authors and needs_author_extraction:
                                    try:
                                        # Ensure authors is a list
                                        if isinstance(extracted_authors, str):
                                            extracted_authors = [extracted_authors]
                                            
                                        if isinstance(extracted_authors, list) and extracted_authors:
                                            with self.db.get_connection() as conn:
                                                cursor = conn.cursor()
                                                cursor.execute(
                                                    "UPDATE documents SET authors = ? WHERE document_id = ?",
                                                    (json.dumps(extracted_authors), doc_id)
                                                )
                                                conn.commit()
                                            log.info(f"Updated authors for document {doc_id}: {extracted_authors}")
                                    except Exception as e:
                                        log.error(f"Failed to update authors for {doc_id}: {e}")

                                # Extract and update publication metadata in documents table
                                publication_year = extracted_data.pop('publication_year', None)
                                doi = extracted_data.pop('doi', None)
                                source = extracted_data.pop('source', None)
                                
                                # Update documents table with extracted metadata
                                if publication_year or doi or source:
                                    try:
                                        update_parts = []
                                        update_values = []
                                        
                                        if publication_year:
                                            update_parts.append("publication_date = ?")
                                            update_values.append(str(publication_year))
                                        
                                        if doi:
                                            update_parts.append("doi = ?")
                                            update_values.append(str(doi))
                                        
                                        if source:
                                            update_parts.append("url = ?")  # Using url field to store source/journal name
                                            update_values.append(str(source))
                                        
                                        if update_parts:
                                            update_values.append(doc_id)
                                            with self.db.get_connection() as conn:
                                                cursor = conn.cursor()
                                                cursor.execute(
                                                    f"UPDATE documents SET {', '.join(update_parts)} WHERE document_id = ?",
                                                    tuple(update_values)
                                                )
                                                conn.commit()
                                            log.info(f"Updated metadata for document {doc_id}: year={publication_year}, doi={doi}, source={source}")
                                    except Exception as e:
                                        log.error(f"Failed to update metadata for {doc_id}: {e}")

                                # Extract confidence level and set extraction confidence
                                confidence_level = extracted_data.get('confidence_level', 'moderate')
                                confidence_mapping = {'high': 0.9, 'moderate': 0.7, 'low': 0.5}
                                extraction_confidence = confidence_mapping.get(confidence_level, 0.7)
                                
                                # Extract topics directly from AI response (bypassing broken graph extraction)
                                topics = extracted_data.pop('topics', [])
                                if not topics or not isinstance(topics, list):
                                    topics = []
                                
                                # Insert the review item
                                self.db.insert_literature_review_item(
                                    review_id=review_id,
                                    document_id=doc_id,
                                    extracted_topics=topics,
                                    extraction_method='ai_comprehensive',
                                    extraction_confidence=extraction_confidence,
                                    **extracted_data
                                )
                                
                                stats["successful"] += 1
                                log.info(f"Extracted details for document {i+1}/{len(document_ids)}: {doc_id}")
                            else:
                                # Fallback: create basic entry
                                self.db.insert_literature_review_item(
                                    review_id=review_id,
                                    document_id=doc_id,
                                    extraction_method='basic',
                                    extraction_confidence=0.3
                                )
                                stats["failed"] += 1
                                
                        except json.JSONDecodeError as e:
                            log.error(f"Failed to parse AI response for {doc_id}: {e}")
                            # Create basic entry
                            self.db.insert_literature_review_item(
                                review_id=review_id,
                                document_id=doc_id,
                                extraction_method='basic',
                                extraction_confidence=0.3
                            )
                            stats["failed"] += 1
                    else:
                        stats["failed"] += 1
                        
                except Exception as e:
                    log.error(f"Error processing document {doc_id}: {e}")
                    stats["failed"] += 1
            
            # Cleanup agent
            agent.shutdown()
            
        except Exception as e:
            log.error(f"Error in auto-extraction: {e}")
        
        return stats
    
    def soft_delete_literature_review(self, review_id: str, deleted_by: str) -> Dict[str, Any]:
        """Soft delete a literature review"""
        success = self.db.soft_delete_literature_review(review_id, deleted_by)
        return {
            'success': success,
            'message': 'Review deleted successfully' if success else 'Review not found or already deleted'
        }
    
    def restore_literature_review(self, review_id: str, restored_by: str) -> Dict[str, Any]:
        """Restore a soft-deleted literature review"""
        success = self.db.restore_literature_review(review_id, restored_by)
        return {
            'success': success,
            'message': 'Review restored successfully' if success else 'Review not found or not deleted'
        }
    
    def get_deleted_reviews(self, limit: int = 50) -> List[Dict]:
        """Get list of soft-deleted reviews"""
        all_reviews = self.db.list_literature_reviews(limit=limit*2, include_deleted=True)
        deleted_reviews = [r for r in all_reviews if r.get('is_deleted', False)]
        return deleted_reviews[:limit]
    
    async def add_papers_to_review(self, review_id: str, document_ids: List[str], 
                                 changed_by: str, auto_reanalyze: bool = True) -> Dict[str, Any]:
        """Add papers to an existing literature review with optional re-analysis"""
        result = self.db.add_papers_to_review(review_id, document_ids, changed_by)
        
        if result['success'] and result['added_count'] > 0 and auto_reanalyze:
            # Re-analyze the review with new papers
            try:
                await self._reanalyze_review(review_id, document_ids)
                result['reanalyzed'] = True
            except Exception as e:
                log.error(f"Error re-analyzing review after adding papers: {e}")
                result['reanalysis_error'] = str(e)
        
        return result
    
    async def remove_papers_from_review(self, review_id: str, document_ids: List[str], 
                                      changed_by: str, auto_reanalyze: bool = True) -> Dict[str, Any]:
        """Remove papers from an existing literature review with optional re-analysis"""
        result = self.db.remove_papers_from_review(review_id, document_ids, changed_by)
        
        if result['success'] and result['removed_count'] > 0 and auto_reanalyze:
            # Re-analyze the review after removing papers
            try:
                review = self.db.get_literature_review(review_id)
                if review and review.get('document_ids'):
                    await self._reanalyze_review(review_id, review['document_ids'])
                    result['reanalyzed'] = True
            except Exception as e:
                log.error(f"Error re-analyzing review after removing papers: {e}")
                result['reanalysis_error'] = str(e)
        
        return result
    
    def get_review_audit_log(self, review_id: str, limit: int = 50) -> List[Dict]:
        """Get audit log for a literature review"""
        return self.db.get_review_audit_log(review_id, limit)
    
    def get_available_documents_for_review(self, review_id: Optional[str] = None) -> List[Dict]:
        """Get list of available documents that can be added to a review"""
        # Get all documents from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT document_id, title, authors, publication_date, file_name, abstract
                FROM documents
                ORDER BY title
            """)
            
            rows = cursor.fetchall()
            all_docs = []
            for row in rows:
                doc = {
                    'document_id': row[0],
                    'title': row[1] or 'Untitled',
                    'authors': json.loads(row[2]) if row[2] else [],
                    'publication_date': row[3],
                    'file_name': row[4] or '',
                    'abstract': row[5] or ''
                }
                all_docs.append(doc)
        
        if review_id:
            # Filter out documents already in the review
            review = self.db.get_literature_review(review_id)
            if review and review.get('document_ids'):
                existing_ids = set(review['document_ids'])
                all_docs = [doc for doc in all_docs if doc['document_id'] not in existing_ids]
        
        # Format for frontend consumption
        formatted_docs = []
        for doc in all_docs:
            formatted_doc = {
                'document_id': doc['document_id'],
                'title': doc.get('title', 'Untitled'),
                'authors': doc.get('authors', []),
                'publication_date': doc.get('publication_date'),
                'file_name': doc.get('file_name', ''),
                'abstract': doc.get('abstract', '')[:200] + '...' if doc.get('abstract') and len(doc.get('abstract', '')) > 200 else doc.get('abstract', '')
            }
            formatted_docs.append(formatted_doc)
        
        return formatted_docs
    
    async def _reanalyze_review(self, review_id: str, document_ids: List[str]) -> None:
        """Re-analyze a review after papers are added/removed"""
        try:
            # Extract topics for the new/remaining documents
            topics = await self._extract_topics_from_documents(document_ids)
            
            # Update topic classification in the review
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE literature_reviews 
                    SET topic_classification = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE review_id = ?
                """, (json.dumps(topics), review_id))
            
            # Re-run auto-extraction for new papers if enabled
            review = self.db.get_literature_review(review_id)
            if review:
                await self._auto_extract_paper_details(review_id, document_ids)
            
        except Exception as e:
            log.error(f"Error in review re-analysis: {e}")
            raise
    
    def update_review_topic_classification(self, review_id: str) -> Dict[str, Any]:
        """
        Aggregate topics from all literature_review_items and update the review's topic_classification.
        This is useful when items have extracted_topics but the review-level topic_classification is empty.
        """
        try:
            # Get all items for this review
            items = self.db.get_literature_review_items(review_id)
            
            if not items:
                return {
                    "status": "no_items",
                    "message": "No items found for this review",
                    "topic_count": 0
                }
            
            # Build topic_groups dict: {topic: [doc_ids]}
            topic_groups = {}
            total_topics_found = 0
            
            for item in items:
                document_id = item.get('document_id')
                extracted_topics = item.get('extracted_topics', [])
                
                # Handle both JSON string and list formats
                if isinstance(extracted_topics, str):
                    try:
                        extracted_topics = json.loads(extracted_topics)
                    except:
                        extracted_topics = []
                
                if not extracted_topics:
                    continue
                
                total_topics_found += len(extracted_topics)
                
                # Add document to each topic group
                for topic in extracted_topics:
                    if not topic:
                        continue
                    
                    # Normalize topic (lowercase, strip)
                    topic_normalized = topic.strip().lower()
                    
                    if topic_normalized not in topic_groups:
                        topic_groups[topic_normalized] = []
                    
                    if document_id not in topic_groups[topic_normalized]:
                        topic_groups[topic_normalized].append(document_id)
            
            # Update review's topic_classification
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE literature_reviews 
                    SET topic_classification = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE review_id = ?
                """, (json.dumps(topic_groups), review_id))
                conn.commit()
            
            return {
                "status": "success",
                "message": f"Updated topic classification with {len(topic_groups)} unique topics",
                "topic_count": len(topic_groups),
                "total_topic_mentions": total_topics_found,
                "items_processed": len(items)
            }
            
        except Exception as e:
            log.error(f"Error updating review topic classification: {e}")
            raise
    
    async def process_missing_items(self, review_id: str) -> Dict[str, Any]:
        """
        Identify and process any documents in the review that haven't been extracted yet.
        Useful for resuming interrupted review creation.
        """
        try:
            # Get the review
            review = self.db.get_literature_review(review_id)
            if not review:
                raise ValueError(f"Literature review not found: {review_id}")
            
            all_doc_ids = review.get('document_ids', [])
            if isinstance(all_doc_ids, str):
                all_doc_ids = json.loads(all_doc_ids)
            
            # Get already processed items
            items = self.db.get_literature_review_items(review_id)
            processed_doc_ids = {item['document_id'] for item in items}
            
            # Identify missing documents
            missing_doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in processed_doc_ids]
            
            if not missing_doc_ids:
                return {
                    "status": "complete",
                    "message": "All documents have been processed",
                    "processed_count": 0,
                    "total_count": len(all_doc_ids)
                }
            
            log.info(f"Found {len(missing_doc_ids)} missing items for review {review_id}. Starting processing...")
            
            # Process missing items
            stats = await self._auto_extract_paper_details(review_id, missing_doc_ids)
            
            return {
                "status": "success",
                "message": f"Processed {stats['processed']} missing items",
                "stats": stats,
                "total_count": len(all_doc_ids),
                "missing_count_before": len(missing_doc_ids)
            }
            
        except Exception as e:
            log.error(f"Error processing missing items: {e}")
            raise
