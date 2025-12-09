"""
core/re_ranker.py

Maximal Marginal Relevance (MMR) re-ranking implementation for diversity optimization
in retrieval results. Implements the diversify_results function referenced in Week 3 blog post.

This module provides re-ranking strategies that balance relevance with source diversity
to ensure broader perspective coverage in RAG responses.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import log


class DocumentReRanker:
    """Re-rank retrieval results for improved diversity and relevance"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def diversify_results(self, nodes: List[Dict], 
                         diversity_weight: float = 0.3,
                         max_per_source: int = 3) -> List[Dict]:
        """
        Re-rank results to balance relevance with source diversity using MMR.
        
        Args:
            nodes: Retrieved document chunks with 'text', 'score', 'metadata'
            diversity_weight: Weight for diversity vs relevance (0.0-1.0)
            max_per_source: Maximum chunks per source document
            
        Returns:
            Re-ranked nodes optimizing relevance-diversity trade-off
        """
        if len(nodes) <= 2:
            return nodes
        
        log.info(f"Re-ranking {len(nodes)} results with diversity weight {diversity_weight}")
        
        # Step 1: Ensure source diversity (prevent single document dominance)
        diversified_nodes = self._ensure_source_diversity(nodes, max_per_source)
        
        if len(diversified_nodes) <= 2:
            return diversified_nodes
        
        # Step 2: Calculate text similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(diversified_nodes)
        
        # Step 3: Apply MMR re-ranking
        mmr_ranked = self._apply_mmr_ranking(
            diversified_nodes, similarity_matrix, diversity_weight
        )
        
        log.info(f"MMR re-ranking completed: {len(mmr_ranked)} results")
        return mmr_ranked
    
    def _ensure_source_diversity(self, nodes: List[Dict], 
                                max_per_source: int) -> List[Dict]:
        """Ensure no single document dominates results"""
        source_counts = defaultdict(int)
        diversified_nodes = []
        
        # Sort by relevance score first
        sorted_nodes = sorted(nodes, key=lambda x: x.get('score', 0), reverse=True)
        
        for node in sorted_nodes:
            source = node.get('metadata', {}).get('file_name', 'unknown')
            
            if source_counts[source] < max_per_source:
                diversified_nodes.append(node)
                source_counts[source] += 1
        
        log.debug(f"Source diversity filter: {len(nodes)} -> {len(diversified_nodes)} nodes")
        return diversified_nodes
    
    def _calculate_similarity_matrix(self, nodes: List[Dict]) -> np.ndarray:
        """Calculate pairwise similarity matrix between document chunks"""
        texts = [node.get('text', '') for node in nodes]
        
        try:
            # Use TF-IDF for more sophisticated similarity
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
        except (ValueError, Exception) as e:
            log.warning(f"TF-IDF similarity failed: {e}, using Jaccard fallback")
            # Fallback to Jaccard similarity
            similarity_matrix = self._jaccard_similarity_matrix(texts)
        
        return similarity_matrix
    
    def _jaccard_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Calculate Jaccard similarity matrix as fallback"""
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            words_i = set(texts[i].lower().split())
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    words_j = set(texts[j].lower().split())
                    if words_i or words_j:
                        jaccard = len(words_i & words_j) / len(words_i | words_j)
                        similarity_matrix[i][j] = jaccard
                        similarity_matrix[j][i] = jaccard
        
        return similarity_matrix
    
    def _apply_mmr_ranking(self, nodes: List[Dict], 
                          similarity_matrix: np.ndarray,
                          diversity_weight: float) -> List[Dict]:
        """Apply Maximal Marginal Relevance (MMR) re-ranking algorithm"""
        
        selected_indices = []
        remaining_indices = list(range(len(nodes)))
        
        # Start with highest relevance result
        if remaining_indices:
            best_idx = max(remaining_indices, key=lambda i: nodes[i].get('score', 0))
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Iteratively select results balancing relevance and diversity
        while remaining_indices and len(selected_indices) < min(20, len(nodes)):
            mmr_scores = []
            
            for candidate_idx in remaining_indices:
                relevance = nodes[candidate_idx].get('score', 0)
                
                # Calculate max similarity to already selected results
                max_similarity = 0
                for selected_idx in selected_indices:
                    max_similarity = max(
                        max_similarity, 
                        similarity_matrix[candidate_idx][selected_idx]
                    )
                
                # MMR score balances relevance and diversity
                mmr_score = ((1 - diversity_weight) * relevance - 
                            diversity_weight * max_similarity)
                
                mmr_scores.append((mmr_score, candidate_idx))
            
            # Select highest MMR score
            if mmr_scores:
                mmr_scores.sort(reverse=True)
                next_idx = mmr_scores[0][1]
                selected_indices.append(next_idx)
                remaining_indices.remove(next_idx)
        
        # Return re-ranked nodes
        return [nodes[i] for i in selected_indices]
    
    def temporal_relevance_filter(self, nodes: List[Dict], 
                                 boost_recent: bool = True,
                                 max_age_years: Optional[float] = None) -> List[Dict]:
        """
        Apply temporal and confidence-based filtering to results.
        
        Args:
            nodes: Retrieved document chunks
            boost_recent: Whether to boost more recent publications
            max_age_years: Maximum age in years (None for no limit)
            
        Returns:
            Filtered and re-scored nodes
        """
        from datetime import datetime, timedelta
        
        filtered_nodes = []
        current_date = datetime.now()
        
        for node in nodes:
            # Extract publication date from metadata
            metadata = node.get('metadata', {})
            pub_date_str = metadata.get('publication_date')
            
            pub_date = current_date  # Default to current date
            if pub_date_str:
                try:
                    # Handle various date formats
                    for fmt in ['%Y-%m-%d', '%Y', '%Y-%m']:
                        try:
                            pub_date = datetime.strptime(pub_date_str, fmt)
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass  # Use default current_date
            
            # Apply age filter
            age_years = (current_date - pub_date).days / 365.25
            if max_age_years and age_years > max_age_years:
                continue
            
            # Calculate temporal boost
            temporal_boost = 1.0
            if boost_recent:
                if age_years <= 1:
                    temporal_boost = 1.3
                elif age_years <= 2:
                    temporal_boost = 1.2
                elif age_years <= 5:
                    temporal_boost = 1.0
                else:
                    temporal_boost = 0.8
            
            # Apply confidence scoring
            base_score = node.get('score', 0.5)
            confidence_factors = self._calculate_confidence_factors(node)
            
            # Update node score
            adjusted_score = base_score * temporal_boost * confidence_factors
            
            # Create updated node
            updated_node = node.copy()
            updated_node['score'] = adjusted_score
            updated_node['temporal_boost'] = temporal_boost
            updated_node['confidence_factors'] = confidence_factors
            
            filtered_nodes.append(updated_node)
        
        # Sort by adjusted scores
        filtered_nodes.sort(key=lambda x: x['score'], reverse=True)
        
        log.info(f"Temporal filtering: {len(nodes)} -> {len(filtered_nodes)} nodes")
        return filtered_nodes
    
    def _calculate_confidence_factors(self, node: Dict) -> float:
        """Calculate confidence multiplier based on multiple factors"""
        confidence_multiplier = 1.0
        metadata = node.get('metadata', {})
        text = node.get('text', '')
        
        # Citation count boost (if available)
        citation_count = metadata.get('citation_count', 0)
        if citation_count > 100:
            confidence_multiplier *= 1.2
        elif citation_count > 50:
            confidence_multiplier *= 1.15
        elif citation_count > 10:
            confidence_multiplier *= 1.05
        
        # Complete sentence boost
        sentence_count = len([s for s in text.split('.') if len(s.strip()) > 10])
        if sentence_count >= 5:
            confidence_multiplier *= 1.1
        elif sentence_count >= 3:
            confidence_multiplier *= 1.05
        
        # Venue quality boost
        venue = metadata.get('venue', '').lower()
        high_quality_venues = [
            'ieee', 'acm', 'springer', 'nature', 'science', 'pnas',
            'neurips', 'icml', 'iclr', 'aaai', 'ijcai'
        ]
        
        if any(term in venue for term in high_quality_venues):
            confidence_multiplier *= 1.1
        
        # Length and completeness boost
        if len(text) > 500:  # Substantial content
            confidence_multiplier *= 1.05
        
        return confidence_multiplier
    
    def content_type_diversification(self, nodes: List[Dict]) -> List[Dict]:
        """Diversify results by content type (abstract, intro, methods, etc.)"""
        
        content_types = defaultdict(list)
        
        for node in nodes:
            # Classify content type based on text patterns
            content_type = self._classify_content_type(node.get('text', ''))
            content_types[content_type].append(node)
        
        # Select diverse content types
        diversified = []
        max_per_type = max(1, 15 // len(content_types)) if content_types else 1
        
        for content_type, type_nodes in content_types.items():
            # Sort by score and take top nodes
            type_nodes.sort(key=lambda x: x.get('score', 0), reverse=True)
            diversified.extend(type_nodes[:max_per_type])
        
        # Final sort by score
        diversified.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        log.debug(f"Content type diversification: {len(nodes)} -> {len(diversified)} nodes")
        return diversified[:20]  # Limit to top 20
    
    def _classify_content_type(self, text: str) -> str:
        """Classify content type based on text patterns"""
        text_lower = text.lower()
        
        # Abstract indicators
        if any(term in text_lower for term in ['abstract', 'summary', 'we present', 'this paper']):
            return 'abstract'
        
        # Introduction indicators
        elif any(term in text_lower for term in ['introduction', 'motivation', 'background']):
            return 'introduction'
        
        # Methods indicators
        elif any(term in text_lower for term in ['method', 'approach', 'algorithm', 'implementation']):
            return 'methods'
        
        # Results indicators
        elif any(term in text_lower for term in ['results', 'evaluation', 'experiment', 'performance']):
            return 'results'
        
        # Conclusion indicators
        elif any(term in text_lower for term in ['conclusion', 'future work', 'discussion']):
            return 'conclusion'
        
        # References
        elif re.search(r'\[\d+\]|\(\d{4}\)', text):
            return 'references'
        
        else:
            return 'content'


class RetrievalOptimizer:
    """High-level interface for optimizing retrieval results"""
    
    def __init__(self):
        self.re_ranker = DocumentReRanker()
    
    def optimize_retrieval_results(self, nodes: List[Dict], 
                                  strategy: str = "mmr",
                                  **kwargs) -> List[Dict]:
        """
        Optimize retrieval results using specified strategy.
        
        Args:
            nodes: Retrieved document chunks
            strategy: Optimization strategy ('mmr', 'temporal', 'content_type', 'combined')
            **kwargs: Strategy-specific parameters
            
        Returns:
            Optimized list of nodes
        """
        
        if not nodes:
            return nodes
        
        log.info(f"Optimizing {len(nodes)} results using {strategy} strategy")
        
        if strategy == "mmr":
            return self.re_ranker.diversify_results(
                nodes,
                diversity_weight=kwargs.get('diversity_weight', 0.3),
                max_per_source=kwargs.get('max_per_source', 3)
            )
        
        elif strategy == "temporal":
            return self.re_ranker.temporal_relevance_filter(
                nodes,
                boost_recent=kwargs.get('boost_recent', True),
                max_age_years=kwargs.get('max_age_years', None)
            )
        
        elif strategy == "content_type":
            return self.re_ranker.content_type_diversification(nodes)
        
        elif strategy == "combined":
            # Apply multiple strategies in sequence
            optimized = nodes
            
            # Step 1: Content type diversification
            optimized = self.re_ranker.content_type_diversification(optimized)
            
            # Step 2: Temporal filtering
            optimized = self.re_ranker.temporal_relevance_filter(
                optimized,
                boost_recent=kwargs.get('boost_recent', True),
                max_age_years=kwargs.get('max_age_years', None)
            )
            
            # Step 3: MMR re-ranking
            optimized = self.re_ranker.diversify_results(
                optimized,
                diversity_weight=kwargs.get('diversity_weight', 0.3),
                max_per_source=kwargs.get('max_per_source', 3)
            )
            
            return optimized
        
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")


# Convenience functions for backward compatibility
def diversify_results(nodes: List[Dict], diversity_weight: float = 0.3) -> List[Dict]:
    """Convenience function matching blog post example"""
    re_ranker = DocumentReRanker()
    return re_ranker.diversify_results(nodes, diversity_weight)


def apply_temporal_relevance_filters(nodes: List[Dict], 
                                   boost_recent: bool = True) -> List[Dict]:
    """Convenience function for temporal filtering"""
    re_ranker = DocumentReRanker()
    return re_ranker.temporal_relevance_filter(nodes, boost_recent)