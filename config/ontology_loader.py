# config/ontology_loader.py
"""
Ontology configuration loader for knowledge graph classification.

This module loads graph ontology configuration from YAML files,
allowing modification of types and keywords without code changes.
"""

import yaml
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from utils.logger import log
from config.classification_cache import get_classification_cache


class OntologyLoader:
    """Load and manage graph ontology configuration"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ontology loader.
        
        Args:
            config_path: Path to YAML config file. 
                        If None, uses default config/graph_ontology.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "graph_ontology.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
        self._relationship_enum = None
        self._concept_enum = None
        self._keyword_map = None
        self._pattern_rules = None
        self._relationship_norm_map = None
        
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Ontology config not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            
            log.info(f"✓ Loaded graph ontology from {self.config_path}")
            
            # Clear caches when config is reloaded
            self._build_relationship_enum.cache_clear()
            self._build_concept_enum.cache_clear()
            self._build_keyword_map.cache_clear()
            
        except Exception as e:
            log.error(f"Error loading ontology config: {e}")
            raise
    
    def reload_config(self) -> None:
        """Reload configuration from disk (useful for dynamic updates)"""
        log.info("Reloading ontology configuration...")
        self.load_config()
    
    @lru_cache(maxsize=1)
    def _build_relationship_enum(self) -> Enum:
        """Build RelationType enum from config"""
        relationship_types = self._config.get('relationship_types', [])
        
        enum_dict = {
            rel['name'].upper(): rel['name']
            for rel in relationship_types
        }
        
        return Enum('RelationType', enum_dict)
    
    @lru_cache(maxsize=1)
    def _build_concept_enum(self) -> Enum:
        """Build ConceptType enum from config"""
        concept_types = self._config.get('concept_types', [])
        
        enum_dict = {
            concept['name'].upper(): concept['display_name']
            for concept in concept_types
        }
        
        return Enum('ConceptType', enum_dict)
    
    @lru_cache(maxsize=1)
    def _build_keyword_map(self) -> Dict[str, str]:
        """
        Build keyword to concept type mapping from config.
        
        Returns:
            Dict mapping keywords to concept type display names
        """
        keyword_mappings = self._config.get('keyword_mappings', {})
        
        keyword_map = {}
        for concept_name, keywords in keyword_mappings.items():
            # Get the display name for this concept
            concept_display = self._get_concept_display_name(concept_name)
            
            # Map all keywords to this concept
            for keyword in keywords:
                keyword_map[keyword.lower()] = concept_display
        
        return keyword_map
    
    def _get_concept_display_name(self, concept_name: str) -> str:
        """Get display name for a concept by its name"""
        concept_types = self._config.get('concept_types', [])
        
        for concept in concept_types:
            if concept['name'] == concept_name:
                return concept['display_name']
        
        return concept_name
    
    def get_relationship_types(self) -> Enum:
        """Get RelationType enum"""
        if self._relationship_enum is None:
            self._relationship_enum = self._build_relationship_enum()
        return self._relationship_enum
    
    def get_concept_types(self) -> Enum:
        """Get ConceptType enum"""
        if self._concept_enum is None:
            self._concept_enum = self._build_concept_enum()
        return self._concept_enum
    
    def get_keyword_mappings(self) -> Dict[str, str]:
        """Get keyword to concept type mappings"""
        if self._keyword_map is None:
            self._keyword_map = self._build_keyword_map()
        return self._keyword_map
    
    def get_classification_patterns(self) -> List[Dict]:
        """Get pattern-based classification rules"""
        if self._pattern_rules is None:
            self._pattern_rules = self._config.get('classification_patterns', [])
        return self._pattern_rules
    
    def get_relationship_normalization_map(self) -> Dict[str, List[str]]:
        """Get relationship normalization mappings"""
        if self._relationship_norm_map is None:
            self._relationship_norm_map = self._config.get('relationship_mappings', {})
        return self._relationship_norm_map
    
    def classify_node(self, node_label: str) -> str:
        """
        Classify a node based on its label using loaded configuration.
        
        Args:
            node_label: The label/name of the node
            
        Returns:
            String representation of concept type or 'unknown'
        """
        node_lower = node_label.lower()
        
        # 1. Exact keyword match
        keyword_map = self.get_keyword_mappings()
        for keyword, concept_type in keyword_map.items():
            if keyword in node_lower:
                return concept_type
        
        # 2. Pattern-based classification
        patterns = self.get_classification_patterns()
        
        for pattern_rule in patterns:
            if 'pattern' in pattern_rule:
                # Regex pattern matching
                flags = re.IGNORECASE if pattern_rule.get('flags') == 'IGNORECASE' else 0
                if re.search(pattern_rule['pattern'], node_label if not flags else node_lower, flags):
                    # Check secondary conditions if any
                    if 'secondary_check' in pattern_rule:
                        secondary = pattern_rule['secondary_check']
                        if 'extensions' in secondary:
                            if not any(ext in node_lower for ext in secondary['extensions']):
                                continue
                    return pattern_rule['name']
            
            elif 'keywords' in pattern_rule:
                # Keyword-based pattern
                if any(kw in node_lower for kw in pattern_rule['keywords']):
                    return pattern_rule['name']
        
        # 3. Default to unknown
        return 'unknown'
    
    def normalize_relationship(self, relationship: str) -> str:
        """
        Normalize relationship to standard type.
        
        Args:
            relationship: Raw relationship string
            
        Returns:
            Normalized relationship name
        """
        rel_lower = relationship.lower().strip()
        
        normalization_map = self.get_relationship_normalization_map()
        
        # Check each mapping
        for normalized_name, variants in normalization_map.items():
            if rel_lower in [v.lower() for v in variants]:
                return normalized_name
        
        # Return original if no match
        return relationship
    
    def get_concept_categories(self) -> Dict[str, List[str]]:
        """
        Get concepts organized by category.
        
        Returns:
            Dict mapping category names to lists of concept names
        """
        concept_types = self._config.get('concept_types', [])
        
        categories = {}
        for concept in concept_types:
            category = concept.get('category', 'misc')
            if category not in categories:
                categories[category] = []
            categories[category].append(concept['name'])
        
        return categories
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded ontology"""
        return {
            'relationship_types': len(self._config.get('relationship_types', [])),
            'concept_types': len(self._config.get('concept_types', [])),
            'keyword_mappings': sum(
                len(keywords) 
                for keywords in self._config.get('keyword_mappings', {}).values()
            ),
            'classification_patterns': len(self._config.get('classification_patterns', [])),
            'relationship_mappings': len(self._config.get('relationship_mappings', {})),
            'categories': len(set(
                concept.get('category', 'misc')
                for concept in self._config.get('concept_types', [])
            ))
        }
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate the loaded configuration.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required sections
        required_sections = [
            'relationship_types',
            'concept_types',
            'keyword_mappings',
            'classification_patterns',
            'relationship_mappings'
        ]
        
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required section: {section}")
        
        # Validate relationship types
        rel_types = self._config.get('relationship_types', [])
        for i, rel in enumerate(rel_types):
            if 'name' not in rel:
                errors.append(f"Relationship type at index {i} missing 'name'")
        
        # Validate concept types
        concept_types = self._config.get('concept_types', [])
        concept_names = set()
        
        for i, concept in enumerate(concept_types):
            if 'name' not in concept:
                errors.append(f"Concept type at index {i} missing 'name'")
            else:
                if concept['name'] in concept_names:
                    errors.append(f"Duplicate concept name: {concept['name']}")
                concept_names.add(concept['name'])
            
            if 'display_name' not in concept:
                errors.append(f"Concept type '{concept.get('name', i)}' missing 'display_name'")
        
        # Validate keyword mappings reference valid concepts
        keyword_mappings = self._config.get('keyword_mappings', {})
        for concept_name in keyword_mappings.keys():
            if concept_name not in concept_names:
                errors.append(
                    f"Keyword mapping references unknown concept: {concept_name}"
                )
        
        # Validate pattern rules
        patterns = self._config.get('classification_patterns', [])
        for i, pattern in enumerate(patterns):
            if 'name' not in pattern:
                errors.append(f"Pattern rule at index {i} missing 'name'")
            
            # Must have either 'pattern' or 'keywords'
            if 'pattern' not in pattern and 'keywords' not in pattern:
                errors.append(
                    f"Pattern rule at index {i} must have 'pattern' or 'keywords'"
                )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def classify_with_llm(
        self,
        label: str,
        context: Optional[Dict] = None,
        llm_provider=None,
        use_cache: bool = True
    ) -> str:
        """
        Use LLM to classify a node based on label and optional context.
        
        This is used as a fallback when pattern/keyword matching fails.
        
        Args:
            label: The node label to classify
            context: Optional dict with:
                - neighbors: List of connected node labels
                - edge_types: List of relationship types
                - document_title: Source document name
            llm_provider: LLM provider instance (e.g., Settings.llm)
            use_cache: Whether to use/update the classification cache
            
        Returns:
            Concept type classification (or 'unknown' if LLM fails)
        """
        # Check cache first
        if use_cache:
            cache = get_classification_cache()
            cached_result = cache.get(label)
            if cached_result is not None:
                return cached_result
        
        if llm_provider is None:
            log.warning("LLM classification called but no provider given")
            return "unknown"
        
        # Build available types list
        concept_types = [
            concept['display_name']
            for concept in self._config.get('concept_types', [])
        ]
        
        # Build prompt with context
        prompt_parts = [
            "Classify this knowledge graph node into one of the available "
            "concept types.",
            "",
            f"Node label: \"{label}\"",
            ""
        ]
        
        # Add context if available
        if context:
            prompt_parts.append("Context:")
            if 'neighbors' in context and context['neighbors']:
                neighbors_str = ', '.join(
                    f'"{n}"' for n in context['neighbors'][:5]
                )
                prompt_parts.append(f"- Connected to: {neighbors_str}")
            
            if 'edge_types' in context and context['edge_types']:
                edges_str = ', '.join(context['edge_types'][:5])
                prompt_parts.append(f"- Relationships: {edges_str}")
            
            if 'document_title' in context:
                prompt_parts.append(
                    f"- Source document: {context['document_title']}"
                )
            
            prompt_parts.append("")
        
        # Add available types
        prompt_parts.extend([
            "Available concept types:",
            *[f"- {ctype}" for ctype in sorted(set(concept_types))],
            "",
            "Respond with ONLY the concept type name "
            "(e.g., 'algorithm', 'method', 'data').",
            "If none fit well, respond with 'unknown'.",
            "",
            "Classification:"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        try:
            # Query LLM
            response = llm_provider.complete(prompt).text.strip().lower()
            
            # Clean response (remove quotes, extra whitespace)
            response = response.strip('"\'` \n')
            
            # Validate response is a valid concept type
            valid_types = {ct.lower() for ct in concept_types}
            
            if response in valid_types or response == 'unknown':
                log.debug(f"LLM classified '{label}' as '{response}'")
                
                # Cache the result
                if use_cache:
                    cache.set(label, response, metadata=context)
                
                return response
            else:
                log.warning(
                    f"LLM returned invalid type '{response}' for '{label}'"
                )
                return "unknown"
                
        except Exception as e:
            log.error(f"LLM classification error for '{label}': {e}")
            return "unknown"
    
    def is_non_concept(self, label: str) -> bool:
        """
        Check if a label matches non-concept patterns (timestamps, paths, variables).
        
        This is used in the hybrid approach to filter out synthetic elements
        before attempting LLM classification.
        
        Args:
            label: The node label to check
            
        Returns:
            True if label matches a non-concept pattern
        """
        patterns = self._config.get('classification_patterns', [])
        
        for rule in patterns:
            # Only check non_concept patterns
            if rule.get('name') != 'non_concept':
                continue
            
            if 'pattern' in rule:
                flags = 0
                if 'flags' in rule:
                    flag_str = rule['flags'].upper()
                    if 'IGNORECASE' in flag_str:
                        flags |= re.IGNORECASE
                
                try:
                    if re.search(rule['pattern'], label, flags):
                        log.debug(f"'{label}' matched non-concept pattern: {rule.get('description')}")
                        return True
                except re.error as e:
                    log.warning(f"Invalid regex pattern '{rule['pattern']}': {e}")
                    continue
        
        return False


# Global singleton instance
_ontology_loader = None


def get_ontology_loader(config_path: Optional[Path] = None) -> OntologyLoader:
    """
    Get or create the global ontology loader instance.
    
    Args:
        config_path: Optional path to config file (only used on first call)
        
    Returns:
        OntologyLoader instance
    """
    global _ontology_loader
    
    if _ontology_loader is None:
        _ontology_loader = OntologyLoader(config_path)
    
    return _ontology_loader


def reload_ontology() -> None:
    """Reload ontology configuration from disk"""
    global _ontology_loader
    
    if _ontology_loader is not None:
        _ontology_loader.reload_config()
    else:
        _ontology_loader = OntologyLoader()
