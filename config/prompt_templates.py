"""
config/prompt_templates.py

Production-ready citation-enforced prompt templates for RAG systems.
Implements the structured prompt templates described in Week 3 blog post.

These templates ensure consistent citation behavior, structured responses,
and explicit handling of uncertainty in RAG-generated responses.
"""

from llama_index.core import PromptTemplate
from typing import Dict, Optional


# Citation-enforced comprehensive QA template from blog post
COMPREHENSIVE_QA_TEMPLATE = PromptTemplate(
    """You are an AI research assistant with access to a curated knowledge base of academic documents. Your role is to provide accurate, well-sourced analysis based exclusively on the provided context.

## RETRIEVED CONTEXT ##
{context_str}

## CONVERSATION HISTORY ##
{conversation_history}

## USER QUESTION ##
{query_str}

## RESPONSE INSTRUCTIONS ##

1. **Evidence-Based Analysis**: Base your response exclusively on the provided context. Do not introduce external knowledge or speculation.

2. **Mandatory Citations**: For every claim, cite the source document using this exact format: [Source: filename.pdf]. Multiple sources should be listed as [Sources: file1.pdf, file2.pdf].

3. **Multi-Document Synthesis**: When analyzing multiple papers:
   - Discuss each relevant document explicitly
   - Compare and contrast findings across sources
   - Identify agreements, disagreements, and complementary insights
   - Synthesize coherent conclusions from multiple perspectives

4. **Structured Response Format**:
   - **Summary**: Brief overview of findings
   - **Detailed Analysis**: Comprehensive examination with citations
   - **Key Insights**: Main takeaways and implications
   - **Limitations**: What the sources don't address

5. **Uncertainty Handling**: If the context doesn't fully address the question:
   - State this explicitly
   - Provide what information is available
   - Identify specific gaps in the evidence

6. **Response Quality**:
   - Be comprehensive yet focused
   - Use precise technical language
   - Maintain academic tone
   - Ensure logical flow and coherence

## RESPONSE ##

**Summary:**
[Provide a concise overview of your findings]

**Detailed Analysis:**
[Comprehensive examination with mandatory citations]

**Key Insights:**
[Main conclusions and implications]

**Limitations:**
[What questions remain unanswered by the available sources]
"""
)


# Simple QA template for basic searches
SIMPLE_QA_TEMPLATE = PromptTemplate(
    """You are an AI assistant with access to a knowledge base of PDF documents.
            
Context from documents:
{context_str}

Conversation history:
{conversation_history}

User question: {query_str}

Instructions:
1. Answer based on the provided context and conversation history
2. ALWAYS cite specific documents by their file names when making claims
3. If analyzing multiple papers, discuss each one explicitly
4. Synthesize findings across documents when relevant
5. If the context doesn't contain relevant information, say so
6. Be comprehensive and detailed in your analysis

Answer:"""
)


# Enhanced template with graph context integration
GRAPH_ENHANCED_TEMPLATE = PromptTemplate(
    """You are an AI research assistant with access to both document content and knowledge graph relationships.

## DOCUMENT CONTEXT ##
{context_str}

## KNOWLEDGE GRAPH INSIGHTS ##
{graph_context}

## CONVERSATION HISTORY ##
{conversation_history}

## USER QUESTION ##
{query_str}

## INSTRUCTIONS ##

1. **Integrate Multiple Sources**: Use both document content and knowledge graph relationships to provide comprehensive answers.

2. **Citation Requirements**: 
   - Document claims: [Source: filename.pdf]
   - Graph relationships: [Graph: entity1 -> relationship -> entity2]

3. **Relationship Analysis**: When relevant, explain how concepts connect based on the knowledge graph structure.

4. **Evidence Hierarchy**: Prioritize document content for factual claims, use graph relationships for conceptual connections.

**Answer:**
"""
)


# Summarization template for condensed responses
SUMMARIZATION_TEMPLATE = PromptTemplate(
    """Provide a comprehensive but concise summary answering the following question based on the provided document context.

## QUESTION ##
{query_str}

## DOCUMENT CONTEXT ##
{context_str}

## INSTRUCTIONS ##
- Focus on the most important information that directly addresses the question
- Cite specific documents for all claims using [Source: filename.pdf] format
- Limit response to 3-4 paragraphs maximum
- Include key findings, methodologies, and conclusions
- If multiple documents address the question, synthesize their findings

**Summary:**
"""
)


# Analysis template for comprehensive document examination
ANALYSIS_ALL_TEMPLATE = PromptTemplate(
    """You are conducting a comprehensive analysis of a document collection. Provide a thorough examination addressing the user's question across ALL available documents.

## USER QUESTION ##
{query_str}

## COMPLETE DOCUMENT COLLECTION ##
{context_str}

## ANALYSIS INSTRUCTIONS ##

1. **Comprehensive Coverage**: Review ALL provided documents systematically
2. **Structured Analysis**: Organize findings by themes, methodologies, or chronology as appropriate
3. **Cross-Document Synthesis**: Identify patterns, contradictions, and knowledge gaps across the collection
4. **Evidence Mapping**: For each major claim, provide multiple supporting sources where available
5. **Critical Assessment**: Evaluate the strength and limitations of the evidence base

## RESPONSE FORMAT ##

**Executive Summary:**
[High-level overview of findings across the entire collection]

**Thematic Analysis:**
[Organize findings by major themes or research areas]

**Methodological Assessment:**
[Compare and evaluate research approaches used]

**Evidence Synthesis:**
[Synthesize conclusions with supporting citations]

**Knowledge Gaps and Future Directions:**
[Identify areas needing further research]

**Complete Source List:**
[List all documents analyzed with brief descriptions]
"""
)


# Error handling template for when context is insufficient
INSUFFICIENT_CONTEXT_TEMPLATE = PromptTemplate(
    """I apologize, but I cannot provide a complete answer to your question based on the available context.

## YOUR QUESTION ##
{query_str}

## AVAILABLE CONTEXT ##
{context_str}

## WHAT I CAN TELL YOU ##
Based on the limited available information:
{partial_answer}

## WHAT'S MISSING ##
To fully answer your question, I would need information about:
{missing_information}

## SUGGESTIONS ##
- Try rephrasing your question to be more specific
- Check if additional relevant documents are available in the knowledge base
- Consider asking about the specific aspects I was able to address above

Would you like me to focus on any particular aspect that I can address with the available information?
"""
)


# Comparative analysis template
COMPARATIVE_TEMPLATE = PromptTemplate(
    """Compare and analyze the following aspects based on the provided documents:

## COMPARISON REQUEST ##
{query_str}

## DOCUMENT SOURCES ##
{context_str}

## CONVERSATION CONTEXT ##
{conversation_history}

## COMPARATIVE ANALYSIS INSTRUCTIONS ##

1. **Structured Comparison**: Create clear comparisons with side-by-side analysis
2. **Evidence-Based**: Support all comparative claims with specific citations
3. **Balanced Perspective**: Present strengths and weaknesses of each approach/finding
4. **Synthesis**: Conclude with integrated insights from the comparison

## RESPONSE FORMAT ##

**Comparison Overview:**
[Brief summary of what is being compared]

**Detailed Comparison:**
[Point-by-point analysis with citations]

**Strengths and Weaknesses:**
[Balanced assessment of each approach/finding]

**Synthesis and Conclusions:**
[Integrated insights and recommendations]
"""
)


class PromptTemplateManager:
    """Manage and customize prompt templates for different use cases"""
    
    def __init__(self):
        self.templates = {
            'comprehensive': COMPREHENSIVE_QA_TEMPLATE,
            'simple': SIMPLE_QA_TEMPLATE,
            'graph_enhanced': GRAPH_ENHANCED_TEMPLATE,
            'summarization': SUMMARIZATION_TEMPLATE,
            'analysis_all': ANALYSIS_ALL_TEMPLATE,
            'insufficient_context': INSUFFICIENT_CONTEXT_TEMPLATE,
            'comparative': COMPARATIVE_TEMPLATE
        }
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific prompt template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        return self.templates[template_name]
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt template with the provided arguments"""
        template = self.get_template(template_name)
        return template.format(**kwargs)
    
    def add_custom_template(self, name: str, template: PromptTemplate):
        """Add a custom prompt template"""
        self.templates[name] = template
    
    def get_template_for_mode(self, search_mode: str, 
                             has_graph: bool = False,
                             context_quality: str = "good") -> PromptTemplate:
        """
        Get the appropriate template based on search mode and context quality.
        
        Args:
            search_mode: 'simple', 'enhanced', 'summarize', 'analyze_all', 'comparative'
            has_graph: Whether knowledge graph context is available
            context_quality: 'good', 'limited', 'insufficient'
            
        Returns:
            Appropriate PromptTemplate instance
        """
        
        if context_quality == "insufficient":
            return self.templates['insufficient_context']
        
        if search_mode == "simple":
            return self.templates['simple']
        
        elif search_mode == "enhanced":
            if has_graph:
                return self.templates['graph_enhanced']
            else:
                return self.templates['comprehensive']
        
        elif search_mode == "summarize":
            return self.templates['summarization']
        
        elif search_mode == "analyze_all":
            return self.templates['analysis_all']
        
        elif search_mode == "comparative":
            return self.templates['comparative']
        
        else:
            # Default to comprehensive template
            return self.templates['comprehensive']
    
    def customize_citation_format(self, template_name: str, 
                                 citation_format: str) -> PromptTemplate:
        """
        Customize citation format for a template.
        
        Args:
            template_name: Name of template to customize
            citation_format: New citation format string
            
        Returns:
            Customized PromptTemplate
        """
        
        original_template = self.get_template(template_name)
        template_text = original_template.template
        
        # Replace citation format instructions
        old_formats = [
            "[Source: filename.pdf]",
            "[Sources: file1.pdf, file2.pdf]"
        ]
        
        for old_format in old_formats:
            template_text = template_text.replace(old_format, citation_format)
        
        return PromptTemplate(template_text)
    
    def validate_template_variables(self, template_name: str, 
                                   variables: Dict[str, str]) -> Dict[str, str]:
        """
        Validate that all required template variables are provided.
        
        Args:
            template_name: Name of template to validate
            variables: Dictionary of variable values
            
        Returns:
            Dictionary with missing variables filled with defaults
        """
        
        template = self.get_template(template_name)
        
        # Extract variable names from template
        import re
        variable_pattern = r'\{([^}]+)\}'
        required_vars = set(re.findall(variable_pattern, template.template))
        
        # Fill missing variables with defaults
        validated_vars = variables.copy()
        defaults = {
            'context_str': 'No context available.',
            'conversation_history': 'No previous conversation.',
            'query_str': 'No query provided.',
            'graph_context': 'No graph information available.',
            'partial_answer': 'Limited information available.',
            'missing_information': 'Additional context needed.'
        }
        
        for var in required_vars:
            if var not in validated_vars:
                validated_vars[var] = defaults.get(var, f'[{var} not provided]')
        
        return validated_vars


# Global template manager instance
template_manager = PromptTemplateManager()


# Convenience functions for common use cases
def get_comprehensive_template() -> PromptTemplate:
    """Get the comprehensive QA template with citation enforcement"""
    return COMPREHENSIVE_QA_TEMPLATE


def get_simple_template() -> PromptTemplate:
    """Get the simple QA template for basic searches"""
    return SIMPLE_QA_TEMPLATE


def format_comprehensive_prompt(query: str, context: str, 
                               conversation_history: str = "") -> str:
    """Format the comprehensive prompt with provided parameters"""
    return COMPREHENSIVE_QA_TEMPLATE.format(
        query_str=query,
        context_str=context,
        conversation_history=conversation_history
    )


def create_custom_template(template_text: str) -> PromptTemplate:
    """Create a custom PromptTemplate from template text"""
    return PromptTemplate(template_text)