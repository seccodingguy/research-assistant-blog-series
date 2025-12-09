# utils/exceptions.py
"""
Custom exception classes for the Research Assistant application.
Provides specific exception types for better error handling and debugging.
"""


class ResearchAssistantError(Exception):
    """Base exception for all research assistant errors"""
    pass


class DocumentProcessingError(ResearchAssistantError):
    """Raised when document processing fails"""
    pass


class InvalidDocumentError(DocumentProcessingError):
    """Raised when a document is invalid or corrupted"""
    pass


class DocumentNotFoundError(DocumentProcessingError):
    """Raised when a document cannot be found"""
    pass


class SearchError(ResearchAssistantError):
    """Raised when search operations fail"""
    pass


class InvalidQueryError(SearchError):
    """Raised when a search query is invalid"""
    pass


class EmbeddingError(ResearchAssistantError):
    """Raised when embedding operations fail"""
    pass


class DatabaseError(ResearchAssistantError):
    """Raised when database operations fail"""
    pass


class ConnectionPoolError(DatabaseError):
    """Raised when connection pool operations fail"""
    pass


class GraphError(ResearchAssistantError):
    """Raised when knowledge graph operations fail"""
    pass


class PaperSearchError(ResearchAssistantError):
    """Raised when paper search operations fail"""
    pass


class APIError(ResearchAssistantError):
    """Raised when external API calls fail"""
    pass


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded"""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails"""
    pass


class ConfigurationError(ResearchAssistantError):
    """Raised when configuration is invalid"""
    pass


class ValidationError(ResearchAssistantError):
    """Raised when input validation fails"""
    pass


class FileValidationError(ValidationError):
    """Raised when file validation fails"""
    pass


class PathValidationError(ValidationError):
    """Raised when path validation fails"""
    pass
