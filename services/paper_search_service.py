"""
Paper Search Service - Integrates search engines with PDF downloading.

This service orchestrates paper searches across multiple academic databases
and provides seamless PDF downloading capabilities.
"""
import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from interfaces.interfaces import IService, SearchResult
from services.pdf_download_service import PDFDownloadService

# Import all search engines
from search_engines.arxiv import search_arxiv
from search_engines.semantic_scholar import search_semantic_scholar
from search_engines.crossref import search_crossref
from search_engines.duckduckgo_academic import search_duckduckgo_academic
from search_engines.google_scholar import search_google_scholar

logfilename = os.path.join("logs", "paper_search_service.log")

# Configure logging
logging.basicConfig(
    filename=logfilename,
    level=logging.INFO,
    filemode="w",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
FileOutputHandler = logging.FileHandler(logfilename)
logger.addHandler(FileOutputHandler)


class PaperSearchService(IService):
    """Service for searching academic papers and downloading PDFs."""

    # Search engine configurations
    SEARCH_ENGINES = {
        'arxiv': {
            'url': 'http://export.arxiv.org/api/query',
            'function': search_arxiv,
            'enabled': True
        },
        'semantic_scholar': {
            'url': 'https://api.semanticscholar.org/graph/v1/paper/search',
            'function': search_semantic_scholar,
            'enabled': True
        },
        'crossref': {
            'url': 'https://api.crossref.org/works',
            'function': search_crossref,
            'enabled': True
        },
        'duckduckgo_academic': {
            'url': 'https://duckduckgo.com/',
            'function': search_duckduckgo_academic,
            'enabled': False  # Often requires more complex scraping
        },
        'google_scholar': {
            'url': 'https://scholar.google.com/scholar',
            'function': search_google_scholar,
            'enabled': False  # Requires API key or scraping
        }
    }

    def __init__(
        self,
        download_directory: Union[str, Path] = "./downloads/search_results",
        config_path: Optional[Union[str, Path]] = None
    ):
        self._name = "paper_search_service"
        self._initialized = False
        self.download_directory = Path(download_directory)
        
        # Initialize PDF download service
        self.pdf_service = PDFDownloadService(download_directory)
        
        # Load configuration
        self.config = self._load_config(config_path)
        self._apply_config()
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'total_results': 0,
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'searches_by_engine': {},
            'results_by_engine': {}
        }

    @property
    def name(self) -> str:
        return self._name

    async def initialize(self) -> None:
        """Initialize the search service."""
        if self._initialized:
            return

        logger.info("Initializing Paper Search Service")
        
        # Create directories
        self.download_directory.mkdir(parents=True, exist_ok=True)
        (self.download_directory / "metadata").mkdir(exist_ok=True)
        
        # Initialize PDF download service
        await self.pdf_service.initialize()
        
        self._initialized = True
        logger.info("Paper Search Service initialized")

    async def shutdown(self) -> None:
        """Shutdown the search service."""
        if not self._initialized:
            return

        logger.info("Shutting down Paper Search Service")
        
        # Shutdown PDF service
        await self.pdf_service.shutdown()
        
        self._initialized = False
        logger.info("Paper Search Service shutdown complete")

    async def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 10,
        download_pdfs: bool = False,
        max_concurrent_downloads: int = 3,
        custom_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for academic papers across multiple sources.

        Args:
            query: Search query string
            sources: List of search engines to use (None = use all enabled)
            max_results_per_source: Maximum results per search engine
            download_pdfs: Whether to automatically download PDFs
            max_concurrent_downloads: Max concurrent PDF downloads
            custom_dir: Optional custom directory for downloads

        Returns:
            Dictionary with search results and download status
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Starting search: '{query}'")
        self.stats['total_searches'] += 1
        
        # Determine which sources to use
        if sources is None:
            sources = [name for name, config in self.SEARCH_ENGINES.items() 
                      if config['enabled']]
        else:
            # Filter to only enabled sources
            sources = [s for s in sources if s in self.SEARCH_ENGINES 
                      and self.SEARCH_ENGINES[s]['enabled']]
        
        if not sources:
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'sources_searched': [],
                'message': 'No search engines enabled'
            }

        # Execute searches concurrently
        search_tasks = []
        for source in sources:
            engine_config = self.SEARCH_ENGINES[source]
            task = self._search_engine(
                source,
                engine_config,
                query,
                max_results_per_source
            )
            search_tasks.append(task)

        results_by_source = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Aggregate results
        all_results = []
        sources_searched = []
        
        for source, results in zip(sources, results_by_source):
            if isinstance(results, Exception):
                logger.error(f"Error searching {source}: {results}")
                continue
            
            sources_searched.append(source)
            self.stats['searches_by_engine'][source] = \
                self.stats['searches_by_engine'].get(source, 0) + 1
            self.stats['results_by_engine'][source] = \
                self.stats['results_by_engine'].get(source, 0) + len(results)
            
            all_results.extend(results)

        # Remove duplicates based on DOI or title
        unique_results = self._deduplicate_results(all_results)
        
        self.stats['total_results'] += len(unique_results)
        
        logger.info(f"Search complete: {len(unique_results)} unique results from "
                   f"{len(sources_searched)} sources")

        # Prepare response
        response = {
            'query': query,
            'results': [self._result_to_dict(r) for r in unique_results],
            'total_results': len(unique_results),
            'sources_searched': sources_searched,
            'timestamp': datetime.now().isoformat()
        }

        # Download PDFs if requested
        if download_pdfs and unique_results:
            logger.info(f"Starting PDF downloads for {len(unique_results)} papers")
            
            # Use custom directory or create subfolder for this search
            if custom_dir:
                subfolder = custom_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                                   for c in query)[:50]
                subfolder = f"{timestamp}_{safe_query}"
            
            download_result = await self.pdf_service.download_batch(
                unique_results,
                subfolder=subfolder,
                max_concurrent=max_concurrent_downloads
            )
            
            response['downloads'] = download_result
            self.stats['total_downloads'] += download_result['downloadable']
            self.stats['successful_downloads'] += download_result['successful']
            self.stats['failed_downloads'] += download_result['failed']
            
            logger.info(f"Downloads complete: {download_result['successful']} successful, "
                       f"{download_result['failed']} failed")

        # Save search results metadata
        await self._save_search_results(query, response)

        return response

    async def search_and_download(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: int = 10,
        max_concurrent_downloads: int = 3,
        custom_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to search and automatically download PDFs.

        Args:
            query: Search query
            sources: Search engines to use
            max_results: Maximum total results
            max_concurrent_downloads: Max concurrent downloads
            custom_dir: Optional custom directory for downloads

        Returns:
            Search and download results
        """
        return await self.search(
            query=query,
            sources=sources,
            max_results_per_source=max_results,
            download_pdfs=True,
            max_concurrent_downloads=max_concurrent_downloads,
            custom_dir=custom_dir
        )

    async def download_by_doi(
        self,
        doi: str,
        subfolder: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for and download a paper by DOI.

        Args:
            doi: Digital Object Identifier
            subfolder: Optional subfolder for download

        Returns:
            Download result
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Searching for DOI: {doi}")
        
        # Search CrossRef for this DOI
        from search_engines.crossref import get_crossref_by_doi
        
        result = await get_crossref_by_doi(
            self.SEARCH_ENGINES['crossref']['url'],
            doi
        )
        
        if not result:
            return {
                'success': False,
                'doi': doi,
                'message': 'Paper not found'
            }

        # Try to download PDF
        download_result = await self.pdf_service.download_pdf(result, subfolder)
        
        return {
            'success': download_result['success'],
            'doi': doi,
            'paper': self._result_to_dict(result),
            'download': download_result
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            'pdf_service_stats': self.pdf_service.get_stats(),
            'download_directory': str(self.download_directory),
            'initialized': self._initialized,
            'enabled_engines': [name for name, cfg in self.SEARCH_ENGINES.items() 
                              if cfg['enabled']]
        }

    def get_available_sources(self) -> List[str]:
        """Get list of available search engines."""
        return list(self.SEARCH_ENGINES.keys())

    def get_enabled_sources(self) -> List[str]:
        """Get list of enabled search engines."""
        return [name for name, config in self.SEARCH_ENGINES.items() 
                if config['enabled']]

    def enable_source(self, source: str) -> bool:
        """Enable a search engine."""
        if source in self.SEARCH_ENGINES:
            self.SEARCH_ENGINES[source]['enabled'] = True
            logger.info(f"Enabled search engine: {source}")
            return True
        return False

    def disable_source(self, source: str) -> bool:
        """Disable a search engine."""
        if source in self.SEARCH_ENGINES:
            self.SEARCH_ENGINES[source]['enabled'] = False
            logger.info(f"Disabled search engine: {source}")
            return True
        return False

    async def _search_engine(
        self,
        source: str,
        engine_config: Dict[str, Any],
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Execute search on a single engine."""
        try:
            logger.info(f"Searching {source} for: '{query}'")
            
            # Get search function
            search_func = engine_config['function']
            url = engine_config['url']
            
            # Execute search (handling different function signatures)
            if source == 'semantic_scholar':
                api_key = self.config.get('semantic_scholar_api_key')
                results = await search_func(url, query, max_results, api_key)
            else:
                results = await search_func(url, query, max_results)
            
            logger.info(f"{source} returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching {source}: {e}")
            return []

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on DOI or title similarity."""
        seen_dois = set()
        seen_titles = set()
        unique = []
        
        for result in results:
            # Check DOI first (most reliable)
            if result.doi:
                doi_normalized = result.doi.lower().strip()
                if doi_normalized in seen_dois:
                    continue
                seen_dois.add(doi_normalized)
            
            # Check title similarity
            if result.title:
                title_normalized = result.title.lower().strip()
                # Simple deduplication - could be enhanced with fuzzy matching
                if title_normalized in seen_titles:
                    continue
                seen_titles.add(title_normalized)
            
            unique.append(result)
        
        logger.debug(f"Deduplicated {len(results)} results to {len(unique)} unique")
        return unique

    def _result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert SearchResult to dictionary."""
        return {
            'title': result.title,
            'authors': result.authors,
            'abstract': result.abstract,
            'year': result.year,
            'doi': result.doi,
            'url': result.url,
            'pdf_url': result.pdf_url,
            'source': result.source,
            'citations': result.citations,
            'metadata': result.metadata
        }

    async def _save_search_results(
        self,
        query: str,
        results: Dict[str, Any]
    ) -> None:
        """Save search results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                               for c in query)[:50]
            
            filename = f"search_{timestamp}_{safe_query}.json"
            filepath = self.download_directory / "metadata" / filename
            
            # Save as JSON
            async with asyncio.Lock():
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved search results to: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save search results: {e}")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path is None:
            config_path = Path("system_config.json")
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract search configuration
            if 'search' in config:
                return config['search']
            
            return {}
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _apply_config(self) -> None:
        """Apply configuration to search engines."""
        if not self.config:
            return
        
        # Apply enabled sources from config
        if 'enabled_sources' in self.config:
            enabled = set(self.config['enabled_sources'])
            for source in self.SEARCH_ENGINES:
                self.SEARCH_ENGINES[source]['enabled'] = source in enabled
        
        logger.info(f"Applied configuration: {self.get_enabled_sources()} enabled")
