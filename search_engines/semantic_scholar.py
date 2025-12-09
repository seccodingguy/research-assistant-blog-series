from typing import List, Dict, Any
import aiohttp
import asyncio
import logging
import traceback
import os
import xml.etree.ElementTree as ET
from interfaces.interfaces import SearchResult
import time

logfilename = os.path.join("logs", "semantic_search.log")

# Configure logging
logging.basicConfig(
    filename=logfilename,
    level=logging.INFO,  # Changed from DEBUG to INFO to prevent log spam
    filemode="w",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
FileOutputHandler = logging.FileHandler(logfilename)

logger.addHandler(FileOutputHandler)

# Suppress third-party DEBUG logging to prevent massive log files
logging.getLogger("pdfminer").setLevel(logging.WARNING)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

session = aiohttp.ClientSession()

async def search_semantic_scholar(url: str, query: str, max_results: int, semantic_scholar_api_key: str = None) -> List[SearchResult]:
        results = []
        
        try:
            url = url
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,authors,year,abstract,venue,citationCount,paperId,externalIds,url,openAccessPdf'
            }
            
            semantic_scholar_api_key = "xsg2S5NkwP7DtqmvtTa8FtKdMK2e6FO3NIj9cLL1"
            
            if semantic_scholar_api_key:
                headers['x-api-key'] = semantic_scholar_api_key
            time.sleep(1) # Semantic Scholar rate limiting
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    time.sleep(1)
                    for paper in data.get('data', []):
                        # Extract authors
                        authors = [author['name'] for author in paper.get('authors', [])]
                        
                        # Get PDF URL
                        pdf_url = None
                        if paper.get('openAccessPdf'):
                            pdf_url = paper['openAccessPdf'].get('url')
                        
                        # Get DOI
                        doi = None
                        external_ids = paper.get('externalIds', {})
                        if 'DOI' in external_ids:
                            doi = external_ids['DOI']
                        
                        # Paper URL
                        paper_url = paper.get('url', f"https://www.semanticscholar.org/paper/{paper['paperId']}")
                        
                            
                        results.append(SearchResult(
                            title=paper.get('title', ''),
                            url=paper_url,
                            source="Semantic Scholar",
                            pdf_url=paper_url,
                            year=paper.get('year'),
                            citations=paper.get('citationCount'),
                            doi=doi,
                            abstract=paper.get('abstract'),
                            authors=authors,
                            metadata={'semantic_scholar_id': paper.get('paperId')}
                        ))
                elif response.status == 429:
                    logger.warning(f"Semantic Scholar rate limited (429). "
                                   f"Request: {query}")
                    return []  # Return empty results instead of crashing
                else:
                    logger.warning(f"Semantic Scholar API error: {response.status} "
                                   f"for query: {query}")
                    return []  # Return empty results for other errors
                
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            logger.debug(traceback.format_exc())
        
        return results

def parse_semantic_scholar_results(data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Semantic Scholar API response."""
        results = []
        
        papers = data.get('data', [])
        for paper in papers:
            authors = []
            if paper.get('authors'):
                authors = [author.get('name', '') for author in paper['authors']]
            
            # Get PDF URL from openAccessPdf
            pdf_url = None
            if paper.get('openAccessPdf') and paper['openAccessPdf'].get('url'):
                pdf_url = paper['openAccessPdf']['url']
            
            result = SearchResult(
                title=paper.get('title', ''),
                abstract=paper.get('abstract', ''),
                authors=authors,
                year=paper.get('year'),
                citations=paper.get('citationCount'),
                source="Semantic Scholar",
                url=paper.get('url'),
                pdf_url=pdf_url,
                metadata={'semantic_scholar_id': paper.get('paperId')}
            )
            
            results.append(result)
        
        return results