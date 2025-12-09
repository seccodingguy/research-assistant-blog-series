from typing import List, Dict, Any, Optional
import aiohttp
#import asyncio
import logging
import traceback
import os
#from bs4 import BeautifulSoup
#import re
#import xml.etree.ElementTree as ET
from interfaces.interfaces import SearchResult

logfilename = os.path.join("logs", "crossref.log")

# Configure logging
logging.basicConfig(
    filename=logfilename,
    level=logging.DEBUG,
    filemode="w",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
FileOutputHandler = logging.FileHandler(logfilename)

logger.addHandler(FileOutputHandler)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

session = aiohttp.ClientSession()

async def search_crossref(url: str, query: str, max_results: int) -> List[SearchResult]:
        """Search Crossref."""
        
        params = {
            'query': query,
            'rows': max_results,
            'sort': 'score',
            'order': 'desc'
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Crossref API error: {response.status}")
                    return []
                
                data = await response.json()
               
                logger.debug(f"Crossref response data: {data}")
                return parse_crossref_results(data)
                
        except Exception as e:
            logger.error(f"Crossref search error: {e}")
            logger.error(traceback.format_exc())
            return []
   
    
async def get_crossref_by_doi(url: str, doi: str) -> Optional[SearchResult]:
        """Get paper by DOI from Crossref."""
        url = f"{url}/{doi}"
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                work = data.get('message', {})
                return parse_crossref_work(work)
                
        except Exception as e:
            logger.error(f"Crossref DOI lookup error: {e}")
            logger.debug(traceback.format_exc())
            return None


def parse_crossref_results(data: Dict[str, Any]) -> List[SearchResult]:
        """Parse Crossref API response."""
        results = []
        
        items = data.get('message', {}).get('items', [])
        for item in items:
            result = parse_crossref_work(item)
            if result:
                results.append(result)
        
        return results
    
def parse_crossref_work(work: Dict[str, Any]) -> SearchResult:
        """Parse a single Crossref work item."""
        # Extract authors
        authors = []
        if work.get('author'):
            for author in work['author']:
                given = author.get('given', '')
                family = author.get('family', '')
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)
        
        # Extract year from published date
        year = None
        if work.get('published-print') and work['published-print'].get('date-parts'):
            year = work['published-print']['date-parts'][0][0]
        elif work.get('published-online') and work['published-online'].get('date-parts'):
            year = work['published-online']['date-parts'][0][0]
        
        # Build URL
        url = None
        if work.get('URL'):
            url = work['URL']
        elif work.get('DOI'):
            url = f"https://doi.org/{work['DOI']}"
        
        return SearchResult(
            title=' '.join(work.get('title', [''])),
            abstract=work.get('abstract', ''),
            authors=authors,
            year=year,
            doi=work.get('DOI'),
            citations=work.get('is-referenced-by-count'),
            source="Crossref",
            url=url,
            pdf_url=url,  # Crossref doesn't provide direct PDF links
            metadata={'crossref_type': work.get('type')}
        )