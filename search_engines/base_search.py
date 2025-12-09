from typing import List
import aiohttp
import asyncio
import logging
import os
from interfaces.interfaces import SearchResult

logfilename = os.path.join("logs", "arxiv.log")

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


async def base_search(self, url: str, query: str, max_results: int) -> List[SearchResult]:
    """Search base."""
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
        
    try:
        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"base search API error: {response.status}")
                return []
                
            content = await response.text()
            return self._parse_arxiv_results(content)
                
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        return []