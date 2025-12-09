from typing import List
import aiohttp
import asyncio
import logging
import os
import traceback
import xml.etree.ElementTree as ET
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

session = aiohttp.ClientSession()

async def search_arxiv(url: str, query: str, max_results: int) -> List[SearchResult]:
    """Search arXiv."""
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
        
    try:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"arXiv API error: {response.status}")
                return []
                
            content = await response.text()
            return _parse_arxiv_results(content)
                
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        return []
    
def _extract_year_from_date(date_string: str) -> int:
    """Extract year from date string (format: YYYY-MM-DDTHH:MM:SSZ)."""
    try:
        if date_string and len(date_string) >= 4:
            return int(date_string[:4])
        return 0
    except (ValueError, TypeError):
        return 0

def _parse_arxiv_results(xml_content: str) -> List[SearchResult]:
        """Parse arXiv XML response."""
        results = []
        logger.debug("Parsing arXiv results with XML content length: %d", len(xml_content)  )
        logger.debug(f"XML Content: {xml_content}...")  # Log first 500 characters for inspection
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('.//atom:entry', ns):
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                authors = entry.findall('.//atom:author/atom:name', ns)
                published = entry.find('atom:published', ns)
                pdf_link = entry.find(".//atom:link[@title='pdf']", ns)
                
                # Extract arXiv ID from entry ID
                entry_id = entry.find('atom:id', ns)
                arxiv_id = None
                if entry_id is not None:
                    arxiv_id = entry_id.text.split('/')[-1]
                
                result = SearchResult(
                    title=title.text.strip() if title is not None else "",
                    abstract=summary.text.strip() if summary is not None else "",
                    authors=[author.text for author in authors],
                    year=_extract_year_from_date(published.text if published is not None else ""),
                    source="arXiv",
                    url=entry_id.text if entry_id is not None else "",
                    pdf_url=pdf_link.get('href') if pdf_link is not None else None,
                    metadata={'arxiv_id': arxiv_id}
                )
                
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error parsing arXiv results: {e}")
            logger.error(traceback.format_exc())
            
        
        return results