
from typing import List
import aiohttp
import asyncio
import logging
import os
import xml.etree.ElementTree as ET
import re
from bs4 import BeautifulSoup
from interfaces.interfaces import SearchResult

logfilename = os.path.join("logs", "duckduckgo_academic.log")

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

# Headers for web scraping
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

async def search_duckduckgo_academic(url: str, query: str, max_results: int) -> List[SearchResult]:
        """Search DuckDuckGo for academic papers"""
        results = []
        
        try:
            # Add academic keywords to improve results
            academic_query = f"{query} research paper pdf site:arxiv.org OR site:researchgate.net OR site:academia.edu"
            
            url = "https://html.duckduckgo.com/html/"
            data = {
                'q': academic_query,
                'b': '',
                'kl': 'us-en'
            }
            
            async with session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    for i, result in enumerate(soup.find_all('div', class_='result')):
                        if i >= max_results:
                            break
                        
                        try:
                            # Title and URL
                            logger.debug(f"result: {result}")
                            title_elem = result.find('a', class_='result__a')
                            if not title_elem:
                                continue
                            
                            title = title_elem.text
                            paper_url = title_elem.get('href', '')
                            
                            # Snippet
                            snippet_elem = result.find('a', class_='result__snippet')
                            snippet = snippet_elem.text if snippet_elem else ''
                            
                            # Try to extract year from snippet or title
                            year_match = re.search(r'(19|20)\d{2}', snippet + ' ' + title)
                            year = int(year_match.group()) if year_match else None
                            
                            # Check if it's a PDF
                            pdf_url = paper_url # if paper_url.endswith('.pdf') else None
                            
                            result = SearchResult(
                                title=title,
                                url=paper_url,
                                abstract=snippet,
                                year=year,
                                source="duckduckgo",
                                pdf_url=pdf_url,
                                #authors=[],  # DuckDuckGo does not provide author info
                                #metadata={'duckduckgo_id': paper.get('id')}
                            )
                
                            results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Error parsing DuckDuckGo result: {e}")
                            continue
                
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return results