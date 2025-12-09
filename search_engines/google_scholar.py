from typing import List, Dict, Any
import aiohttp
import asyncio
import logging
import traceback
import os
from bs4 import BeautifulSoup
import re
import xml.etree.ElementTree as ET
from interfaces.interfaces import SearchResult

logfilename = os.path.join("logs", "google_scholar.log")

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

async def search_google_scholar(url: str, query: str, max_results: int) -> List[SearchResult]:
        """Search Google Scholar (web scraping)"""
        results = []
        
        try:
            # Google Scholar URL
            url = url
            params = {
                'q': query,
                'hl': 'en',
                'num': max_results
            }
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Parse results
                    for result_div in soup.find_all('div', {'data-lid': True}):
                        try:
                            # Title and URL
                            title_elem = result_div.find('h3', class_='gs_rt')
                            if not title_elem:
                                continue
                            
                            link_elem = title_elem.find('a')
                            if link_elem:
                                title = link_elem.text
                                paper_url = link_elem.get('href', '')
                            else:
                                title = title_elem.text
                                paper_url = ''
                            
                            # Clean title
                            title = re.sub(r'\[.*?\]', '', title).strip()
                            
                            # Authors and year
                            info_elem = result_div.find('div', class_='gs_a')
                            authors = []
                            year = None
                            venue = None
                            
                            if info_elem:
                                info_text = info_elem.text
                                # Extract authors (before first dash)
                                parts = info_text.split(' - ')
                                if parts:
                                    authors = [a.strip() for a in parts[0].split(',')]
                                # Extract year
                                year_match = re.search(r'(19|20)\d{2}', info_text)
                                if year_match:
                                    year = int(year_match.group())
                                # Extract venue
                                if len(parts) > 1:
                                    venue = parts[1].strip()
                            
                            # Abstract
                            abstract_elem = result_div.find('div', class_='gs_rs')
                            abstract = abstract_elem.text if abstract_elem else ''
                            
                            # PDF link
                            pdf_url = None
                            pdf_elem = result_div.find('div', class_='gs_or_ggsm')
                            if pdf_elem:
                                pdf_link = pdf_elem.find('a')
                                if pdf_link:
                                    pdf_url = pdf_link.get('href', '')
                            
                            # Citations
                            citations = None
                            cite_elem = result_div.find('a', string=re.compile(r'Cited by \d+'))
                            if cite_elem:
                                cite_match = re.search(r'Cited by (\d+)', cite_elem.text)
                                if cite_match:
                                    citations = int(cite_match.group(1))
                            
                            results.append(SearchResult(
                                title=title,
                                url=paper_url,
                                source="Google Scholar",
                                pdf_url=pdf_url,
                                authors=authors,  # Limit authors
                                year=year,
                                citations=citations,
                                abstract=abstract,
                                metadata={'venue': venue}
                            ))
                            
                        except Exception as e:
                            logger.error(f"Error parsing Google Scholar result: {e}")
                            continue
                
        except Exception as e:
            logger.error(f"Google Scholar search error: {e}")
            logger.debug(traceback.format_exc())
        
        return results