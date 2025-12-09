"""
PDF download service for handling paper downloads and file management.

This service manages downloading PDF files from search results, organizing them
in a structured directory layout, and providing proper error handling.
"""
import asyncio
import logging
import traceback
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse
import aiohttp
import aiofiles
from datetime import datetime
import hashlib
import re

from interfaces.interfaces import IService, SearchResult

logfilename = os.path.join("logs", "pdf_download_service.log")

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


class PDFDownloadService(IService):
    """Service for downloading PDF files from search results."""

    def __init__(self, download_directory: Union[str, Path] = "./downloads"):
        self._name = "pdf_download_service"
        self.download_directory = Path(download_directory)
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Download configuration
        self.max_file_size = 100 * 1024 * 1024  # 100MB limit
        self.supported_extensions = {'.pdf', '.PDF'}

        # Statistics tracking
        self.stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped_downloads': 0
        }

    @property
    def name(self) -> str:
        return self._name

    async def initialize(self) -> None:
        """Initialize the download service."""
        if self._initialized:
            # Check if we're in a different event loop
            current_loop = asyncio.get_running_loop()
            if self._loop is not current_loop:
                logger.warning(
                    "Event loop changed, reinitializing session..."
                )
                # Don't try to close the old session - it's tied to the old loop
                # Just abandon it and create a new one
                self.session = None
                self._initialized = False
                self._loop = None
            else:
                return

        logger.info("Initializing PDF download service")

        # Store reference to current event loop
        self._loop = asyncio.get_running_loop()

        # Create download directory structure
        await self._setup_directories()

        # Initialize HTTP session with explicit timeout configuration
        # Set timeout to None to avoid timeout-related errors
        connector = aiohttp.TCPConnector(
            limit=10,  # Total connection pool size
            limit_per_host=3,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True
        )

        # Create timeout with all values set to None (no timeout)
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=None,
            sock_read=None,
            sock_connect=None
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Research Assistant PDF Downloader/2.0',
                'Accept': 'application/pdf,*/*;q=0.8'
            }
        )

        self._initialized = True
        logger.info("PDF download service initialized")

    async def shutdown(self) -> None:
        """Shutdown the download service."""
        if not self._initialized:
            return

        logger.info("Shutting down PDF download service")

        if self.session:
            await self.session.close()
            self.session = None

        self._initialized = False
        self._loop = None
        logger.info("PDF download service shutdown complete")

    async def download_pdf(
        self,
        search_result: SearchResult,
        subfolder: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download a single PDF from a search result.

        Args:
            search_result: The search result containing PDF URL
            subfolder: Optional subfolder to organize downloads

        Returns:
            Dictionary with download results and metadata
        """
        if not self._initialized:
            await self.initialize()

        if not search_result.pdf_url:
            return {
                'success': False,
                'reason': 'no_pdf_url',
                'message': 'No PDF URL found in search result'
            }

        self.stats['total_downloads'] += 1

        try:
            # Generate filename from title and URL
            filename = self._generate_filename(search_result)

            # Determine download path
            download_path = self._get_download_path(filename, subfolder)

            # Check if file already exists
            if download_path.exists():
                logger.info(f"PDF already exists: {download_path}")
                self.stats['skipped_downloads'] += 1
                return {
                    'success': True,
                    'reason': 'already_exists',
                    'path': str(download_path),
                    'message': f'PDF already downloaded: {filename}'
                }

            # await asyncio.sleep(1)

            for i in range(1000):  # Dummy loop to illustrate the fix
                pass

            # Download the PDF
            result = await self._download_file(search_result.pdf_url, download_path)
            # await asyncio.sleep(1)  # Be polite and avoid rapid requests
            for i in range(1000):  # Dummy loop to illustrate the fix
                pass

            if result['success']:
                self.stats['successful_downloads'] += 1

                # Add metadata
                await self._save_metadata(search_result, download_path)

                return {
                    'success': True,
                    'path': str(download_path),
                    'size': result.get('size', 0),
                    'message': f'Successfully downloaded: {filename}'
                }
            else:
                self.stats['failed_downloads'] += 1
                return result

        except Exception as e:
            self.stats['failed_downloads'] += 1
            logger.error(f"Error downloading PDF: {e}")
            logger.debug(traceback.format_exc())
            return {
                'success': False,
                'reason': 'download_error',
                'message': f'Download failed: {str(e)} - {traceback.format_exc()}'
            }

    async def download_batch(
        self,
        search_results: List[SearchResult],
        subfolder: Optional[str] = None,
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Download multiple PDFs concurrently.

        Args:
            search_results: List of search results to download
            subfolder: Optional subfolder for organization
            max_concurrent: Maximum concurrent downloads

        Returns:
            Dictionary with batch download results
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Starting batch download of {len(search_results)} PDFs")

        # Filter results with PDF URLs
        downloadable_results = [r for r in search_results if r.pdf_url]

        if not downloadable_results:
            return {
                'total_requested': len(search_results),
                'downloadable': 0,
                'results': [],
                'message': 'No PDFs available for download'
            }

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_semaphore(result: SearchResult) -> Dict[str, Any]:
            async with semaphore:
                try:
                    download_result = await self.download_pdf(result, subfolder)
                    for i in range(1000):  # Dummy loop to illustrate the fix
                        pass
                    download_result['title'] = result.title
                    download_result['url'] = result.pdf_url
                except Exception as e:
                    logger.error(f"Error downloading PDF: {e}")
                    download_result = {
                        'success': False,
                        'title': result.title,
                        'url': result.pdf_url,
                        'reason': 'exception',
                        'message': f'{str(e)} - {traceback.format_exc()}'
                    }
                
                return download_result

        # Execute downloads
        # Create tasks explicitly to ensure proper task context for aiohttp
        tasks = [asyncio.create_task(download_with_semaphore(result))
                 for result in downloadable_results]
        logger.info("Executing download tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Download tasks completed.")

        # Process results
        successful = 0
        failed = 0
        skipped = 0

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'title': downloadable_results[i].title,
                    'url': downloadable_results[i].pdf_url,
                    'reason': 'exception',
                    'message': str(result)
                })
                failed += 1
            else:
                processed_results.append(result)
                if result['success']:
                    if result.get('reason') == 'already_exists':
                        skipped += 1
                    else:
                        successful += 1
                else:
                    failed += 1

        logger.info(
            f"Batch download complete: {successful} successful, {failed} failed, {skipped} skipped")

        return {
            'total_requested': len(search_results),
            'downloadable': len(downloadable_results),
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'results': processed_results
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        return {
            **self.stats,
            'download_directory': str(self.download_directory),
            'initialized': self._initialized
        }

    async def _setup_directories(self) -> None:
        """Create the download directory structure."""
        try:
            # Main downloads directory
            self.download_directory.mkdir(parents=True, exist_ok=True)

            # Subdirectories for organization
            (self.download_directory / "metadata").mkdir(exist_ok=True)
            (self.download_directory / "by_date").mkdir(exist_ok=True)
            (self.download_directory / "by_source").mkdir(exist_ok=True)

            logger.info(
                f"Download directories created: {self.download_directory}")

        except Exception as e:
            logger.error(f"Failed to create download directories: {e}")
            logger.error(traceback.format_exc())
            raise

    def _generate_filename(self, search_result: SearchResult) -> str:
        """Generate a safe filename from search result."""
        # Start with the title
        if search_result.title:
            base_name = search_result.title
        else:
            # Fallback to URL-based name
            parsed_url = urlparse(search_result.pdf_url)
            base_name = Path(parsed_url.path).stem or "unknown_paper"

        # Clean the filename
        # Remove/replace problematic characters
        base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        base_name = re.sub(r'\s+', '_', base_name.strip())

        # Limit length
        if len(base_name) > 100:
            base_name = base_name[:97] + "..."

        # Add unique identifier if needed
        if search_result.doi:
            # Use DOI hash for uniqueness (using sha256 for security)
            doi_hash = hashlib.sha256(
                search_result.doi.encode()).hexdigest()[:8]
            base_name += f"_{doi_hash}"
        elif search_result.pdf_url:
            # Use URL hash for uniqueness (using sha256 for security)
            url_hash = hashlib.sha256(
                search_result.pdf_url.encode()
            ).hexdigest()[:8]
            base_name += f"_{url_hash}"

        return f"{base_name}.pdf"

    def _get_download_path(
        self, filename: str, subfolder: Optional[str] = None
    ) -> Path:
        """Get the full download path for a file."""
        if subfolder:
            # Create subfolder if it doesn't exist
            folder_path = self.download_directory / subfolder
            folder_path.mkdir(parents=True, exist_ok=True)
            return folder_path / filename
        else:
            return self.download_directory / filename

    async def _download_file(self, url: str, path: Path) -> Dict[str, Any]:
        """Download a file from URL to path."""
        try:
            logger.info(f"Downloading: {url}")

            # Ensure session exists and is valid for current event loop
            current_loop = asyncio.get_running_loop()
            if (not self.session or 
                self.session.closed or 
                self._loop is not current_loop):
                logger.warning(
                    "Session invalid or event loop changed, reinitializing..."
                )
                await self.initialize()

            # Apply timeout per-request using asyncio.timeout
            try:
                # async with asyncio.timeout(60):  # 60 second total timeout
                async with self.session.get(url) as response:
                    if response.status != 200:
                        return {
                            'success': False,
                            'reason': 'http_error',
                            'message': f'HTTP {response.status}: {response.reason}'
                        }

                    # Check content type
                    content_type = response.headers.get(
                        'content-type', '').lower()
                    is_pdf = ('pdf' in content_type or
                              url.lower().endswith('.pdf'))
                    if not is_pdf:
                        logger.warning(
                            f"Unexpected content type: {content_type}")

                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_file_size:
                        return {
                            'success': False,
                            'reason': 'file_too_large',
                            'message': (f'File size ({content_length} bytes) '
                                        f'exceeds limit')
                        }

                    # Download content
                    total_size = 0
                    async with aiofiles.open(path, 'wb') as file:
                        async for chunk in response.content.iter_chunked(8192):
                            total_size += len(chunk)
                            if total_size > self.max_file_size:
                                # Clean up partial file
                                await file.close()
                                path.unlink(missing_ok=True)
                                return {
                                    'success': False,
                                    'reason': 'file_too_large',
                                    'message': ('File size exceeds limit '
                                                'during download')
                                }
                            await file.write(chunk)

                    logger.info(f"Downloaded {total_size} bytes to {path}")
                    return {
                        'success': True,
                        'size': total_size
                    }
            except asyncio.TimeoutError:
                return {
                    'success': False,
                    'reason': 'timeout',
                    'message': 'Download timeout'
                }

        except asyncio.TimeoutError:
            return {
                'success': False,
                'reason': 'timeout',
                'message': 'Download timeout'
            }
        except Exception as e:
            # Clean up partial file on error
            path.unlink(missing_ok=True)
            logger.error(f"Download failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'reason': 'download_error',
                'message': f"{str(e)}-{traceback.format_exc()}"
            }

    async def _save_metadata(
        self, search_result: SearchResult, pdf_path: Path
    ) -> None:
        """Save metadata for downloaded PDF."""
        try:
            metadata = {
                'title': search_result.title,
                'authors': search_result.authors,
                'abstract': search_result.abstract,
                'year': search_result.year,
                'doi': search_result.doi,
                'source': search_result.source,
                'pdf_url': search_result.pdf_url,
                'url': search_result.url,
                'citations': search_result.citations,
                'download_timestamp': datetime.now().isoformat(),
                'local_path': str(pdf_path),
                'file_size': (pdf_path.stat().st_size
                              if pdf_path.exists() else None)
            }

            # Save as JSON file
            metadata_path = (self.download_directory / "metadata" /
                             f"{pdf_path.stem}.json")
            async with aiofiles.open(
                metadata_path, 'w', encoding='utf-8'
            ) as f:
                import json
                content = json.dumps(metadata, indent=2, ensure_ascii=False)
                await f.write(content)

            logger.debug(f"Metadata saved: {metadata_path}")

        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
            logger.warning(traceback.format_exc())
