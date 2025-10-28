# utils/file_watcher.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from typing import Callable
import time

from utils.logger import log
from config import settings


class PDFFileHandler(FileSystemEventHandler):
    """Handle file system events for PDF files"""
    
    def __init__(self, callback: Callable):
        self.callback = callback
        self.processed_files = set()
    
    def on_created(self, event):
        """Handle file creation event"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's a PDF and not already processed
        if file_path.suffix.lower() == '.pdf' and file_path not in self.processed_files:
            log.info(f"New PDF detected: {file_path.name}")
            self.processed_files.add(file_path)
            
            # Wait a bit to ensure file is fully written
            time.sleep(1)
            
            # Call the callback
            try:
                self.callback(file_path)
            except Exception as e:
                log.error(f"Error processing new file {file_path}: {e}")


class FileWatcher:
    """Watch a directory for new PDF files"""
    
    def __init__(self, callback: Callable, watch_path: Path = None):
        if watch_path is None:
            watch_path = settings.DOWNLOADS_FOLDER
        
        self.watch_path = watch_path
        self.callback = callback
        self.observer = Observer()
        self.event_handler = PDFFileHandler(callback)
        
        log.info(f"FileWatcher initialized for: {watch_path}")
    
    def start(self):
        """Start watching the directory"""
        self.observer.schedule(
            self.event_handler, 
            str(self.watch_path), 
            recursive=False
        )
        self.observer.start()
        log.info(f"Started watching directory: {self.watch_path}")
    
    def stop(self):
        """Stop watching the directory"""
        self.observer.stop()
        self.observer.join()
        log.info("Stopped watching directory")