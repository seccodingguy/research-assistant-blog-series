# utils/logger.py
from loguru import logger
from pathlib import Path
import sys
from config import settings

def setup_logger():
    """Configure logger with file and console output"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with rich formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True
    )
    
    # File handler
    if settings.LOG_FILE:
        logger.add(
            settings.LOG_FILE,
            rotation="500 MB",
            retention="10 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL
        )
    
    return logger

log = setup_logger()