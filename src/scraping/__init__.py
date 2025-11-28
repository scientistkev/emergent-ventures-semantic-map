"""
Scraping module for collecting EV winner data from various sources.
"""

from .base import BaseScraper, ScrapedEntry
from .marginal_revolution import MarginalRevolutionScraper
from .mercatus import MercatusScraper
from .web import WebScraper
from .orchestrator import ScraperOrchestrator

__all__ = [
    'BaseScraper',
    'ScrapedEntry',
    'MarginalRevolutionScraper',
    'MercatusScraper',
    'WebScraper',
    'ScraperOrchestrator',
]

