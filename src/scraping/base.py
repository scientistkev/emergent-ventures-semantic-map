"""
Base scraper architecture with common utilities.
"""

import time
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse


@dataclass
class ScrapedEntry:
    """Represents a single scraped entry with raw content."""
    source_url: str
    cohort: str
    raw_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


class BaseScraper(ABC):
    """
    Abstract base class for all scrapers.
    
    Provides common functionality:
    - Rate limiting
    - Error handling and retries
    - Robots.txt checking
    - Output format standardization
    """
    
    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        respect_robots_txt: bool = True,
        user_agent: str = "EV-Semantic-Map-Bot/1.0"
    ):
        """
        Initialize base scraper.
        
        Args:
            rate_limit_delay: Seconds to wait between requests
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            respect_robots_txt: Whether to check robots.txt before scraping
            user_agent: User agent string for requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.respect_robots_txt = respect_robots_txt
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        self.last_request_time = 0.0
        self._robots_cache = {}
    
    def _check_robots_txt(self, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            True if allowed, False otherwise
        """
        if not self.respect_robots_txt:
            return True
        
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Check cache
        if base_url in self._robots_cache:
            rp = self._robots_cache[base_url]
        else:
            rp = RobotFileParser()
            robots_url = urljoin(base_url, '/robots.txt')
            try:
                rp.set_url(robots_url)
                rp.read()
                self._robots_cache[base_url] = rp
            except Exception:
                # If we can't read robots.txt, allow by default
                return True
        
        return rp.can_fetch(self.user_agent, url)
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(
        self,
        url: str,
        method: str = 'GET',
        **kwargs
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retries and error handling.
        
        Args:
            url: URL to request
            method: HTTP method
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object or None if failed
        """
        # Check robots.txt
        if not self._check_robots_txt(url):
            print(f"URL blocked by robots.txt: {url}")
            return None
        
        # Rate limiting
        self._rate_limit()
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                kwargs.setdefault('timeout', self.timeout)
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    return None
        
        return None
    
    @abstractmethod
    def scrape(self, source: str) -> List[ScrapedEntry]:
        """
        Scrape data from a source.
        
        Args:
            source: Source identifier (URL, file path, etc.)
            
        Returns:
            List of scraped entries
        """
        pass
    
    def validate_entry(self, entry: ScrapedEntry) -> bool:
        """
        Validate a scraped entry.
        
        Args:
            entry: Scraped entry to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not entry.source_url:
            return False
        if not entry.raw_content or len(entry.raw_content.strip()) < 10:
            return False
        if not entry.cohort:
            return False
        return True
    
    def save_results(
        self,
        entries: List[ScrapedEntry],
        output_path: Path
    ):
        """
        Save scraped entries to JSON file.
        
        Args:
            entries: List of scraped entries
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        data = []
        for entry in entries:
            entry_dict = {
                'source_url': entry.source_url,
                'cohort': entry.cohort,
                'raw_content': entry.raw_content,
                'metadata': entry.metadata,
                'extracted_data': entry.extracted_data,
                'timestamp': entry.timestamp
            }
            data.append(entry_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(entries)} entries to {output_path}")

