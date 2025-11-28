"""
Generic web scraper for press announcements and other HTML sources.
"""

from typing import List, Optional
from bs4 import BeautifulSoup
from .base import BaseScraper, ScrapedEntry


class WebScraper(BaseScraper):
    """
    Generic web scraper for press announcements and other HTML sources.
    
    Extracts structured content from HTML and handles various website formats.
    """
    
    def __init__(self, **kwargs):
        """Initialize web scraper."""
        super().__init__(rate_limit_delay=1.5, **kwargs)
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from HTML page.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Main content text
        """
        # Try common content selectors
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            'main',
            '.content',
            '#content',
            '.main-content'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                return content.get_text(separator='\n', strip=True)
        
        # Fallback: remove script and style, get body text
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        body = soup.find('body')
        if body:
            return body.get_text(separator='\n', strip=True)
        
        return soup.get_text(separator='\n', strip=True)
    
    def _extract_cohort_info(self, soup: BeautifulSoup, url: str) -> dict:
        """
        Extract cohort information from page.
        
        Args:
            soup: BeautifulSoup object
            url: Source URL
            
        Returns:
            Dictionary with cohort metadata
        """
        import re
        metadata = {}
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            metadata['description'] = meta_desc['content']
        
        # Extract from URL
        if 'india' in url.lower():
            metadata['region'] = 'India'
        
        # Try to find cohort number in title or first paragraph
        title_text = metadata.get('title', '')
        first_p = soup.find('p')
        if first_p:
            title_text += ' ' + first_p.get_text()
        
        cohort_match = re.search(
            r'(?:Emergent\s+Ventures?|EV)\s+(?:India\s+)?(?:cohort\s+)?#?(\d+)|(?:India\s+)?(\d+)(?:th|st|nd|rd)\s+(?:cohort|round)',
            title_text,
            re.IGNORECASE
        )
        if cohort_match:
            cohort_num = cohort_match.group(1) or cohort_match.group(2)
            metadata['cohort_number'] = cohort_num
        
        return metadata
    
    def scrape(self, source: str) -> List[ScrapedEntry]:
        """
        Scrape content from web page.
        
        Args:
            source: URL of the webpage
            
        Returns:
            List of scraped entries
        """
        entries = []
        
        # Fetch the page
        response = self._make_request(source)
        if not response:
            return entries
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Extract cohort metadata
        cohort_metadata = self._extract_cohort_info(soup, source)
        
        # Determine cohort name
        cohort_name = cohort_metadata.get('title', '')
        if cohort_metadata.get('region') == 'India':
            cohort_num = cohort_metadata.get('cohort_number', '')
            if cohort_num:
                cohort_name = f"EV India {cohort_num}"
        else:
            cohort_num = cohort_metadata.get('cohort_number', '')
            if cohort_num:
                cohort_name = f"EV Cohort {cohort_num}"
        
        # Extract main content
        content = self._extract_main_content(soup)
        
        # Create entry
        entry = ScrapedEntry(
            source_url=source,
            cohort=cohort_name or "Unknown",
            raw_content=content,
            metadata=cohort_metadata
        )
        
        if self.validate_entry(entry):
            entries.append(entry)
        
        return entries

