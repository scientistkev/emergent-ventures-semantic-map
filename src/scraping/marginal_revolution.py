"""
Marginal Revolution blog post scraper for EV winner announcements.
"""

import re
from typing import List, Optional
from bs4 import BeautifulSoup
from .base import BaseScraper, ScrapedEntry


class MarginalRevolutionScraper(BaseScraper):
    """
    Scraper for Marginal Revolution blog posts announcing EV winners.
    
    Handles different post formats across cohorts and extracts:
    - Cohort metadata (date, cohort name/number)
    - Winner information (name, project description)
    """
    
    def __init__(self, **kwargs):
        """Initialize Marginal Revolution scraper."""
        super().__init__(rate_limit_delay=2.0, **kwargs)  # Be respectful with MR
    
    def _extract_cohort_info(self, soup: BeautifulSoup, url: str) -> dict:
        """
        Extract cohort information from post.
        
        Args:
            soup: BeautifulSoup object of the page
            url: Source URL
            
        Returns:
            Dictionary with cohort metadata
        """
        metadata = {}
        
        # Try to extract date from URL or page
        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
        if date_match:
            metadata['date'] = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        
        # Extract title for cohort name
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Try to extract cohort number from title or content
        title_text = metadata.get('title', '')
        cohort_match = re.search(
            r'(?:Emergent\s+Ventures?|EV)\s+(?:India\s+)?(?:cohort\s+)?#?(\d+)|(?:India\s+)?(\d+)(?:th|st|nd|rd)\s+(?:cohort|round)',
            title_text,
            re.IGNORECASE
        )
        if cohort_match:
            cohort_num = cohort_match.group(1) or cohort_match.group(2)
            metadata['cohort_number'] = cohort_num
        
        # Check if it's India-specific
        if 'india' in title_text.lower():
            metadata['region'] = 'India'
        
        return metadata
    
    def _extract_winners_from_post(self, soup: BeautifulSoup) -> List[dict]:
        """
        Extract winner information from post content.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            List of winner dictionaries with raw text
        """
        winners = []
        
        # Find main content area
        content = soup.find('div', class_=re.compile(r'entry-content|post-content|content'))
        if not content:
            content = soup.find('article')
        if not content:
            content = soup.find('main')
        if not content:
            content = soup
        
        text = content.get_text(separator='\n', strip=True)
        
        # Try different patterns to identify winners
        # Pattern 1: Bullet points or numbered list
        lines = text.split('\n')
        current_winner = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like a winner entry (name pattern)
            # Common patterns:
            # - "Name, description"
            # - "Name — description"
            # - "Name: description"
            # - "• Name, description"
            # - Numbered entries
            
            # Remove bullet points and numbering
            clean_line = re.sub(r'^[•\-\*]\s*', '', line)
            clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
            
            # Look for name patterns (capitalized words, may include initials)
            name_match = re.match(r'^([A-Z][a-zA-Z\s\.\-]+?)(?:\s*[,:—\-]\s*|\s+works?\s+on\s+)(.+)', clean_line)
            
            if name_match:
                # Save previous winner if exists
                if current_winner:
                    winners.append(current_winner)
                
                name = name_match.group(1).strip()
                description = name_match.group(2).strip()
                
                current_winner = {
                    'name': name,
                    'description': description,
                    'raw_text': clean_line
                }
            elif current_winner:
                # Continuation of current winner's description
                current_winner['description'] += ' ' + clean_line
                current_winner['raw_text'] += '\n' + clean_line
        
        # Don't forget last winner
        if current_winner:
            winners.append(current_winner)
        
        # If no structured winners found, try to extract from paragraphs
        if not winners:
            paragraphs = content.find_all(['p', 'li'])
            for para in paragraphs:
                para_text = para.get_text(strip=True)
                if len(para_text) > 20:  # Likely a winner entry
                    name_match = re.match(
                        r'^([A-Z][a-zA-Z\s\.\-]+?)(?:\s*[,:—\-]\s*|\s+works?\s+on\s+)(.+)',
                        para_text
                    )
                    if name_match:
                        winners.append({
                            'name': name_match.group(1).strip(),
                            'description': name_match.group(2).strip(),
                            'raw_text': para_text
                        })
        
        return winners
    
    def scrape(self, source: str) -> List[ScrapedEntry]:
        """
        Scrape EV winner announcements from Marginal Revolution URL.
        
        Args:
            source: URL of the MR blog post
            
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
            elif 'India' in cohort_name:
                cohort_name = f"EV India {cohort_name}"
        else:
            cohort_num = cohort_metadata.get('cohort_number', '')
            if cohort_num:
                cohort_name = f"EV Cohort {cohort_num}"
        
        # Extract winners
        winners = self._extract_winners_from_post(soup)
        
        # Create entries for each winner
        for winner in winners:
            entry = ScrapedEntry(
                source_url=source,
                cohort=cohort_name or "Unknown",
                raw_content=winner.get('raw_text', ''),
                metadata={
                    **cohort_metadata,
                    'winner_name': winner.get('name', ''),
                    'winner_description': winner.get('description', '')
                }
            )
            
            if self.validate_entry(entry):
                entries.append(entry)
        
        # If no individual winners found, create one entry with full content
        if not entries:
            content = soup.get_text(separator='\n', strip=True)
            entry = ScrapedEntry(
                source_url=source,
                cohort=cohort_name or "Unknown",
                raw_content=content,
                metadata=cohort_metadata
            )
            
            if self.validate_entry(entry):
                entries.append(entry)
        
        return entries

