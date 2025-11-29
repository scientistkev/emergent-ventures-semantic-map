"""
Mercatus Center PDF scraper for extracting EV winner data.
"""

import re
from typing import List, Optional
import pdfplumber
from pathlib import Path
from .base import BaseScraper, ScrapedEntry


class MercatusScraper(BaseScraper):
    """
    Scraper for Mercatus Center PDF documents containing EV winner lists.
    
    Extracts text from PDF documents and parses structured winner lists.
    Handles various PDF formats.
    """
    
    def __init__(self, **kwargs):
        """Initialize Mercatus PDF scraper."""
        super().__init__(**kwargs)
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
        
        return '\n\n'.join(text_content)
    
    def _extract_cohort_info(self, text: str, source: str) -> dict:
        """
        Extract cohort information from PDF text.
        
        Args:
            text: PDF text content
            source: Source path/URL
            
        Returns:
            Dictionary with cohort metadata
        """
        metadata = {}
        
        # Try to extract cohort number
        cohort_match = re.search(
            r'(?:Emergent\s+Ventures?|EV)\s+(?:India\s+)?(?:cohort\s+)?#?(\d+)|(?:India\s+)?(\d+)(?:th|st|nd|rd)\s+(?:cohort|round)',
            text[:500],
            re.IGNORECASE
        )
        if cohort_match:
            cohort_num = cohort_match.group(1) or cohort_match.group(2)
            metadata['cohort_number'] = cohort_num
        
        # Check if it's India-specific
        if 'india' in text[:500].lower():
            metadata['region'] = 'India'
        
        # Extract date if present
        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text[:500])
        if date_match:
            metadata['date'] = date_match.group(1)
        
        # Extract title/header
        first_lines = text.split('\n')[:5]
        for line in first_lines:
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                metadata['title'] = line
                break
        
        return metadata
    
    def _extract_winners_from_text(self, text: str) -> List[dict]:
        """
        Extract winner information from PDF text.
        
        Args:
            text: PDF text content
            
        Returns:
            List of winner dictionaries
        """
        winners = []
        lines = text.split('\n')
        
        current_winner = None
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_winner:
                    winners.append(current_winner)
                    current_winner = None
                continue
            
            # Remove page numbers and headers/footers
            if re.match(r'^\d+$', line) or len(line) < 5:
                continue
            
            # Look for name patterns
            name_match = re.match(
                r'^([A-Z][a-zA-Z\s\.\-]+?)(?:\s*[,:—\-]\s*|\s+works?\s+on\s+)(.+)',
                line
            )
            
            if name_match:
                if current_winner:
                    winners.append(current_winner)
                
                name = name_match.group(1).strip()
                description = name_match.group(2).strip()
                
                current_winner = {
                    'name': name,
                    'description': description,
                    'raw_text': line
                }
            elif current_winner:
                # Continuation line
                current_winner['description'] += ' ' + line
                current_winner['raw_text'] += '\n' + line
            elif re.match(r'^[A-Z][a-zA-Z\s\.\-]{2,30}$', line):
                # Potential name without description yet
                current_winner = {
                    'name': line,
                    'description': '',
                    'raw_text': line
                }
        
        if current_winner:
            winners.append(current_winner)
        
        return winners
    
    def scrape(self, source: str) -> List[ScrapedEntry]:
        """
        Scrape EV winner data from PDF file.
        
        Args:
            source: Path to PDF file or URL
            
        Returns:
            List of scraped entries
        """
        entries = []
        
        # Handle URL vs file path
        if source.startswith('http'):
            # Download PDF first
            response = self._make_request(source)
            if not response:
                return entries
            
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(response.content)
                pdf_path = Path(tmp_file.name)
        else:
            pdf_path = Path(source)
            if not pdf_path.exists():
                print(f"PDF file not found: {source}")
                return entries
        
        # Extract text
        text_content = self._extract_text_from_pdf(pdf_path)
        if not text_content:
            return entries
        
        # Clean up temp file if downloaded
        if source.startswith('http') and pdf_path.exists():
            pdf_path.unlink()
        
        # Extract cohort info
        cohort_metadata = self._extract_cohort_info(text_content, source)
        
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
        
        # Extract winners
        winners = self._extract_winners_from_text(text_content)
        
        # Create entries
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
        
        # If no individual winners found, create entry with full content
        if not entries:
            entry = ScrapedEntry(
                source_url=source,
                cohort=cohort_name or "Unknown",
                raw_content=text_content,
                metadata=cohort_metadata
            )
            
            if self.validate_entry(entry):
                entries.append(entry)
        
        return entries

