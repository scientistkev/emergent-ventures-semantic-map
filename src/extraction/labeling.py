"""
Automated labeling for domains, categories, and geographies using LLM.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


class AutoLabeler:
    """
    Automated labeling for:
    - Domain tagging using existing vocabulary
    - Category classification
    - Geography normalization
    - Outcome prediction (startup, research, content, policy)
    """
    
    def __init__(
        self,
        vocab_dir: Optional[Path] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        """
        Initialize auto-labeler.
        
        Args:
            vocab_dir: Directory containing vocabulary files
            model: OpenAI model to use
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.model = model
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Load vocabularies
        vocab_dir = vocab_dir or Path('data/vocabularies')
        self.domains = self._load_json(vocab_dir / 'domains.json')
        self.categories = self._load_json(vocab_dir / 'categories.json')
        self.locations = self._load_json(vocab_dir / 'locations.json')
        self.domain_mapping = self._load_json(vocab_dir / 'domain_mapping.json')
    
    def _load_json(self, path: Path) -> Any:
        """Load JSON file, return empty list/dict if not found."""
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return [] if 'domain_mapping' not in str(path) else {}
    
    def label_domains(
        self,
        text: str,
        existing_domains: Optional[List[str]] = None
    ) -> List[str]:
        """
        Automatically label domains using existing vocabulary.
        
        Args:
            text: Text to extract domains from
            existing_domains: Already extracted domains (will be validated)
            
        Returns:
            List of normalized domain labels
        """
        # Use existing domains if provided and valid
        if existing_domains:
            normalized = []
            for domain in existing_domains:
                domain_lower = domain.lower().strip()
                # Map to canonical form
                canonical = self.domain_mapping.get(domain_lower, domain_lower)
                if canonical in self.domains:
                    normalized.append(canonical)
                elif canonical not in normalized:
                    normalized.append(canonical)
            
            if normalized:
                return normalized
        
        # Use LLM to extract additional domains from text
        prompt = f"""Extract relevant domains/fields from the following text about an Emergent Ventures winner.

Available domain vocabulary:
{json.dumps(self.domains, indent=2)}

Text:
{text[:2000]}

Return a JSON array of domain labels that best match the content. Use only domains from the vocabulary list above, or suggest new ones if needed (limit to 5 domains max).

Return format: {{"domains": ["domain1", "domain2", ...]}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a domain classification assistant. Return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            domains = result.get('domains', [])
            
            # Normalize domains
            normalized = []
            for domain in domains:
                domain_lower = str(domain).lower().strip()
                canonical = self.domain_mapping.get(domain_lower, domain_lower)
                if canonical not in normalized:
                    normalized.append(canonical)
            
            return normalized[:5]  # Limit to 5 domains
        
        except Exception as e:
            print(f"Error in domain labeling: {e}")
            return existing_domains or []
    
    def classify_category(
        self,
        text: str,
        existing_category: Optional[str] = None
    ) -> str:
        """
        Classify project category.
        
        Args:
            text: Text describing the project
            existing_category: Already extracted category (will be validated)
            
        Returns:
            Category label
        """
        valid_categories = {
            'startup', 'research', 'content', 'policy',
            'career development', 'grant', 'project', 'other'
        }
        
        # Validate existing category
        if existing_category:
            existing_lower = existing_category.lower().strip()
            if existing_lower in valid_categories:
                return existing_lower
        
        # Use LLM to classify
        prompt = f"""Classify the following Emergent Ventures project into one of these categories:
- startup: Building a company/product
- research: Academic or scientific research
- content: Creating content (writing, media, etc.)
- policy: Policy work or advocacy
- career development: Personal career/skill development
- grant: General grant funding
- project: Specific project work
- other: Other/unknown

Text:
{text[:1500]}

Return JSON: {{"category": "category_name", "confidence": 0.0-1.0}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a category classification assistant. Return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            category = result.get('category', 'other').lower().strip()
            
            # Validate category
            if category not in valid_categories:
                return existing_category or 'other'
            
            return category
        
        except Exception as e:
            print(f"Error in category classification: {e}")
            return existing_category or 'other'
    
    def normalize_location(
        self,
        location: str,
        text: Optional[str] = None
    ) -> str:
        """
        Normalize location/geography.
        
        Args:
            location: Location string to normalize
            text: Optional context text to help identify location
            
        Returns:
            Normalized location string
        """
        if not location:
            # Try to extract from text
            if text:
                prompt = f"""Extract the geographic location/country from this text:

{text[:1000]}

Return JSON: {{"location": "country or region"}}"""

                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a location extraction assistant. Return valid JSON."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"}
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    location = result.get('location', '')
                except Exception as e:
                    print(f"Error extracting location: {e}")
        
        if not location:
            return ""
        
        # Normalize location (handle multi-location strings)
        location_parts = [part.strip() for part in str(location).split('/')]
        primary_location = location_parts[0] if location_parts else ""
        
        # Check against known locations
        location_lower = primary_location.lower()
        for known_loc in self.locations:
            if location_lower == known_loc.lower():
                return known_loc
        
        return primary_location
    
    def predict_outcome(
        self,
        entry: Dict[str, Any]
    ) -> str:
        """
        Predict likely outcome (startup, research, content, policy).
        
        Args:
            entry: Entry dictionary with project information
            
        Returns:
            Predicted outcome label
        """
        text_parts = [
            entry.get('project_name', ''),
            entry.get('project_description', ''),
            entry.get('category', '')
        ]
        text = ' '.join([p for p in text_parts if p])
        
        prompt = f"""Based on this Emergent Ventures project, predict the likely outcome:

Project: {entry.get('project_name', 'N/A')}
Description: {entry.get('project_description', 'N/A')}
Category: {entry.get('category', 'N/A')}

Predict the most likely outcome:
- startup: Will likely become a startup/company
- research: Will produce research/publications
- content: Will create content/media
- policy: Will influence policy
- unknown: Cannot determine

Return JSON: {{"outcome": "outcome_name", "confidence": 0.0-1.0}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an outcome prediction assistant. Return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('outcome', 'unknown')
        
        except Exception as e:
            print(f"Error in outcome prediction: {e}")
            return 'unknown'
    
    def label_entry(
        self,
        entry: Dict[str, Any],
        apply_all: bool = True
    ) -> Dict[str, Any]:
        """
        Apply all labeling functions to an entry.
        
        Args:
            entry: Entry dictionary to label
            apply_all: If True, apply all labeling functions
            
        Returns:
            Entry with additional labels
        """
        labeled_entry = entry.copy()
        
        # Build context text
        text_parts = [
            entry.get('project_name', ''),
            entry.get('project_description', ''),
            entry.get('raw_content', '')
        ]
        context_text = ' '.join([p for p in text_parts if p])
        
        if apply_all:
            # Label domains
            existing_domains = entry.get('domains', [])
            labeled_entry['domains'] = self.label_domains(
                context_text,
                existing_domains
            )
            
            # Classify category
            existing_category = entry.get('category')
            labeled_entry['category'] = self.classify_category(
                context_text,
                existing_category
            )
            
            # Normalize location
            existing_location = entry.get('location')
            labeled_entry['location'] = self.normalize_location(
                existing_location or '',
                context_text
            )
            
            # Predict outcome (optional, add as metadata)
            outcome = self.predict_outcome(entry)
            if '_metadata' not in labeled_entry:
                labeled_entry['_metadata'] = {}
            labeled_entry['_metadata']['predicted_outcome'] = outcome
        
        return labeled_entry

