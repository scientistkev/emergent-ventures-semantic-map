"""
LLM-powered extraction module to convert unstructured text into structured JSON entries.
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


@dataclass
class ExtractionResult:
    """Result of LLM extraction with confidence score."""
    data: Dict[str, Any]
    confidence_score: float
    raw_response: str
    error: Optional[str] = None


class LLMExtractor:
    """
    Use OpenAI API to extract structured fields from raw text.
    
    Handles:
    - Prompt engineering for reliable extraction
    - Missing/incomplete information gracefully
    - Extraction of: name, age, location, education, project_name, project_description, domains, category, cohort
    """
    
    EXTRACTION_PROMPT = """You are extracting structured information about an Emergent Ventures winner from unstructured text.

Extract the following information:
- name: Full name of the winner
- age: Age if mentioned (integer or null)
- location: Geographic location/country
- education: Educational background or institution
- project_name: Name/title of their project
- project_description: Description of what they're working on
- domains: List of relevant domains/fields (e.g., ["AI", "biotech", "education"])
- category: Type of project - one of: "startup", "research", "content", "policy", "career development", "other"
- cohort: Cohort name/number if mentioned

Text to extract from:
{text}

Return a JSON object with the extracted fields. Use null for missing information. Be conservative - only extract information that is clearly stated in the text.

Example output format:
{{
  "name": "John Doe",
  "age": 25,
  "location": "India",
  "education": "PhD in Computer Science, MIT",
  "project_name": "AI Education Platform",
  "project_description": "Building an AI-powered platform to personalize education",
  "domains": ["artificial intelligence", "education"],
  "category": "startup",
  "cohort": "EV India 13"
}}"""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM extractor.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for generation (0.0 for deterministic)
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
    
    def extract(
        self,
        text: str,
        cohort: Optional[str] = None,
        max_retries: int = 3
    ) -> ExtractionResult:
        """
        Extract structured data from unstructured text.
        
        Args:
            text: Raw text to extract from
            cohort: Cohort name if known (will be added if not in text)
            max_retries: Maximum retry attempts
            
        Returns:
            ExtractionResult with extracted data and confidence score
        """
        # Prepare prompt
        prompt = self.EXTRACTION_PROMPT.format(text=text[:3000])  # Limit text length
        
        # Add cohort hint if provided
        if cohort and 'cohort' not in text.lower():
            prompt += f"\n\nNote: This entry is from cohort: {cohort}"
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data extraction assistant. Always return valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                raw_response = response.choices[0].message.content
                
                # Parse JSON response
                try:
                    extracted_data = json.loads(raw_response)
                except json.JSONDecodeError as e:
                    # Try to extract JSON from response if wrapped in markdown
                    import re
                    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                    if json_match:
                        extracted_data = json.loads(json_match.group(0))
                    else:
                        raise ValueError(f"Failed to parse JSON: {e}")
                
                # Fill in cohort if not extracted but provided
                if cohort and (not extracted_data.get('cohort') or extracted_data.get('cohort') == 'null'):
                    extracted_data['cohort'] = cohort
                
                # Calculate confidence score (simple heuristic)
                confidence = self._calculate_confidence(extracted_data, text)
                
                return ExtractionResult(
                    data=extracted_data,
                    confidence_score=confidence,
                    raw_response=raw_response
                )
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Extraction attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    return ExtractionResult(
                        data={},
                        confidence_score=0.0,
                        raw_response="",
                        error=str(e)
                    )
        
        return ExtractionResult(
            data={},
            confidence_score=0.0,
            raw_response="",
            error="Max retries exceeded"
        )
    
    def extract_batch(
        self,
        texts: List[str],
        cohorts: Optional[List[str]] = None
    ) -> List[ExtractionResult]:
        """
        Extract structured data from multiple texts.
        
        Args:
            texts: List of raw texts to extract from
            cohorts: Optional list of cohort names (one per text)
            
        Returns:
            List of ExtractionResults
        """
        if cohorts is None:
            cohorts = [None] * len(texts)
        
        results = []
        for text, cohort in zip(texts, cohorts):
            result = self.extract(text, cohort)
            results.append(result)
        
        return results
    
    def _calculate_confidence(
        self,
        extracted_data: Dict[str, Any],
        original_text: str
    ) -> float:
        """
        Calculate confidence score for extraction.
        
        Args:
            extracted_data: Extracted data dictionary
            original_text: Original text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        max_score = 0.0
        
        # Name extraction
        max_score += 1.0
        if extracted_data.get('name') and len(str(extracted_data['name']).strip()) > 0:
            name = str(extracted_data['name']).lower()
            if name in original_text.lower():
                score += 1.0
        
        # Project description extraction
        max_score += 1.0
        if extracted_data.get('project_description') and len(str(extracted_data['project_description']).strip()) > 10:
            score += 1.0
        
        # Domain extraction
        max_score += 1.0
        if extracted_data.get('domains') and isinstance(extracted_data['domains'], list) and len(extracted_data['domains']) > 0:
            score += 1.0
        
        # Category extraction
        max_score += 1.0
        if extracted_data.get('category') and extracted_data['category'] != 'null':
            score += 1.0
        
        # Location extraction
        max_score += 0.5
        if extracted_data.get('location') and extracted_data['location'] != 'null':
            score += 0.5
        
        return score / max_score if max_score > 0 else 0.0

