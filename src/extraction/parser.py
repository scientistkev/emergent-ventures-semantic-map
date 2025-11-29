"""
Parser to validate and structure LLM extraction results.
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple
from .llm_extractor import ExtractionResult


# Valid categories
VALID_CATEGORIES = {
    'startup', 'research', 'content', 'policy',
    'career development', 'other', 'grant', 'project'
}

# Valid domain patterns (flexible matching)
DOMAIN_PATTERNS = [
    r'ai|artificial intelligence|machine learning|ml|deep learning',
    r'biotech|biotechnology|biology|life sciences',
    r'education|edtech|learning',
    r'healthcare|health|medical|medicine',
    r'hardware|robotics|engineering',
    r'energy|renewable|sustainability|climate',
    r'finance|fintech|economics',
    r'agriculture|agritech|farming',
    r'software|platform|app',
    r'research|science|physics|mathematics',
]


def parse_extraction_result(
    result: ExtractionResult,
    min_confidence: float = 0.3
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Parse and validate LLM extraction result.
    
    Args:
        result: ExtractionResult from LLM extractor
        min_confidence: Minimum confidence score to accept
        
    Returns:
        Tuple of (parsed_data, warnings)
        - parsed_data: Validated and normalized data dict, or None if invalid
        - warnings: List of warning messages
    """
    warnings = []
    
    # Check for errors
    if result.error:
        warnings.append(f"Extraction error: {result.error}")
        return None, warnings
    
    # Check confidence score
    if result.confidence_score < min_confidence:
        warnings.append(
            f"Low confidence score: {result.confidence_score:.2f} < {min_confidence}"
        )
        # Don't reject, but flag it
    
    data = result.data.copy()
    
    # Validate and normalize name
    if 'name' in data and data['name']:
        data['name'] = str(data['name']).strip()
        if len(data['name']) < 2:
            warnings.append("Name too short, may be invalid")
            data['name'] = None
    else:
        warnings.append("Missing name field")
        data['name'] = None
    
    # Validate age
    if 'age' in data and data['age'] is not None:
        try:
            age = int(data['age'])
            if age < 0 or age > 150:
                warnings.append(f"Age out of reasonable range: {age}")
                data['age'] = None
            else:
                data['age'] = age
        except (ValueError, TypeError):
            warnings.append(f"Invalid age format: {data['age']}")
            data['age'] = None
    else:
        data['age'] = None
    
    # Normalize location
    if 'location' in data and data['location']:
        location = str(data['location']).strip()
        # Clean up common variations
        location = re.sub(r'\s*/\s*.*', '', location)  # Remove secondary locations
        data['location'] = location
    else:
        data['location'] = None
    
    # Normalize education
    if 'education' in data and data['education']:
        data['education'] = str(data['education']).strip()
    else:
        data['education'] = None
    
    # Validate project_name
    if 'project_name' in data and data['project_name']:
        data['project_name'] = str(data['project_name']).strip()
    else:
        data['project_name'] = None
        warnings.append("Missing project_name")
    
    # Validate project_description
    if 'project_description' in data and data['project_description']:
        data['project_description'] = str(data['project_description']).strip()
        if len(data['project_description']) < 10:
            warnings.append("Project description too short")
    else:
        data['project_description'] = None
        warnings.append("Missing project_description")
    
    # Normalize domains
    if 'domains' in data and data['domains']:
        if isinstance(data['domains'], str):
            # Handle string representation of list
            try:
                domains = json.loads(data['domains'])
            except:
                domains = [d.strip() for d in data['domains'].split(',')]
        elif isinstance(data['domains'], list):
            domains = [str(d).strip().lower() for d in data['domains'] if d]
        else:
            domains = []
        
        # Filter empty domains
        domains = [d for d in domains if d and d != 'null' and len(d) > 1]
        
        if not domains:
            warnings.append("No valid domains extracted")
        else:
            data['domains'] = domains
    else:
        data['domains'] = []
        warnings.append("Missing domains")
    
    # Validate category
    if 'category' in data and data['category']:
        category = str(data['category']).lower().strip()
        # Normalize common variations
        category_mapping = {
            'startup': 'startup',
            'research': 'research',
            'content': 'content',
            'policy': 'policy',
            'career development': 'career development',
            'career': 'career development',
            'grant': 'grant',
            'project': 'project',
            'other': 'other'
        }
        category = category_mapping.get(category, category)
        
        if category not in VALID_CATEGORIES:
            warnings.append(f"Invalid category: {category}")
            category = 'other'
        
        data['category'] = category
    else:
        data['category'] = None
        warnings.append("Missing category")
    
    # Normalize cohort
    if 'cohort' in data and data['cohort']:
        data['cohort'] = str(data['cohort']).strip()
    else:
        data['cohort'] = None
    
    # Add confidence score
    data['_extraction_confidence'] = result.confidence_score
    data['_extraction_warnings'] = warnings
    
    # Determine if entry is valid (has minimum required fields)
    is_valid = (
        data.get('name') and
        (data.get('project_name') or data.get('project_description'))
    )
    
    if not is_valid:
        warnings.append("Entry missing required fields (name and project)")
    
    return data if is_valid else None, warnings


def validate_extracted_data(
    data: Dict[str, Any],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate extracted data against schema.
    
    Args:
        data: Extracted data dictionary
        strict: If True, require all fields; if False, only require essential fields
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    if not data.get('name'):
        errors.append("Missing required field: name")
    
    if strict:
        if not data.get('project_name'):
            errors.append("Missing required field: project_name")
        if not data.get('project_description'):
            errors.append("Missing required field: project_description")
    else:
        if not data.get('project_name') and not data.get('project_description'):
            errors.append("Missing required field: project_name or project_description")
    
    # Validate category if present
    if data.get('category') and data['category'] not in VALID_CATEGORIES:
        errors.append(f"Invalid category: {data['category']}")
    
    # Validate domains format
    if 'domains' in data and not isinstance(data['domains'], list):
        errors.append("Domains must be a list")
    
    is_valid = len(errors) == 0
    return is_valid, errors

