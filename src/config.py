"""
Configuration management for source URLs, LLM prompts, and pipeline settings.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ScrapingConfig:
    """Configuration for scraping."""
    rate_limit_delay: float = 1.5
    max_retries: int = 3
    timeout: int = 30
    respect_robots_txt: bool = True
    user_agent: str = "EV-Semantic-Map-Bot/1.0"


@dataclass_json
@dataclass
class LLMConfig:
    """Configuration for LLM extraction and labeling."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    extraction_min_confidence: float = 0.3


@dataclass_json
@dataclass
class PipelineConfig:
    """Configuration for pipeline processing."""
    skip_embeddings: bool = False
    incremental_updates: bool = True
    apply_automated_labeling: bool = True
    min_confidence_threshold: float = 0.3


@dataclass_json
@dataclass
class SourceConfig:
    """Configuration for a single data source."""
    name: str
    url: str
    type: str  # 'marginal_revolution', 'mercatus', 'web'
    enabled: bool = True
    last_scraped: Optional[str] = None
    notes: Optional[str] = None


class Config:
    """
    Main configuration manager.
    
    Manages:
    - Source URLs and scraping parameters
    - LLM prompts and extraction templates
    - Pipeline settings
    """
    
    DEFAULT_CONFIG_PATH = Path('config.json')
    DEFAULT_SOURCES_PATH = Path('data/sources/sources.json')
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        sources_path: Optional[Path] = None
    ):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to main config file
            sources_path: Path to sources registry file
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.sources_path = sources_path or self.DEFAULT_SOURCES_PATH
        
        # Initialize with defaults
        self.scraping = ScrapingConfig()
        self.llm = LLMConfig()
        self.pipeline = PipelineConfig()
        self.sources: List[SourceConfig] = []
        
        # Load if exists
        if self.config_path.exists():
            self.load()
        
        if self.sources_path.exists():
            self.load_sources()
    
    def load(self):
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            data = json.load(f)
            
            if 'scraping' in data:
                self.scraping = ScrapingConfig.from_dict(data['scraping'])
            if 'llm' in data:
                self.llm = LLMConfig.from_dict(data['llm'])
            if 'pipeline' in data:
                self.pipeline = PipelineConfig.from_dict(data['pipeline'])
    
    def save(self):
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'scraping': self.scraping.to_dict(),
            'llm': self.llm.to_dict(),
            'pipeline': self.pipeline.to_dict()
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_sources(self):
        """Load sources registry from file."""
        with open(self.sources_path, 'r') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                self.sources = [SourceConfig.from_dict(s) for s in data]
            else:
                # Legacy format
                sources_list = data.get('sources', [])
                self.sources = [SourceConfig.from_dict(s) for s in sources_list]
    
    def save_sources(self):
        """Save sources registry to file."""
        self.sources_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [s.to_dict() for s in self.sources]
        
        with open(self.sources_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_enabled_sources(self) -> List[SourceConfig]:
        """Get list of enabled sources."""
        return [s for s in self.sources if s.enabled]
    
    def get_source_urls(self) -> List[str]:
        """Get list of enabled source URLs."""
        return [s.url for s in self.get_enabled_sources()]
    
    def add_source(
        self,
        name: str,
        url: str,
        source_type: str,
        enabled: bool = True,
        notes: Optional[str] = None
    ):
        """
        Add a new source.
        
        Args:
            name: Source name
            url: Source URL
            source_type: Type of source ('marginal_revolution', 'mercatus', 'web')
            enabled: Whether source is enabled
            notes: Optional notes about the source
        """
        source = SourceConfig(
            name=name,
            url=url,
            type=source_type,
            enabled=enabled,
            notes=notes
        )
        self.sources.append(source)
        self.save_sources()
    
    def update_source_last_scraped(self, url: str, timestamp: str):
        """
        Update last scraped timestamp for a source.
        
        Args:
            url: Source URL
            timestamp: Timestamp string
        """
        for source in self.sources:
            if source.url == url:
                source.last_scraped = timestamp
                self.save_sources()
                break
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'scraping': self.scraping.to_dict(),
            'llm': self.llm.to_dict(),
            'pipeline': self.pipeline.to_dict(),
            'sources': [s.to_dict() for s in self.sources]
        }
    
    @classmethod
    def create_default_config(cls, config_path: Optional[Path] = None):
        """
        Create default configuration file.
        
        Args:
            config_path: Path to save config file
        """
        config = cls(config_path=config_path)
        config.save()
        
        # Create default sources directory structure
        if config.sources_path:
            config.sources_path.parent.mkdir(parents=True, exist_ok=True)
            if not config.sources_path.exists():
                config.save_sources()
        
        return config

